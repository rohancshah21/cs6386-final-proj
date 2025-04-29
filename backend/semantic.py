import argparse
import sys
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


DIET_RULES = {
    'vegan': {
        'exclude_keywords': [
            'egg', 'milk', 'butter', 'cheese', 'honey', 'yogurt',
            'cream', 'gelatin', 'meat', 'fish', 'chicken', 'beef', 'pork'
        ]
    },
    'gluten_free': {
        'exclude_keywords': [
            'wheat', 'barley', 'rye', 'flour', 'semolina',
            'malt', 'spelt', 'triticale'
        ]
    },
    'keto': {
        'exclude_keywords': [
            'sugar', 'rice', 'pasta', 'bread', 'potato',
            'corn', 'banana', 'oat', 'honey'
        ],
        'max_carbs_per_serving': 10    # grams
    }
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: Data file not found at {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df['high_level_ingredients'] = df['high_level_ingredients'].apply(
        ast.literal_eval)
    return df


def annotate_diets(df: pd.DataFrame) -> pd.DataFrame:
    def is_allowed(ings: list[str], keywords: list[str]) -> bool:
        text = ' '.join(ings).lower()
        return not any(re.search(rf'\b{kw}\b', text) for kw in keywords)

    for diet, rules in DIET_RULES.items():
        col = f'is_{diet}'
        df[col] = df['high_level_ingredients'].apply(
            lambda ings: is_allowed(ings, rules['exclude_keywords'])
        )
        if diet == 'keto' and 'max_carbs_per_serving' in rules and 'carbohydrates_g' in df:
            df[col] &= df['carbohydrates_g'] <= rules['max_carbs_per_serving']
    return df


def build_semantic_index(
    df: pd.DataFrame,
    model_name: str = 'all-MiniLM-L6-v2'
) -> tuple[SentenceTransformer, faiss.Index]:
    model = SentenceTransformer(model_name)
    docs = (
        df['name'].fillna('') + ' | ' +
        df['summary'].fillna('') + ' | ' +
        df['high_level_ingredients']
        .apply(lambda lst: ' '.join(lst) if isinstance(lst, list) else '')
    ).tolist()

    embs = model.encode(docs, show_progress_bar=True,
                        normalize_embeddings=True)
    matrix = np.vstack(embs).astype('float32')

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return model, index


def build_bm25_index(df: pd.DataFrame) -> BM25Okapi:
    docs = (
        df['name'].fillna('') + ' ' +
        df['summary'].fillna('') + ' ' +
        df['high_level_ingredients']
        .apply(lambda lst: ' '.join(lst) if isinstance(lst, list) else '')
    ).tolist()
    tokenized = [re.findall(r'\w+', doc.lower()) for doc in docs]
    return BM25Okapi(tokenized)


def semantic_search(query: str, model, index, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    scores, idx = index.search(q_emb, top_k)
    res = df.iloc[idx[0]].copy().reset_index(drop=True)
    res['semantic_score'] = scores[0]
    return res


def keyword_search(query: str, bm25: BM25Okapi, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    tokens = re.findall(r'\w+', query.lower())
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    res = df.iloc[top_idx].copy().reset_index(drop=True)
    res['keyword_score'] = scores[top_idx]
    return res


def hybrid_search(
    query: str,
    model, index,
    bm25: BM25Okapi,
    df: pd.DataFrame,
    top_k: int,
    alpha: float = 0.5
) -> pd.DataFrame:

    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    sem_scores, sem_idx = index.search(q_emb, len(df))
    sem_scores = sem_scores.flatten()

    tokens = re.findall(r'\w+', query.lower())
    kw_scores = bm25.get_scores(tokens)

    from sklearn.preprocessing import minmax_scale
    sem_n = minmax_scale(sem_scores)
    kw_n = minmax_scale(kw_scores)

    combined = alpha * sem_n + (1 - alpha) * kw_n
    top_idx = np.argsort(combined)[::-1][:top_k]
    res = df.iloc[top_idx].copy().reset_index(drop=True)
    res['semantic_score'] = sem_scores[top_idx]
    res['keyword_score'] = kw_scores[top_idx]
    res['score'] = combined[top_idx]
    return res


def main():
    base = Path(__file__).resolve().parent
    default_csv = base / 'data' / 'final_df.csv'

    p = argparse.ArgumentParser(
        description="Semantic / Keyword / Hybrid search over recipes"
    )
    p.add_argument('-d', '--data',    type=Path, default=default_csv,
                   help=f'CSV path (default: {default_csv})')
    p.add_argument('-m', '--model',   type=str,  default='all-MiniLM-L6-v2',
                   help='SBERT model name')
    p.add_argument('-k', '--top_k',   type=int,  default=5,
                   help='Number of results per query')
    p.add_argument('--diet',          choices=['all', 'vegan', 'gluten_free', 'keto'],
                   default='all', help='Dietary filter')
    p.add_argument('--method',        choices=['semantic', 'keyword', 'hybrid'],
                   default='semantic', help='Search method')
    p.add_argument('--alpha',         type=float, default=0.5,
                   help='Weight for semantic in hybrid (0–1)')
    args = p.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)

    if args.diet != 'all':
        mask = df[f'is_{args.diet}']
        df = df[mask].reset_index(drop=True)
        if df.empty:
            print(f"No recipes match diet '{args.diet}'.", file=sys.stderr)
            sys.exit(1)

    model, sem_index = build_semantic_index(df, args.model)
    bm25 = build_bm25_index(df)

    try:
        while True:
            q = input("Query> ").strip()
            if q.lower() in ('exit', 'quit'):
                break

            if args.method == 'semantic':
                results = semantic_search(q, model, sem_index, df, args.top_k)
                for i, r in results.iterrows():
                    print(
                        f"{i+1}. {r['name']}  (sem={r['semantic_score']:.4f})")

            elif args.method == 'keyword':
                results = keyword_search(q, bm25, df, args.top_k)
                for i, r in results.iterrows():
                    print(f"{i+1}. {r['name']}  (kw={r['keyword_score']:.4f})")

            else:  # hybrid
                results = hybrid_search(q, model, sem_index, bm25, df,
                                        args.top_k, alpha=args.alpha)
                for i, r in results.iterrows():
                    print(f"{i+1}. {r['name']}  (sem={r['semantic_score']:.4f}, "
                          f"kw={r['keyword_score']:.4f}, combined={r['score']:.4f})")

            print("-" * 40)
    except KeyboardInterrupt:
        pass

    print("Goodbye!")


if __name__ == '__main__':
    main()
