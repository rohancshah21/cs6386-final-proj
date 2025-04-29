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
from sklearn.preprocessing import minmax_scale

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
        'max_carbs_per_serving': 10
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
    def is_allowed(ings, keywords):
        text = ' '.join(ings).lower()
        return not any(re.search(rf'\b{kw}\b', text) for kw in keywords)
    for diet, rules in DIET_RULES.items():
        col = f'is_{diet}'
        df[col] = df['high_level_ingredients'].apply(
            lambda ings: is_allowed(ings, rules['exclude_keywords']))
        if diet == 'keto' and 'max_carbs_per_serving' in rules and 'carbohydrates_g' in df:
            df[col] &= df['carbohydrates_g'] <= rules['max_carbs_per_serving']
    return df


def build_semantic_index(df, model_name):
    model = SentenceTransformer(model_name)
    docs = (df['name'].fillna('') + ' | ' + df['summary'].fillna('') + ' | ' +
            df['high_level_ingredients'].apply(lambda lst: ' '.join(lst)))
    embs = model.encode(docs.tolist(), show_progress_bar=True,
                        normalize_embeddings=True)
    mat = np.vstack(embs).astype('float32')
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return model, idx


def build_bm25_index(df):
    docs = (df['name'].fillna('') + ' | ' + df['summary'].fillna('') + ' | ' +
            df['high_level_ingredients'].apply(lambda lst: ' '.join(lst)))
    tokenized = [re.findall(r'\w+', d.lower()) for d in docs]
    return BM25Okapi(tokenized)


def semantic_search(q, model, idx, df, top_k):
    q_emb = model.encode([q], normalize_embeddings=True).astype('float32')
    scores, ids = idx.search(q_emb, top_k)
    res = df.iloc[ids[0]].copy().reset_index(drop=True)
    res['semantic_score'] = scores[0]
    return res


def keyword_search(q, bm25, df, top_k):
    tokens = re.findall(r'\w+', q.lower())
    scores = bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:top_k]
    res = df.iloc[top].copy().reset_index(drop=True)
    res['keyword_score'] = scores[top]
    return res


def hybrid_search(q, model, idx, bm25, df, top_k, alpha):
    q_emb = model.encode([q], normalize_embeddings=True).astype('float32')
    sem_scores, sem_ids = idx.search(q_emb, len(df))
    sem = sem_scores.flatten()
    tokens = re.findall(r'\w+', q.lower())
    kw = bm25.get_scores(tokens)
    sem_n = minmax_scale(sem)
    kw_n = minmax_scale(kw)
    comb = alpha * sem_n + (1-alpha) * kw_n
    top = np.argsort(comb)[::-1][:top_k]
    res = df.iloc[top].copy().reset_index(drop=True)
    res['semantic_score'] = sem[top]
    res['keyword_score'] = kw[top]
    res['score'] = comb[top]
    return res


def print_results(res, method):
    for i, r in res.iterrows():
        line = f"{i+1}. {r['name']}"
        if method == 'semantic':
            line += f"  (sem={r['semantic_score']:.3f})"
        elif method == 'keyword':
            line += f"  (kw={r['keyword_score']:.3f})"
        else:
            line += f"  (sem={r['semantic_score']:.3f}, kw={r['keyword_score']:.3f}, combined={r['score']:.3f})"
        print(line)

        print("Directions:")
        print(r['directions'], "\n")
        print("-"*50)


def main():
    base = Path(__file__).resolve().parent
    default_csv = base / 'data' / 'final_df.csv'

    p = argparse.ArgumentParser(description="Search recipes")
    p.add_argument('-d', '--data', type=Path,
                   default=default_csv, help='CSV path')
    p.add_argument('-m', '--model', type=str,
                   default='all-MiniLM-L6-v2', help='SBERT model')
    p.add_argument('-k', '--top_k', type=int, default=5, help='Top K')
    p.add_argument(
        '--diet', choices=['all', 'vegan', 'gluten_free', 'keto'], default='all')
    p.add_argument(
        '--method', choices=['semantic', 'keyword', 'hybrid'], default='semantic')
    p.add_argument('--alpha', type=float, default=0.5, help='Hybrid weight')
    p.add_argument('query', nargs='*', help='Optional query (quoted)')
    args = p.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)
    if args.diet != 'all':
        df = df[df[f'is_{args.diet}']].reset_index(drop=True)

    model, sem_idx = build_semantic_index(df, args.model)
    bm25 = build_bm25_index(df)

    if args.query:
        q = " ".join(args.query)
        if args.method == 'semantic':
            res = semantic_search(q, model, sem_idx, df, args.top_k)
        elif args.method == 'keyword':
            res = keyword_search(q, bm25, df, args.top_k)
        else:
            res = hybrid_search(q, model, sem_idx, bm25,
                                df, args.top_k, args.alpha)
        print_results(res, args.method)
        return

    print("Enter queries (type exit/quit to stop):")
    while True:
        try:
            q = input("Query> ").strip()
            if q.lower() in ('exit', 'quit'):
                break
            if args.method == 'semantic':
                res = semantic_search(q, model, sem_idx, df, args.top_k)
            elif args.method == 'keyword':
                res = keyword_search(q, bm25, df, args.top_k)
            else:
                res = hybrid_search(q, model, sem_idx, bm25,
                                    df, args.top_k, args.alpha)
            print_results(res, args.method)
        except (KeyboardInterrupt, EOFError):
            break

    print("Goodbye!")


if __name__ == '__main__':
    main()
