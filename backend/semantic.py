import argparse
import sys
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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
    def is_allowed(ingredients: list[str], keywords: list[str]) -> bool:
        text = ' '.join(ingredients).lower()
        return not any(re.search(rf'\b{kw}\b', text) for kw in keywords)

    for diet, rules in DIET_RULES.items():
        col = f'is_{diet}'
        df[col] = df['high_level_ingredients'].apply(
            lambda ings: is_allowed(ings, rules['exclude_keywords'])
        )
        if diet == 'keto' and 'max_carbs_per_serving' in rules and 'carbohydrates_g' in df:
            df[col] &= df['carbohydrates_g'] <= rules['max_carbs_per_serving']
    return df


def build_index(
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

    embeddings = model.encode(
        docs, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.vstack(embeddings).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index


def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(q_emb, top_k)
    results = df.iloc[indices[0]].copy().reset_index(drop=True)
    results['score'] = scores[0]
    return results


def main():
    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir / 'data' / 'final_df.csv'

    parser = argparse.ArgumentParser(
        description="Semantic search over recipes with dietary filtering"
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        default=default_csv,
        help=f'Path to your CSV (default: {default_csv})'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name'
    )
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=5,
        help='Number of top recipes to show per query'
    )
    parser.add_argument(
        '--diet',
        choices=['all', 'vegan', 'gluten_free', 'keto'],
        default='all',
        help='Filter recipes by a dietary restriction'
    )
    args = parser.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)

    if args.diet != 'all':
        mask = df[f'is_{args.diet}']
        df = df[mask].reset_index(drop=True)
        if df.empty:
            print(
                f"ERROR: No recipes match diet '{args.diet}'", file=sys.stderr)
            sys.exit(1)
    print(f"{len(df)} recipes after filtering.")

    model, index = build_index(df, args.model)

    try:
        while True:
            query = input("Query> ").strip()
            if query.lower() in ('exit', 'quit'):
                break
            results = semantic_search(query, model, index, df, args.top_k)
            print("\nTop results:")
            for i, row in results.iterrows():
                print(f"{i+1}. {row['name']}  (score: {row['score']:.4f})")
                if 'summary' in row:
                    print(f"   {row['summary']}")
            print("-" * 40 + "\n")
    except KeyboardInterrupt:
        pass

    print("Goodbye!")


if __name__ == "__main__":
    main()
