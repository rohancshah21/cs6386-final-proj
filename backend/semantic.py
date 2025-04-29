#!/usr/bin/env python3
import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def load_data(path):
    if not path.exists():
        print(f"ERROR: data file not found at {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df['high_level_ingredients'] = df['high_level_ingredients'].apply(
        ast.literal_eval)
    return df


def build_index(
    df,
    model_name: str = 'all-MiniLM-L6-v2'
):
    model = SentenceTransformer(model_name)
    docs = (
        df['name'].fillna('') + ' | ' +
        df['summary'].fillna('') + ' | ' +
        df['high_level_ingredients']
        .apply(lambda lst: ' '.join(lst) if isinstance(lst, list) else '')
    ).tolist()

    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = np.vstack(embeddings).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index


def semantic_search(
    query,
    model,
    index,
    df,
    top_k: int = 5
):
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(q_emb, top_k)
    results = df.iloc[indices[0]].copy().reset_index(drop=True)
    results['score'] = scores[0]
    return results


def main():
    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir / 'data' / 'final_df.csv'

    parser = argparse.ArgumentParser(
        description="Interactive semantic search over your recipe dataset"
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        default=default_csv,
        help=f'Path to your CSV file (default: {default_csv})'
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
        help='Number of top recipes to display per query'
    )
    args = parser.parse_args()

    df = load_data(args.data)
    model, index = build_index(df, args.model)

    try:
        while True:
            query = input("Search query (or 'exit' to quit): ").strip()
            if query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            results = semantic_search(query, model, index, df, args.top_k)
            print("\nTop results:")
            for i, row in results.iterrows():
                print(f"\n{i+1}. {row['name']}  (score: {row['score']:.4f})")
                print(f"   Summary: {row.get('summary', '')}")
            print("\n" + "-"*40 + "\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
