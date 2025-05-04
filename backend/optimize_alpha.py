import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

from hybrid_search import (
    load_data, annotate_diets, load_indices, hybrid_search
)
from openai_eval import rate_with_llm_binary


QUERY_CASES = [
    {
        "query": "creamy pasta",
        "have": ["broccoli"],
        "avoid": ["nuts"],
        "restrictions": "vegan"
    },
    {
        "query": "gluten-free banana muffins",
        "have": ["banana"],
        "avoid": ["wheat", "flour"],
        "restrictions": "gluten_free"
    },
    {
        "query": "keto chicken salad",
        "have": ["chicken"],
        "avoid": ["bread", "sugar"],
        "restrictions": "keto"
    },
    {
        "query": "spicy tofu stir fry",
        "have": ["tofu"],
        "avoid": ["meat"],
        "restrictions": "vegan"
    },
    {
        "query": "low carb salmon dinner",
        "have": ["salmon"],
        "avoid": ["rice", "potato"],
        "restrictions": "keto"
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path,
                        default=Path("backend/data/final_df.csv"))
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--out', type=Path,
                        default=Path("alpha_relevance_plot.png"))
    args = parser.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)

    model_dir = Path(__file__).resolve().parent / 'models'
    model, sem_idx, doc_tokens = load_indices(model_dir)

    alphas = np.linspace(0.0, 1.0, 21)
    avg_scores = []

    for alpha in alphas:
        all_ratings = []
        for case in QUERY_CASES:
            res = hybrid_search(
                q=case['query'],
                model=model,
                idx=sem_idx,
                doc_tokens=doc_tokens,
                df=df,
                top_k=args.top_k,
                alpha=alpha,
                have_ingredients=case.get('have'),
                avoid_ingredients=case.get('avoid'),
                dietary_restriction=case.get('restrictions')
            )

            ratings = rate_with_llm_binary(
                query=case['query'],
                have=case.get('have'),
                avoid=case.get('avoid'),
                restrictions=case.get('restrictions'),
                results=res
            )
            res['relevance'] = res['name'].map(ratings).fillna(0).astype(float)
            all_ratings.extend(res['relevance'].tolist())

        avg_scores.append(np.mean(all_ratings))

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, avg_scores, marker='o', color='purple')
    plt.xlabel("α (semantic weight)")
    plt.ylabel("Average LLM Relevance")
    plt.title("Relevance vs α (Hybrid Weighting)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
