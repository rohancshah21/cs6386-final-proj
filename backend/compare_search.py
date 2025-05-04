from openai_eval import rate_with_llm_binary
from hybrid_search import (
    load_data, annotate_diets, load_indices,
    hybrid_search, semantic_search, keyword_search
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


QUERY_CASES = [
    {
        "query": "creamy pasta",
        "have": ["broccoli"],
        "avoid": ["nuts"],
        "restrictions": "vegan"
    },
    {
        "query": "gluten-free banana muffins",
        "have": None,
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
    },
    {
        "query": "chickpea stew without dairy",
        "have": ["chickpea"],
        "avoid": ["milk", "dairy"],
        "restrictions": "vegan"
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path,
                        default=Path("backend/data/final_df.csv"))
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--out', type=Path,
                        default=Path("method_comparison.png"))
    args = parser.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)
    model_dir = Path(__file__).resolve().parent / 'models'
    model, sem_idx, doc_tokens = load_indices(model_dir)

    avg_keyword_scores = []
    avg_semantic_scores = []
    avg_hybrid_scores = []

    for case in QUERY_CASES:
        q = case["query"]
        have = case.get("have")
        avoid = case.get("avoid")
        restriction = case.get("restrictions")

        # Keyword relevance
        kw_res = keyword_search(q, doc_tokens, df, args.top_k)
        kw_ratings = rate_with_llm_binary(q, have, avoid, restriction, kw_res)
        kw_score = np.mean([kw_ratings.get(name, 0.0)
                           for name in kw_res['name']])
        avg_keyword_scores.append(kw_score)

        # Semantic relevance
        sem_res = semantic_search(q, model, sem_idx, df, args.top_k)
        sem_ratings = rate_with_llm_binary(
            q, have, avoid, restriction, sem_res)
        sem_score = np.mean([sem_ratings.get(name, 0.0)
                            for name in sem_res['name']])
        avg_semantic_scores.append(sem_score)

        # Hybrid relevance
        hyb_res = hybrid_search(q, model, sem_idx, doc_tokens, df, args.top_k,
                                args.alpha, have, avoid, restriction)
        hyb_ratings = rate_with_llm_binary(
            q, have, avoid, restriction, hyb_res)
        hyb_score = np.mean([hyb_ratings.get(name, 0.0)
                            for name in hyb_res['name']])
        avg_hybrid_scores.append(hyb_score)

    # Plotting
    x = np.arange(len(QUERY_CASES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, avg_keyword_scores,
            width=width, label='Keyword', color='gray')
    plt.bar(x, avg_semantic_scores, width=width,
            label='Semantic', color='blue')
    plt.bar(x + width, avg_hybrid_scores, width=width,
            label='Hybrid', color='green')

    plt.xticks(x, [case["query"]
               for case in QUERY_CASES], rotation=15, ha='right')
    plt.ylim(0, 1.1)
    plt.ylabel("Average LLM Relevance")
    plt.title("Top-k Relevance of Different Search Methods (LLM-Rated)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
