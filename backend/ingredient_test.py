from openai_eval import rate_with_llm_binary
from hybrid_search import (
    load_data, annotate_diets, load_indices, hybrid_search
)
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


QUERY_CASES = [
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": None,
        "avoid": None,
        "restrictions": None,
        "label": "query only"
    },
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": ["chicken", "lettuce", "mayonnaise"],
        "avoid": None,
        "restrictions": None,
        "label": "query + 3 ingredients"
    },
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": ["chicken", "lettuce", "mayonnaise", "celery", "onion"],
        "avoid": None,
        "restrictions": None,
        "label": "query + 5 ingredients"
    },
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": ["chicken", "lettuce", "mayonnaise", "celery", "onion", "mustard", "vinegar", "dill", "parsley", "lemon juice"],
        "avoid": None,
        "restrictions": None,
        "label": "query + 10 ingredients"
    },
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": None,
        "avoid": ["mayonnaise", "apples", "mint"],
        "restrictions": None,
        "label": "query + 3 avoided ingredients"
    },
    {
        "query": "chicken salad with creamy dressing and vegetables",
        "have": ["chicken", "lettuce", "mayonnaise"],
        "avoid": ["mayonnaise", "apples", "mint"],
        "restrictions": None,
        "label": "query + 3 incl./avoided ingredients"
    },

]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path,
                        default=Path("backend/data/final_df.csv"))
    parser.add_argument('--model_dir', type=Path,
                        default=Path("backend/models"))
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--out', type=Path,
                        default=Path("ingredient_inclusion.png"))
    args = parser.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)
    model, sem_idx, doc_tokens = load_indices(args.model_dir)

    labels = []
    avg_scores = []

    for case in QUERY_CASES:
        res = hybrid_search(
            case["query"], model, sem_idx, doc_tokens,
            df, args.top_k, args.alpha,
            have_ingredients=case.get("have"),
            avoid_ingredients=case.get("avoid"),
            dietary_restriction=case.get("restrictions")
        )
        ratings = rate_with_llm_binary(
            case["query"],
            case.get("have"),
            case.get("avoid"),
            case.get("restrictions"),
            res
        )
        res['relevance'] = res['name'].map(ratings).fillna(0).astype(float)
        score = res['relevance'].mean()

        labels.append(case["label"])
        avg_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_scores, color='skyblue')
    plt.ylabel("Average LLM Relevance")
    plt.title("Effect of Ingredient Inclusion on Relevance")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
