import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import argparse

from hybrid_search import (
    load_data, annotate_diets, load_indices,
    hybrid_search, semantic_search, keyword_search
)


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
    },
    {
        "query": "chickpea stew without dairy",
        "have": ["chickpea"],
        "avoid": ["milk", "dairy"],
        "restrictions": "vegan"
    }
]


def compute_relevance(row, case):
    score = 0.0
    total_weight = 0.0

    ingredients = [i.lower() for i in row['high_level_ingredients']]
    ingredient_text = " ".join(ingredients)
    combined_text = f"{row['name']} {row.get('summary', '')} {ingredient_text}".lower(
    )

    if case['restrictions']:
        total_weight += 1.0
        if row.get(f"is_{case['restrictions']}", False):
            score += 1.0

    if case['avoid']:
        total_weight += 1.0
        avoid_hits = sum(1 for a in case['avoid'] if a in ingredient_text)
        if avoid_hits == 0:
            score += 1.0
        else:
            penalty = avoid_hits / len(case['avoid'])
            score += max(0, 1.0 - penalty)

    if case['have']:
        total_weight += 1.0
        have_hits = sum(1 for h in case['have'] if h in ingredient_text)
        score += have_hits / len(case['have'])

    query_tokens = set(re.findall(r'\w+', case['query'].lower()))
    total_weight += 1.0
    hits = sum(1 for token in query_tokens if token in combined_text)
    score += hits / len(query_tokens) if query_tokens else 0

    return round(score / total_weight, 3) if total_weight > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path("data/final_df.csv"))
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--out', type=Path,
                        default=Path("method_comparison.png"))
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Hybrid weighting")
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

        # Keyword
        kw_res = keyword_search(q, doc_tokens, df, args.top_k)
        kw_score = np.mean([compute_relevance(r, case)
                           for _, r in kw_res.iterrows()])
        avg_keyword_scores.append(kw_score)

        # Semantic
        sem_res = semantic_search(q, model, sem_idx, df, args.top_k)
        sem_score = np.mean([compute_relevance(r, case)
                            for _, r in sem_res.iterrows()])
        avg_semantic_scores.append(sem_score)

        # Hybrid
        hyb_res = hybrid_search(q, model, sem_idx, doc_tokens,
                                df, args.top_k, args.alpha, have, avoid, restriction)
        hyb_score = np.mean([compute_relevance(r, case)
                            for _, r in hyb_res.iterrows()])
        avg_hybrid_scores.append(hyb_score)

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
    plt.ylabel("Average Relevance")
    plt.title("Top-k Relevance")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
