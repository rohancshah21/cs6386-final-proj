import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from hybrid_search import (
    load_data, annotate_diets, load_indices, hybrid_search
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
                        default=Path("alpha_relevance_plot.png"))
    args = parser.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)

    model_dir = Path(__file__).resolve().parent / 'models'
    model, sem_idx, doc_tokens = load_indices(model_dir)

    alphas = np.linspace(0.0, 1.0, 21)
    avg_scores = []

    for alpha in alphas:
        relevance_scores = []
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
            relevance_scores.extend([
                compute_relevance(row, case) for _, row in res.iterrows()
            ])
        avg = np.mean(relevance_scores)
        avg_scores.append(avg)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, avg_scores, marker='o')
    plt.xlabel("Alpha (semantic weight)")
    plt.ylabel("Average Relevance")
    plt.title("Relevance vs α")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
