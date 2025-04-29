import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale

from hybrid_search import (
    load_data, annotate_diets,
    build_semantic_index, build_bm25_index,
    hybrid_search
)

TEST_CASES = [
    {
        "query":   "creamy vegan pasta without nuts",
        "include": ["pasta", "creamy"],
        "exclude": ["nut", "almond", "peanut"],
        "diet":    "vegan"
    },
    {
        "query":   "gluten free banana muffins",
        "include": ["banana", "muffin"],
        "exclude": ["wheat", "flour"],
        "diet":    "gluten_free"
    },
    {
        "query":   "keto chicken salad",
        "include": ["chicken", "salad"],
        "exclude": ["sugar", "bread"],
        "diet":    "keto"
    },
]


def relevance(r, case):
    text = " ".join(r['high_level_ingredients']).lower()
    inc_hits = sum(1 for w in case['include'] if w in text)
    inc_score = inc_hits / len(case['include'])
    exc_hits = sum(1 for w in case['exclude'] if w in text)
    exc_penalty = 1.0 if exc_hits > 0 else 0.0
    diet_flag = 1.0
    if case.get('diet'):
        diet_flag = 1.0 if r[f"is_{case['diet']}"] else 0.0
    return inc_score * (1 - exc_penalty) * diet_flag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data',   type=Path,
                   default=Path('backend/data/final_df.csv'))
    p.add_argument('-k', '--top_k',  type=int, default=5)
    p.add_argument('-o', '--out',    type=Path,
                   default=Path('alpha_dynamic.png'))
    args = p.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)
    model, sem_idx = build_semantic_index(df, 'all-MiniLM-L6-v2')
    bm25 = build_bm25_index(df)

    alphas = np.linspace(0.1, 1, 21)
    avg_rels = []

    for α in alphas:
        rels = []
        for case in TEST_CASES:
            res = hybrid_search(
                case['query'], model, sem_idx, bm25, df, args.top_k, α
            )
            rels += [relevance(res.iloc[i], case) for i in range(len(res))]
        avg_rels.append(np.mean(rels))

    plt.plot(alphas, avg_rels, marker='o')
    plt.xlabel('alpha (semantic weight)')
    plt.ylabel(f'Avg relevance@{args.top_k}')
    plt.title('Dynamic Ground‐Truth Relevance vs α')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == '__main__':
    main()
