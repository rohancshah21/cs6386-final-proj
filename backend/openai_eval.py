from pathlib import Path
from openai import OpenAI
import re
import pandas as pd
import sys
from hybrid_search import (
    load_data, annotate_diets,
    load_indices, hybrid_search, print_results
)

_client = OpenAI(api_key="")


# --- OpenAI Binary Relevance Rating ---
def rate_with_llm_binary(query: str, have: list[str], avoid: list[str], restrictions: str, results: pd.DataFrame) -> dict:
    # build prompt
    prompt = [
        f"You are a recipe relevance evaluator. The user asked: for '{query}'\n",
        "Rate each recipe's relevance on a scale:\n",
        "1.0 - Highly relevant and matches all requirements\n",
        "0.5 - Somewhat relevant or matches some requirements\n",
        "0.0 - Not relevant at ALL or violates requirements\n\n"
    ]
    if have:
        prompt.append(
            f"User has the following ingredients: {', '.join(have)}\n")
    if avoid:
        prompt.append(
            f"User wants to avoid the following ingredients: {', '.join(avoid)}\n")
    if restrictions:
        prompt.append(f"User has dietary restrictions: {restrictions}\n")
    prompt.append("Here are the top 5 recipes:\n")

    for i, r in results.iterrows():
        prompt.append(
            f"{i+1}. {r['name']} - Ingredients: {' '.join(r['high_level_ingredients'])}\n Description: {r['summary']}\n")
    prompt.append(
        "\nFor each recipe, rate relevance as 1.0, 0.5, or 0.0, formatted as a JSON list of objects with 'name' and 'rating'.\n")
    content = ''.join(prompt)

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You rate recipe relevance with 1.0, 0.5, or 0.0."},
            {"role": "user",   "content": content}
        ]
    )
    text = resp.choices[0].message.content
    m = re.search(r"\[.*\]", text, re.DOTALL)
    ratings = {}
    if m:
        import json
        for obj in json.loads(m.group(0)):
            ratings[obj['name']] = float(obj['rating'])
    return ratings


def evaluate_queries(
    data_path: str,
    model_name: str,
    query_cases: list[dict]
) -> None:

    df = load_data(Path(data_path))
    df = annotate_diets(df)

    model_dir = Path(__file__).resolve().parent / 'models'

    try:
        print("Loading pre-built indices...")
        model, sem_idx, doc_tokens = load_indices(model_dir)
    except (FileNotFoundError, OSError):
        print("Pre-built indices not found. Run with --build first.")
        sys.exit(1)

    # Iterate over each test case
    for case in query_cases:
        q = case['query']
        have = case.get('have')
        avoid = case.get('avoid')
        restrictions = case.get('restrictions')
        top_k = case.get('top_k', 5)
        alpha = case.get('alpha', 0.5)
        method = case.get('method', 'hybrid')

        res = hybrid_search(q, model, sem_idx, doc_tokens,
                            df, top_k, alpha, have, avoid, restrictions)

        # Rate with OpenAI
        ratings = rate_with_llm_binary(q, have, avoid, restrictions, res)
        res['relevance'] = res['name'].map(ratings).fillna(0).astype(float)

        # Print results with relevance indicators
        print(
            f"\n=== Query: '{q}' | method={method} | have={have} | avoid={avoid} | restrictions={restrictions} ===")

        for i, r in res.iterrows():
            print(
                f"{i+1}. {r['name']} - Ingredients: {' '.join(r['high_level_ingredients'])}\n")
            relevance = r['relevance']
            indicator = "✓✓" if relevance == 1.0 else "✓" if relevance == 0.5 else "✗"
            print(f"Relevance rating: {relevance:.1f} {indicator}")
            print("-" * 50)


if __name__ == '__main__':
    # Example queries to automate evaluation
    example_queries = [
        {
            'query': 'creamy pasta',
            'top_k': 5,
            'alpha': 1,
            'have': ['broccoli'],
            'avoid': ['nuts'],
            'restrictions': 'vegan'
        },
        {
            'query': 'fried chicken',
            'top_k': 5,
            'alpha': 1,
            'have': None,
            'avoid': ['wheat'],
            'restrictions': 'gluten_free'
        },
        {
            'query': 'chickpea stew',
            'top_k': 5,
            'alpha': 1,
            'have': ['chickpea'],
            'avoid': None,
            'restrictions': None
        },
        {
            'query': 'quick breakfast smoothie',
            'top_k': 5,
            'alpha': 1,
            'have': ['banana', 'yogurt'],
            'avoid': ['peanut'],
            'restrictions': None
        },
        {
            'query': 'spicy tofu dinner',
            'top_k': 5,
            'alpha': 1,
            'have': ['tofu'],
            'avoid': ['fish sauce'],
            'restrictions': 'vegan'
        },
        {
            'query': 'low carb dinner',
            'top_k': 5,
            'alpha': 1,
            'have': ['chicken'],
            'avoid': ['potato', 'rice'],
            'restrictions': 'keto'
        },
        {
            'query': 'mediterranean salad',
            'top_k': 5,
            'alpha': 1,
            'have': ['cucumber', 'tomato'],
            'avoid': None,
            'restrictions': None
        },
        {
            'query': 'healthy acai bowl',
            'top_k': 5,
            'alpha': 1,
            'have': ['berries'],
            'avoid': ['chocolate'],
            'restrictions': None
        },
        {
            'query': 'asian noodle soup',
            'top_k': 5,
            'alpha': 1,
            'have': None,
            'avoid': None,
            'restrictions': None
        },
        {
            'query': 'party appetizers',
            'top_k': 5,
            'alpha': 1,
            'have': ['cheese'],
            'avoid': ['nuts', 'shellfish'],
            'restrictions': None
        }
    ]
    data_csv = Path(__file__).resolve().parent / 'data' / 'final_df.csv'
    evaluate_queries(str(data_csv), 'all-MiniLM-L6-v2', example_queries)
