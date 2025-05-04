from pathlib import Path
import openai
import os
import re
import pandas as pd
from hybrid_search import (
    load_data, annotate_diets,
    build_semantic_index, build_jaccard_index,
    semantic_search, keyword_search, hybrid_search, print_results
)

# --- OpenAI Binary Relevance Rating ---
def rate_with_llm_binary(query: str, results: pd.DataFrame) -> dict[str,int]:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # build prompt
    prompt = [
        f"You are a recipe relevance evaluator. The user asked: '{query}'\n",
        "Here are 5 candidate recipes:\n"
    ]
    for i, r in results.iterrows():
        prompt.append(f"{i+1}. {r['name']} - Ingredients: {' '.join(r['high_level_ingredients'])}\n")
    prompt.append("\nFor each recipe, return 1 if it is relevant to the query, or 0 if not, formatted as a JSON list of objects with 'name' and 'rating'.\n")
    content = ''.join(prompt)

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", temperature=0.0,
        messages=[
            {"role":"system","content":"You rate recipe relevance with 1 or 0."},
            {"role":"user","content":content}
        ]
    )
    text = resp.choices[0].message.content
    m = re.search(r"\[.*\]", text, re.DOTALL)
    ratings = {}
    if m:
        import json
        for obj in json.loads(m.group(0)):
            ratings[obj['name']] = obj['rating']
    return ratings


def evaluate_queries(
    data_path: str,
    model_name: str,
    query_cases: list[dict]
) -> None:

    df = load_data(Path(data_path))
    df = annotate_diets(df)

    # Build indices
    model, sem_idx = build_semantic_index(df, model_name)
    doc_sets = build_jaccard_index(df)

    # Iterate over each test case
    for case in query_cases:
        q = case['query']
        have = case.get('have')
        avoid = case.get('avoid')
        restrictions = case.get('restrictions')
        top_k = case.get('top_k', 5)
        alpha = case.get('alpha', 0.5)
        method = case.get('method', 'hybrid')

        # Perform search
        if method == 'semantic':
            res = semantic_search(q, model, sem_idx, df, top_k)
        elif method == 'keyword':
            res = keyword_search(q, doc_sets, df, top_k)
        else:
            res = hybrid_search(q, model, sem_idx, doc_sets, df, top_k, alpha)

        # Rate with OpenAI
        ratings = rate_with_llm_binary(q, res)
        res['relevance'] = res['name'].map(ratings).fillna(0).astype(int)

        # Print results
        print(f"\n=== Query: '{q}' | method={method} | have={have} | avoid={avoid} | restrictions={restrictions} ===")
        print_results(res, method)


if __name__ == '__main__':
    # Example queries to automate evaluation
    example_queries = [
        {
            'query': 'creamy pasta',
            'method': 'keyword',
            'top_k': 5,
            'have': ['broccoli'],
            'avoid': ['nuts'],
            'restrictions': ['vegan']
        },
        {
            'query': 'fried chicken',
            'method': 'keyword',
            'top_k': 5,
            'have': None,
            'avoid': ['wheat'],
            'restrictions': ['gluten_free']
        }
    ]
    data_csv = Path(__file__).resolve().parent / 'data' / 'final_df.csv'
    evaluate_queries(str(data_csv), 'all-MiniLM-L6-v2', example_queries)
