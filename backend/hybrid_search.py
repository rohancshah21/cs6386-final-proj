import argparse
import sys
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import minmax_scale
import pickle

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
    def is_allowed(ings, keywords):
        text = ' '.join(ings).lower()
        return not any(re.search(rf'\b{kw}\b', text) for kw in keywords)
    for diet, rules in DIET_RULES.items():
        col = f'is_{diet}'
        df[col] = df['high_level_ingredients'].apply(
            lambda ings: is_allowed(ings, rules['exclude_keywords']))
        if diet == 'keto' and 'max_carbs_per_serving' in rules and 'carbohydrates_g' in df:
            df[col] &= df['carbohydrates_g'] <= rules['max_carbs_per_serving']
    return df


def build_semantic_index(df, model_name):
    model = SentenceTransformer(model_name)
    docs = (df['name'].fillna('') + ' | ' + df['summary'].fillna('') + ' | ' +
            df['high_level_ingredients'].apply(lambda lst: ' '.join(lst)))
    embs = model.encode(docs.tolist(), show_progress_bar=True,
                        normalize_embeddings=True)
    mat = np.vstack(embs).astype('float32')
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return model, idx


def build_jaccard_index(df):
    docs = (df['name'].fillna('') + ' | ' + df['summary'].fillna('') + ' | ' +
            df['high_level_ingredients'].apply(lambda lst: ' '.join(lst)))
    tokenized = [set(re.findall(r'\w+', d.lower())) for d in docs]
    return tokenized


def semantic_search(q, model, idx, df, top_k):
    q_emb = model.encode([q], normalize_embeddings=True).astype('float32')
    scores, ids = idx.search(q_emb, top_k)
    res = df.iloc[ids[0]].copy().reset_index(drop=True)
    res['semantic_score'] = scores[0]
    return res


def jaccard_similarity(query_tokens, doc_tokens):
    intersection = len(query_tokens.intersection(doc_tokens))
    union = len(query_tokens.union(doc_tokens))
    return intersection / union if union > 0 else 0


def keyword_search(q, doc_tokens, df, top_k):
    query_tokens = set(re.findall(r'\w+', q.lower()))
    scores = [jaccard_similarity(query_tokens, doc) for doc in doc_tokens]
    top = np.argsort(scores)[::-1][:top_k]
    res = df.iloc[top].copy().reset_index(drop=True)
    res['keyword_score'] = [scores[i] for i in top]
    return res


def hybrid_search(q, model, idx, doc_tokens, df, top_k, alpha, have_ingredients=None, avoid_ingredients=None, dietary_restrictions=None):
    # First, filter recipes
    filtered_df, ingredient_scores = filter_recipes(df, have_ingredients, avoid_ingredients, dietary_restrictions)
    if filtered_df.empty:
        return filtered_df
    
    # Semantic search
    q_emb = model.encode([q], normalize_embeddings=True).astype('float32')
    sem_scores, sem_ids = idx.search(q_emb, len(filtered_df))
    sem = sem_scores.flatten()
    
    # Keyword search
    query_tokens = set(re.findall(r'\w+', q.lower()))
    filtered_tokens = [doc_tokens[i] for i in filtered_df.index]
    kw = [jaccard_similarity(query_tokens, doc) for doc in filtered_tokens]
    
    # Normalize and combine scores
    sem_n = minmax_scale(sem)
    kw_n = minmax_scale(kw)
    ing_n = minmax_scale(ingredient_scores)
    
    comb = 0.4 * sem_n + 0.3 * kw_n + 0.3 * ing_n
    top = np.argsort(comb)[::-1][:top_k]
    
    res = filtered_df.iloc[top].copy().reset_index(drop=True)
    res['semantic_score'] = sem[top]
    res['keyword_score'] = kw[top]
    res['ingredient_score'] = ing_n[top]
    res['score'] = comb[top]
    return res


def check_ingredients_match(recipe_ingredients, have_ingredients, avoid_ingredients):
    recipe_ingredients_lower = [ing.lower() for ing in recipe_ingredients]
    have_ingredients_lower = [ing.lower() for ing in have_ingredients]
    avoid_ingredients_lower = [ing.lower() for ing in avoid_ingredients]
    
    # Check if recipe contains any ingredients to avoid
    if any(ing in recipe_ingredients_lower for ing in avoid_ingredients_lower):
        return False, 0
    
    # empty recipe ingredients case
    if len(recipe_ingredients) == 0:
        return True, 0
    
    # Calculate match score based on available ingredients
    matching_ingredients = sum(1 for ing in have_ingredients_lower if any(ing in recipe_ing for recipe_ing in recipe_ingredients_lower))
    match_score = matching_ingredients / len(recipe_ingredients)
    
    return True, match_score


def filter_recipes(df, have_ingredients=None, avoid_ingredients=None, dietary_restriction=None):
    if not any([have_ingredients, avoid_ingredients, dietary_restriction]):
        print("No filtering criteria provided. Returning all recipes.")
        return df, np.ones(len(df))
    
    filtered_indices = []
    ingredient_scores = []
    
    for idx, row in df.iterrows():
        valid = True
        score = 1.0
        
        recipe_ingredients = row['high_level_ingredients']

        # Check dietary restrictions
        if dietary_restriction and not row[f'is_{dietary_restriction}']:
            valid = False
            score = 0.0
            
        # Check ingredients
        if valid and (have_ingredients or avoid_ingredients):
            valid, ing_score = check_ingredients_match(
                recipe_ingredients,
                have_ingredients or [],
                avoid_ingredients or []
            )
            score = ing_score
        
        if valid:
            filtered_indices.append(idx)
            ingredient_scores.append(score)
    
    if not filtered_indices:
        return pd.DataFrame(), np.array([])
    
    return df.iloc[filtered_indices], np.array(ingredient_scores)


def save_indices(model, sem_idx, doc_tokens, output_dir):
    """Save search indices to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save semantic index
    faiss.write_index(sem_idx, str(output_dir / "semantic.index"))
    
    # Save document tokens
    with open(output_dir / "doc_tokens.pkl", "wb") as f:
        pickle.dump(doc_tokens, f)
    
    # Save model
    model.save(str(output_dir / "sbert_model"))
    
def load_indices(model_dir):
    """Load search indices from files"""
    model_dir = Path(model_dir)
    
    # Load semantic model and index
    model = SentenceTransformer(str(model_dir / "sbert_model"))
    sem_idx = faiss.read_index(str(model_dir / "semantic.index"))
    
    # Load document tokens
    with open(model_dir / "doc_tokens.pkl", "rb") as f:
        doc_tokens = pickle.load(f)
        
    return model, sem_idx, doc_tokens

def print_results(res, method):
    for i, r in res.iterrows():
        line = f"{i+1}. {r['name']}"
        if method == 'semantic':
            line += f"  (sem={r['semantic_score']:.3f})"
        elif method == 'keyword':
            line += f"  (kw={r['keyword_score']:.3f})"
        else:
            line += f"  (sem={r['semantic_score']:.3f}, kw={r['keyword_score']:.3f}, combined={r['score']:.3f})"
        print(line)

        print("Directions:")
        print(r['directions'], "\n")
        print("-"*50)


def main():
    base = Path(__file__).resolve().parent
    default_csv = base / 'data' / 'final_df.csv'
    model_dir = base / 'models'

    p = argparse.ArgumentParser(description="Search recipes")
    p.add_argument('-d', '--data', type=Path,
                   default=default_csv, help='CSV path')
    p.add_argument('-m', '--model', type=str,
                   default='all-MiniLM-L6-v2', help='SBERT model')
    p.add_argument('-k', '--top_k', type=int, default=5, help='Top K')
    p.add_argument(
        '--diet', choices=['vegan', 'gluten_free', 'keto', 'none'], default='none')
    p.add_argument(
        '--have', nargs='*', help='Ingredients to include (quoted)')
    p.add_argument( 
        '--avoid', nargs='*', help='Ingredients to exclude (quoted)')
    p.add_argument(
        '--method', choices=['semantic', 'keyword', 'hybrid'], default='hybrid')
    p.add_argument('--alpha', type=float, default=0.5, help='Hybrid weight')
    p.add_argument('--build', action='store_true', help='Build and save indices')
    p.add_argument('query', nargs='*', help='Optional query (quoted)')
    args = p.parse_args()

    df = load_data(args.data)
    df = annotate_diets(df)

    if args.build:
        print("Building indices...")
        model, sem_idx = build_semantic_index(df, args.model)
        doc_tokens = build_jaccard_index(df)
        save_indices(model, sem_idx, doc_tokens, model_dir)
        print(f"Indices saved to {model_dir}")
        return
    
    try:
        print("Loading pre-built indices...")
        model, sem_idx, doc_tokens = load_indices(model_dir)
    except (FileNotFoundError, OSError):
        print("Pre-built indices not found. Run with --build first.")
        sys.exit(1)

    if args.query:
        q = " ".join(args.query)
        if args.method == 'semantic':
            res = semantic_search(q, model, sem_idx, df, args.top_k)
        elif args.method == 'keyword':
            res = keyword_search(q, doc_tokens, df, args.top_k)  
        else:
            res = hybrid_search(q, model, sem_idx, doc_tokens, 
                              df, args.top_k, args.alpha)
        print_results(res, args.method)
        return

    print("Enter queries (type exit/quit to stop):")
    while True:
        try:
            q = input("Query> ").strip()
            if q.lower() in ('exit', 'quit'):
                break
            if args.method == 'semantic':
                res = semantic_search(q, model, sem_idx, df, args.top_k)
            elif args.method == 'keyword':
                res = keyword_search(q, doc_tokens, df, args.top_k)  
            else:
                res = hybrid_search(q, model, sem_idx, doc_tokens, 
                                    df, args.top_k, args.alpha)
            print_results(res, args.method)
        except (KeyboardInterrupt, EOFError):
            break

    print("Goodbye!")


if __name__ == '__main__':
    main()
