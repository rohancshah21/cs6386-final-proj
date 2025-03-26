import React, { useState } from "react";
import axios from "axios";

function RecipeSearch() {
  const [query, setQuery] = useState("");
  const [diet, setDiet] = useState("any");
  const [recipes, setRecipes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Handles the text input for user query
  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  // Handles the dropdown for dietary restriction
  const handleDietChange = (e) => {
    setDiet(e.target.value);
  };

  // Example of how you'd query your backend
  const handleSearch = async () => {
    if (!query.trim()) return; // no empty searches
    setLoading(true);
    setError("");
    try {
      // Adjust the URL to match your backend endpoint
      // Example: /api/search?query=...&diet=...
      const response = await axios.get("/api/search", {
        params: { q: query, diet: diet },
      });
      setRecipes(response.data.recipes || []);
    } catch (err) {
      console.error(err);
      setError("An error occurred while fetching recipes.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: "1rem" }}>
        <label htmlFor="query">
          Search for a recipe:
          <input
            id="query"
            type="text"
            value={query}
            onChange={handleInputChange}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>

        <label htmlFor="diet" style={{ marginLeft: "1rem" }}>
          Dietary Restriction:
          <select
            id="diet"
            value={diet}
            onChange={handleDietChange}
            style={{ marginLeft: "0.5rem" }}
          >
            <option value="any">Any</option>
            <option value="vegan">Vegan</option>
            <option value="keto">Keto</option>
            <option value="gluten-free">Gluten-Free</option>
          </select>
        </label>

        <button onClick={handleSearch} style={{ marginLeft: "1rem" }}>
          Search
        </button>
      </div>

      {loading && <p>Loading recipes...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      <ul>
        {recipes.map((recipe) => (
          <li key={recipe.id} style={{ marginBottom: "0.5rem" }}>
            <strong>{recipe.title}</strong>
            <br />
            {recipe.ingredients && (
              <em>
                Ingredients: {recipe.ingredients.join(", ")}
              </em>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default RecipeSearch;
