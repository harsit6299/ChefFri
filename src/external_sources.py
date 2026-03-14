from __future__ import annotations

import httpx

from src.schemas import RecipeRecord


def fetch_from_themealdb(query: str, timeout: float = 12.0) -> list[RecipeRecord]:
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    meals = payload.get("meals") or []
    records: list[RecipeRecord] = []

    for meal in meals:
        ingredients = []
        for i in range(1, 21):
            ing = (meal.get(f"strIngredient{i}") or "").strip()
            measure = (meal.get(f"strMeasure{i}") or "").strip()
            if ing:
                ingredients.append(f"{measure} {ing}".strip())

        instructions = [step.strip() for step in (meal.get("strInstructions") or "").split(".") if step.strip()]
        if not instructions:
            instructions = [(meal.get("strInstructions") or "").strip()] if meal.get("strInstructions") else []

        records.append(
            RecipeRecord(
                recipe_title=meal.get("strMeal") or "Unknown Recipe",
                cuisine=meal.get("strArea") or None,
                course=meal.get("strCategory") or None,
                diet=None,
                ingredients=ingredients,
                instructions=instructions,
                description=(meal.get("strInstructions") or "")[:500] or None,
                url=meal.get("strSource") or meal.get("strYoutube") or None,
                source="themealdb",
            )
        )

    return records
