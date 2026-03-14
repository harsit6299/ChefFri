from __future__ import annotations

import re

import pandas as pd

from src.schemas import RecipeRecord


def _split_pipe_text(value: str | float | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item and item.strip()]


def _clean_text(text: str | float | None) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_recipes(csv_path: str) -> list[RecipeRecord]:
    df = pd.read_csv(csv_path)
    records: list[RecipeRecord] = []

    for _, row in df.iterrows():
        recipe = RecipeRecord(
            recipe_title=_clean_text(row.get("recipe_title")) or "Unknown Recipe",
            cuisine=_clean_text(row.get("cuisine")) or None,
            course=_clean_text(row.get("course")) or None,
            diet=_clean_text(row.get("diet")) or None,
            prep_time=_clean_text(row.get("prep_time")) or None,
            cook_time=_clean_text(row.get("cook_time")) or None,
            ingredients=_split_pipe_text(row.get("ingredients")),
            instructions=_split_pipe_text(row.get("instructions")),
            description=_clean_text(row.get("description")) or None,
            rating=float(row.get("rating")) if pd.notna(row.get("rating")) else None,
            vote_count=int(row.get("vote_count")) if pd.notna(row.get("vote_count")) else None,
            url=_clean_text(row.get("url")) or None,
            source="local",
        )
        records.append(recipe)

    return records


def recipe_to_document(recipe: RecipeRecord) -> str:
    return "\n".join(
        [
            f"Title: {recipe.recipe_title}",
            f"Cuisine: {recipe.cuisine or 'Unknown'}",
            f"Course: {recipe.course or 'Unknown'}",
            f"Diet: {recipe.diet or 'Unknown'}",
            f"Ingredients: {', '.join(recipe.ingredients)}",
            f"Description: {recipe.description or ''}",
            f"Instructions: {' '.join(recipe.instructions)}",
            f"Rating: {recipe.rating if recipe.rating is not None else 'NA'}",
            f"Votes: {recipe.vote_count if recipe.vote_count is not None else 'NA'}",
            f"URL: {recipe.url or 'NA'}",
        ]
    )
