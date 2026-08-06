from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecipeRecord(BaseModel):
    recipe_title: str
    cuisine: str | None = None
    course: str | None = None
    diet: str | None = None
    prep_time: str | None = None
    cook_time: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    description: str | None = None
    rating: float | None = None
    vote_count: int | None = None
    url: str | None = None
    source: str = "local"


class UserQuery(BaseModel):
    text: str
    constraints: dict[str, Any] = Field(default_factory=dict)


class RecipeAnswer(BaseModel):
    predicted_recipe: str
    confidence: float
    summary_steps: list[str]
    ingredients: list[str]
    description: str | None = None
    rating: float | None = None
    vote_count: int | None = None
    source: str
    source_url: str | None = None
    rationale: str
    # True when the graph routed through the TheMealDB fallback node for this
    # query, on either the weak-retrieval or the weak-ranking-confidence edge.
    used_fallback: bool = False
