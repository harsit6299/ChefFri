from __future__ import annotations

from dataclasses import dataclass

from src.llm import GroqClient
from src.retriever import RetrievalResult
from src.schemas import RecipeRecord, UserQuery


@dataclass
class ParsedIntent:
    query: str
    cuisine: str | None
    diet: str | None
    course: str | None
    max_minutes: int | None


class RouterAgent:
    def __init__(self, llm: GroqClient) -> None:
        self.llm = llm

    def parse_intent(self, user_query: UserQuery) -> ParsedIntent:
        system_prompt = (
            "Extract recipe intent into JSON with keys: query, cuisine, diet, course, max_minutes. "
            "Use null when absent. Keep query concise."
        )
        payload = self.llm.complete_json(system_prompt, user_query.text)
        return ParsedIntent(
            query=payload.get("query") or user_query.text,
            cuisine=payload.get("cuisine"),
            diet=payload.get("diet"),
            course=payload.get("course"),
            max_minutes=payload.get("max_minutes"),
        )


class RankerAgent:
    def __init__(self, llm: GroqClient) -> None:
        self.llm = llm

    def pick_best(self, intent: ParsedIntent, candidates: list[RetrievalResult]) -> tuple[RecipeRecord, float, str]:
        if not candidates:
            raise ValueError("No candidates found")

        top = candidates[:5]
        candidate_text = "\n\n".join(
            [
                f"ID:{i}\nTitle:{c.recipe.recipe_title}\nCuisine:{c.recipe.cuisine}\nDiet:{c.recipe.diet}\n"
                f"Course:{c.recipe.course}\nScore:{c.score:.4f}\n"
                f"Ingredients:{', '.join(c.recipe.ingredients[:20])}\n"
                f"Instructions:{' '.join(c.recipe.instructions[:6])}"
                for i, c in enumerate(top)
            ]
        )

        system_prompt = (
            "You are a recipe ranking agent. Return JSON with keys: best_id (int), confidence (0-1), rationale. "
            "Prefer high semantic match, then metadata match, then rating quality if available."
        )
        user_prompt = f"Intent: {intent}\n\nCandidates:\n{candidate_text}"
        decision = self.llm.complete_json(system_prompt, user_prompt)

        # The model can legitimately answer `null` for any of these keys when
        # nothing in the candidate set matches - which is exactly the case the
        # TheMealDB fallback exists to rescue. `dict.get(key, default)` does not
        # help there: the key IS present, its value is just None, so the default
        # never applies and int(None) / float(None) raise. Coalesce explicitly.
        raw_best_id = decision.get("best_id")
        raw_confidence = decision.get("confidence")
        raw_rationale = decision.get("rationale")

        try:
            best_id = int(raw_best_id) if raw_best_id is not None else 0
        except (TypeError, ValueError):
            best_id = 0
        best_id = min(max(best_id, 0), len(top) - 1)

        # A missing confidence means the model declined to judge, so claim none.
        # Deliberately NOT defaulting to the retrieval score: that is only a 0-1
        # quantity in hybrid mode, and in bm25 mode it is a raw unbounded BM25
        # sum that would read as wildly over-confident. Reporting 0.0 instead
        # lets the post-rank edge route to fallback and re-rank, which is the
        # recovery this graph was built to do.
        try:
            confidence = float(raw_confidence) if raw_confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        rationale = str(raw_rationale) if raw_rationale else "Selected highest semantic and metadata match."
        return top[best_id].recipe, confidence, rationale


class SummarizerAgent:
    def __init__(self, llm: GroqClient) -> None:
        self.llm = llm

    def summarize_steps(self, recipe: RecipeRecord, min_steps: int = 5, max_steps: int = 10) -> list[str]:
        system_prompt = (
            "You are a cooking assistant. Convert recipe instructions into 5-10 concise actionable numbered steps. "
            "Return JSON with key steps as an array of strings only."
        )
        user_prompt = (
            f"Title: {recipe.recipe_title}\n"
            f"Ingredients: {', '.join(recipe.ingredients)}\n"
            f"Raw instructions: {' '.join(recipe.instructions)}\n"
            f"Ensure step count between {min_steps} and {max_steps}."
        )
        payload = self.llm.complete_json(system_prompt, user_prompt)
        steps = payload.get("steps") or []
        steps = [str(s).strip() for s in steps if str(s).strip()]

        if len(steps) < min_steps:
            # Fallback: split source instructions when LLM under-produces.
            seed = [s.strip() for s in recipe.instructions if s.strip()]
            steps = seed[:max_steps]

        if len(steps) > max_steps:
            steps = steps[:max_steps]

        return steps
