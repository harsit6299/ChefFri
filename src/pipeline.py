from __future__ import annotations

import os

from src.agents import RankerAgent, RouterAgent, SummarizerAgent
from src.config import settings
from src.data_loader import load_recipes
from src.external_sources import fetch_from_themealdb
from src.llm import GroqClient
from src.retriever import HybridRetriever
from src.schemas import RecipeAnswer, UserQuery


class RecipeRAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever(
            settings.embedding_model,
            settings.index_dir,
            retrieval_mode=settings.retrieval_mode,
        )
        self.llm = GroqClient() if settings.groq_api_key else None

        records_path = os.path.join(settings.index_dir, "records.json")
        docs_path = os.path.join(settings.index_dir, "documents.json")
        faiss_path = os.path.join(settings.index_dir, "recipes.faiss")

        index_exists = os.path.exists(records_path) and os.path.exists(docs_path)
        if settings.retrieval_mode == "hybrid":
            index_exists = index_exists and os.path.exists(faiss_path)

        if index_exists:
            self.retriever.load()
        else:
            records = load_recipes(settings.data_path)
            self.retriever.build(records)
            self.retriever.save()

    def _ensure_agents(self) -> tuple[RouterAgent, RankerAgent, SummarizerAgent]:
        if not self.llm:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env before running query pipeline.")
        return RouterAgent(self.llm), RankerAgent(self.llm), SummarizerAgent(self.llm)

    def _fallback_if_needed(self, query: str, top_score: float) -> None:
        if not settings.use_fallback:
            return
        if top_score >= 0.55:
            return

        external_records = fetch_from_themealdb(query)
        if not external_records:
            return

        # Rebuild combined index with external cache records for this runtime.
        combined = self.retriever.records + external_records
        self.retriever.build(combined)

    def run(self, text_query: str) -> RecipeAnswer:
        router, ranker, summarizer = self._ensure_agents()

        intent = router.parse_intent(UserQuery(text=text_query))
        retrieval_query = " ".join(
            [
                intent.query,
                intent.cuisine or "",
                intent.diet or "",
                intent.course or "",
            ]
        ).strip()

        first_pass = self.retriever.retrieve(retrieval_query, top_k=settings.top_k)
        top_score = first_pass[0].score if first_pass else 0.0
        self._fallback_if_needed(intent.query, top_score)

        candidates = self.retriever.retrieve(retrieval_query, top_k=settings.top_k)
        recipe, confidence, rationale = ranker.pick_best(intent, candidates)
        steps = summarizer.summarize_steps(recipe, min_steps=5, max_steps=10)

        return RecipeAnswer(
            predicted_recipe=recipe.recipe_title,
            confidence=float(max(0.0, min(1.0, confidence))),
            summary_steps=steps,
            ingredients=recipe.ingredients,
            description=recipe.description,
            rating=recipe.rating,
            vote_count=recipe.vote_count,
            source=recipe.source,
            source_url=recipe.url,
            rationale=rationale,
        )
