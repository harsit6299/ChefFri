from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents import ParsedIntent, RankerAgent, RouterAgent, SummarizerAgent
from src.config import settings
from src.data_loader import load_recipes
from src.external_sources import fetch_from_themealdb
from src.llm import GroqClient
from src.retriever import HybridRetriever, RetrievalResult
from src.schemas import RecipeAnswer, RecipeRecord, UserQuery

# Retrieval scores are NOT on the same scale in both retrieval modes, so the
# pre-rank fallback edge cannot use one threshold for both.
#
# hybrid: score = 0.7 * cosine_similarity + 0.3 * minmax(bm25), so it lands in
# 0-1 and the cosine term makes its magnitude a meaningful absolute measure of
# "how close is this really".
#
# bm25: score is the raw, unnormalized Okapi BM25 sum, which is unbounded and
# scales with query length and term rarity rather than with match quality.
# Measured against this corpus, every query with any lexical overlap lands in
# roughly 15-20 - including deliberate nonsense like "authentic klingon
# bloodwine stew" (15.28, matched on "stew") - while a query with no overlap
# at all scores exactly 0. There is no middle band to threshold on, so in bm25
# mode this edge can only honestly detect the degenerate total-miss case.
# Actual quality filtering in bm25 mode is carried by the post-rank edge,
# where the LLM's confidence is a real 0-1 judgement.
HYBRID_SCORE_THRESHOLD = 0.55
BM25_SCORE_THRESHOLD = 1.0

# The ranking agent always reports confidence in 0-1, so this one threshold is
# valid regardless of retrieval mode.
RANK_CONFIDENCE_THRESHOLD = 0.55

# A single fallback attempt per query is enough to demonstrate real recovery
# behaviour without letting a bad query loop forever against TheMealDB.
MAX_FALLBACK_ATTEMPTS = 1


def retrieval_score_threshold() -> float:
    """Fallback threshold for the current retrieval mode. See the notes above."""
    return BM25_SCORE_THRESHOLD if settings.retrieval_mode == "bm25" else HYBRID_SCORE_THRESHOLD


class GraphState(TypedDict, total=False):
    """Shared state threaded through every node in the LangGraph run.

    Each node only returns the keys it changes; LangGraph merges them into
    this dict, so nodes stay small and don't need to know about the whole
    pipeline shape.
    """

    text_query: str
    intent: ParsedIntent
    retrieval_query: str
    candidates: list[RetrievalResult]
    top_score: float
    used_fallback: bool
    fallback_attempts: int
    recipe: RecipeRecord
    confidence: float
    rationale: str
    steps: list[str]


class RecipeAgenticPipeline:
    """LangGraph-orchestrated version of the recipe RAG pipeline.

    `pipeline.py` runs router -> retrieve -> (maybe fallback) -> rank ->
    summarize as a fixed sequence written directly in Python: the fallback
    decision is the only branch, and it is made once, before ranking even
    happens.

    This version expresses the same agents as nodes in a graph and lets two
    independent conditional edges decide the path at runtime:

    1. After retrieval, if the raw hybrid score is weak, route to fallback
       before ever asking the ranking agent to choose.
    2. After ranking, if the LLM's own confidence in its pick is weak -
       even when the initial retrieval score looked fine - route to
       fallback anyway and re-rank once fresh data is merged in.

    Both edges share one `fallback_attempts` counter gated by
    MAX_FALLBACK_ATTEMPTS, so the graph can react to either signal without
    ever looping more than once.
    """

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

        self.router: RouterAgent | None = None
        self.ranker: RankerAgent | None = None
        self.summarizer: SummarizerAgent | None = None
        self.graph = self._build_graph()

    def _ensure_agents(self) -> None:
        if not self.llm:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env before running query pipeline.")
        if self.router is None:
            self.router = RouterAgent(self.llm)
            self.ranker = RankerAgent(self.llm)
            self.summarizer = SummarizerAgent(self.llm)

    # ---- Graph nodes --------------------------------------------------
    # Each node takes the current state and returns only the keys it adds
    # or changes; LangGraph merges the return value into GraphState.

    def _route_node(self, state: GraphState) -> dict:
        self._ensure_agents()
        intent = self.router.parse_intent(UserQuery(text=state["text_query"]))
        retrieval_query = " ".join(
            [intent.query, intent.cuisine or "", intent.diet or "", intent.course or ""]
        ).strip()
        return {"intent": intent, "retrieval_query": retrieval_query}

    def _retrieve_node(self, state: GraphState) -> dict:
        candidates = self.retriever.retrieve(state["retrieval_query"], top_k=settings.top_k)
        top_score = candidates[0].score if candidates else 0.0
        return {"candidates": candidates, "top_score": top_score}

    def _fallback_node(self, state: GraphState) -> dict:
        attempts = state.get("fallback_attempts", 0) + 1

        # The fallback is a best-effort enrichment, not a hard dependency: if
        # TheMealDB is unreachable or returns an error, the graph should still
        # go on to rank and summarize whatever local candidates it already has
        # rather than failing the whole query.
        try:
            external_records = fetch_from_themealdb(state["intent"].query)
        except Exception as exc:  # network error, non-2xx, malformed payload
            print(f"[fallback] TheMealDB lookup failed, continuing with local candidates: {exc}")
            return {"used_fallback": True, "fallback_attempts": attempts}

        if external_records:
            self.retriever.add_records(external_records)

        return {"used_fallback": True, "fallback_attempts": attempts}

    def _rank_node(self, state: GraphState) -> dict:
        recipe, confidence, rationale = self.ranker.pick_best(state["intent"], state["candidates"])
        return {"recipe": recipe, "confidence": confidence, "rationale": rationale}

    def _summarize_node(self, state: GraphState) -> dict:
        steps = self.summarizer.summarize_steps(state["recipe"], min_steps=5, max_steps=10)
        return {"steps": steps}

    # ---- Conditional routers -------------------------------------------

    @staticmethod
    def _decide_after_retrieve(state: GraphState) -> str:
        if (
            settings.use_fallback
            and state.get("top_score", 0.0) < retrieval_score_threshold()
            and not state.get("used_fallback")
            and state.get("fallback_attempts", 0) < MAX_FALLBACK_ATTEMPTS
        ):
            return "fallback"
        return "rank"

    @staticmethod
    def _decide_after_rank(state: GraphState) -> str:
        low_confidence = state.get("confidence", 1.0) < RANK_CONFIDENCE_THRESHOLD
        can_retry = (
            settings.use_fallback
            and not state.get("used_fallback")
            and state.get("fallback_attempts", 0) < MAX_FALLBACK_ATTEMPTS
        )
        if low_confidence and can_retry:
            return "fallback"
        return "summarize"

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("route", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("fallback", self._fallback_node)
        graph.add_node("rank", self._rank_node)
        graph.add_node("summarize", self._summarize_node)

        graph.set_entry_point("route")
        graph.add_edge("route", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._decide_after_retrieve,
            {"fallback": "fallback", "rank": "rank"},
        )
        graph.add_edge("fallback", "retrieve")
        graph.add_conditional_edges(
            "rank",
            self._decide_after_rank,
            {"fallback": "fallback", "summarize": "summarize"},
        )
        graph.add_edge("summarize", END)

        return graph.compile()

    def run(self, text_query: str) -> RecipeAnswer:
        self._ensure_agents()
        final_state: GraphState = self.graph.invoke(
            {"text_query": text_query, "used_fallback": False, "fallback_attempts": 0}
        )

        recipe: RecipeRecord = final_state["recipe"]
        confidence = float(max(0.0, min(1.0, final_state.get("confidence", 0.0))))

        return RecipeAnswer(
            predicted_recipe=recipe.recipe_title,
            confidence=confidence,
            summary_steps=final_state.get("steps", []),
            ingredients=recipe.ingredients,
            description=recipe.description,
            rating=recipe.rating,
            vote_count=recipe.vote_count,
            source=recipe.source,
            source_url=recipe.url,
            rationale=final_state.get("rationale", ""),
            used_fallback=bool(final_state.get("used_fallback", False)),
        )
