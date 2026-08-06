from __future__ import annotations

"""Lightweight smoke test for the LangGraph wiring in graph_pipeline.py.

This does NOT call Groq or download embedding models - it scripts the
retrieval score / ranking confidence a mocked node returns each time it
runs, wires those mocks into the exact same graph shape (and the exact
same conditional-edge functions) that RecipeAgenticPipeline uses in
production, and asserts the routing behaves as intended:

  - strong retrieval + strong ranking confidence -> no fallback
  - weak retrieval score -> fallback BEFORE ranking, then finishes
  - weak ranking confidence (even with fine retrieval) -> fallback AFTER
    ranking, then re-ranks and finishes
  - both signals weak -> still only ONE fallback call, never an infinite
    loop

Run after `pip install -r requirements.txt`:

    python -m src.test_graph_smoke
"""

from langgraph.graph import END, StateGraph

from src.graph_pipeline import GraphState, RecipeAgenticPipeline, retrieval_score_threshold

# Retrieval scores live on a different scale per retrieval mode (0-1 in hybrid,
# raw unbounded BM25 in bm25 mode), so the scripted scores below are expressed
# relative to whatever threshold the ambient RETRIEVAL_MODE actually uses. That
# keeps these assertions about graph *wiring* rather than about one mode's
# numbers, and stops the test flipping results when .env changes.
_THRESHOLD = retrieval_score_threshold()
STRONG_SCORE = _THRESHOLD * 2.0
WEAK_SCORE = _THRESHOLD * 0.2


def _make_scripted_graph(retrieve_scores, rank_confidences):
    calls = {"retrieve": 0, "rank": 0, "fallback": 0}

    def route_node(state):
        return {"intent": None, "retrieval_query": state["text_query"]}

    def retrieve_node(state):
        score = retrieve_scores[min(calls["retrieve"], len(retrieve_scores) - 1)]
        calls["retrieve"] += 1
        return {"candidates": ["fake-candidate"], "top_score": score}

    def fallback_node(state):
        calls["fallback"] += 1
        attempts = state.get("fallback_attempts", 0) + 1
        return {"used_fallback": True, "fallback_attempts": attempts}

    def rank_node(state):
        confidence = rank_confidences[min(calls["rank"], len(rank_confidences) - 1)]
        calls["rank"] += 1
        return {"recipe": "fake-recipe", "confidence": confidence, "rationale": "test"}

    def summarize_node(state):
        return {"steps": ["step 1", "step 2"]}

    graph = StateGraph(GraphState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("rank", rank_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        RecipeAgenticPipeline._decide_after_retrieve,
        {"fallback": "fallback", "rank": "rank"},
    )
    graph.add_edge("fallback", "retrieve")
    graph.add_conditional_edges(
        "rank",
        RecipeAgenticPipeline._decide_after_rank,
        {"fallback": "fallback", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile(), calls


def run_case(name, retrieve_scores, rank_confidences, expected_fallback_calls):
    graph, calls = _make_scripted_graph(retrieve_scores, rank_confidences)
    final_state = graph.invoke(
        {"text_query": "test query", "used_fallback": False, "fallback_attempts": 0}
    )
    assert final_state["steps"] == ["step 1", "step 2"], f"{name}: summarize node did not run"
    assert calls["fallback"] == expected_fallback_calls, (
        f"{name}: expected {expected_fallback_calls} fallback call(s), got {calls['fallback']}"
    )
    print(f"PASS  {name}  (fallback calls: {calls['fallback']})")


def main() -> None:
    run_case(
        "high score + high confidence -> no fallback",
        retrieve_scores=[STRONG_SCORE],
        rank_confidences=[0.9],
        expected_fallback_calls=0,
    )

    run_case(
        "low score triggers pre-rank fallback",
        retrieve_scores=[WEAK_SCORE, STRONG_SCORE],
        rank_confidences=[0.9],
        expected_fallback_calls=1,
    )

    run_case(
        "low confidence triggers post-rank fallback",
        retrieve_scores=[STRONG_SCORE, STRONG_SCORE],
        rank_confidences=[0.3, 0.9],
        expected_fallback_calls=1,
    )

    run_case(
        "bounded loop: never more than one fallback",
        retrieve_scores=[WEAK_SCORE, WEAK_SCORE],
        rank_confidences=[0.1, 0.1],
        expected_fallback_calls=1,
    )

    print("\nAll graph-wiring smoke tests passed.")


if __name__ == "__main__":
    main()
