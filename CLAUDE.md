# ChefFri (Recipe Summarizer) — Project Context for Claude Code

This file is project context for Claude Code / any Claude extension opening this
folder (e.g. in Antigravity IDE). It was originally written by Claude in a
separate cloud session (Claude Cowork) on 2026-08-06, working directly with
Harshit on this codebase through a device bridge, and has been updated twice
since — once by an Antigravity/Claude Code session that actually installed,
ran, and fixed real bugs in the LangGraph upgrade, and again by the Cowork
session to reflect a resume-wording decision. Read it before doing anything
else in this repo.

## What this project is

"ChefFri" — an agentic RAG recipe assistant built on `food_recipes.csv`. Given
a natural-language food request, it retrieves the best-matching recipe
(hybrid FAISS + BM25 retrieval), ranks candidates via an LLM (Groq), summarizes
instructions into 5–10 actionable steps, and falls back to the TheMealDB API
when local retrieval/ranking confidence is low. Served via FastAPI, a CLI, and
a Gradio front end.

## Current architecture

- `src/pipeline.py` — `RecipeRAGPipeline`: the ORIGINAL fixed sequential
  pipeline (router → retrieve → maybe-fallback → rank → summarize). Fallback
  is decided once, by a single hardcoded score threshold, before ranking even
  happens. Kept in the repo as a reference/legacy implementation — it is
  **no longer wired into api/cli/gradio**.
- `src/graph_pipeline.py` — `RecipeAgenticPipeline`: LangGraph-based version,
  now used by `api.py` / `cli.py` / `gradio_app.py`. Same 5 pipeline stages as
  graph nodes, but two independent conditional edges decide routing at
  runtime instead of one hardcoded `if`:
  - **after retrieve**: weak hybrid/bm25 score → route to fallback *before*
    ranking ever happens
  - **after rank**: weak LLM ranking confidence → route to fallback *even if
    retrieval looked fine*, then re-rank once fresh data is merged in
  - both edges share one `fallback_attempts` counter (`MAX_FALLBACK_ATTEMPTS = 1`)
    so the graph can react to either signal without ever looping more than once
  - retrieval-mode-specific thresholds: `HYBRID_SCORE_THRESHOLD = 0.55` vs
    `BM25_SCORE_THRESHOLD = 1.0` (see "Fixes applied" below for why)
- `src/agents.py` — `RouterAgent`, `RankerAgent`, `SummarizerAgent` (LLM-driven,
  Groq, JSON-mode structured completions). These are the only 3 stages that
  involve an actual LLM decision — see "Agent count" below.
- `src/retriever.py` — `HybridRetriever`: FAISS (dense, Sentence Transformers
  `all-MiniLM-L6-v2`) + BM25 (sparse), weighted fusion `alpha=0.7` dense /
  `0.3` sparse, with backfill from sparse results if dense results run short.
  Index is persisted to `index_store/` and auto-reloaded instead of rebuilt
  every run. Has `add_records()` for incremental updates (see fixes below).
- `src/external_sources.py` — TheMealDB fallback fetch (`fetch_from_themealdb`)
- `src/llm.py` — `GroqClient` wrapper (default model `llama-3.3-70b-versatile`)
- `src/api.py`, `src/cli.py`, `src/gradio_app.py` — three entry points, all
  run on `RecipeAgenticPipeline`
- `src/test_graph_smoke.py` — mocks the LLM/retriever entirely and exercises
  just the LangGraph routing logic (no `GROQ_API_KEY` or embedding download
  needed)
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` — containerized
  deployment; CPU-only torch pinned before `requirements.txt`, index_store
  bind-mounted rather than baked in, API key passed via `env_file`.

## ✅ Verified working (2026-08-06)

The LangGraph orchestration was installed and run for real on Harshit's
Windows machine (Python 3.13.12): `pip install -r requirements.txt` resolved
`langgraph 0.3.34`, `python -m src.test_graph_smoke` passed all 4 cases, and
a real Groq round trip through all three agents returned a correct recipe.
The fallback path was also exercised against the live network with a
deliberately unmatchable query, correctly returning `used_fallback: true`.

Four real bugs were found and fixed by actually running the code (not just
reviewing it) — retrieval-score scale mismatch between hybrid/bm25 modes,
the fallback rebuilding the whole index and leaking duplicates across
requests, `used_fallback` being invisible in responses, and `RankerAgent`
crashing on a legitimate `null` from the LLM. All four are fixed; don't
reintroduce them. Full details were in the previous version of this file's
"Fixes applied" section if you need the exact before/after code.

## Agent count — resolved, resume now says "3 agents"

This used to be flagged as a nuance to be careful about; **it's now settled**.
Of the 5 pipeline stages, only 3 involve the LLM making a decision — Router,
Ranker, Summarizer. The other 2 (hybrid retrieval, TheMealDB fallback) are
deterministic code the agents rely on, not agents themselves.

Harshit decided to say **"3 agents"** on his resume (previously "5
agents/components", matching the old README wording) specifically because
it's more precise and fully defensible under interview questioning, even
though it slightly undersells the total pipeline size. **The README has
been updated to match**: it now says "3 LLM agents + 2 supporting
components" instead of "5 agents/components."

If you touch the agent/node count in `graph_pipeline.py` or `agents.py`
going forward, keep the README and this number in sync with whatever the
resume says — check with Harshit before changing either independently.

## Known details worth knowing

- `.env` needs `GROQ_API_KEY` (see `.env.example`). Optional: `GROQ_MODEL`,
  `RETRIEVAL_MODE` (`hybrid` or `bm25`), `TOP_K`, `USE_FALLBACK`.
- `RETRIEVAL_MODE=bm25` skips the Sentence Transformers embedding download —
  useful for a faster/lighter local setup if hybrid mode is slow to init.
- The `.venv` in this repo was built for Python 3.13. Make sure whatever
  environment you use here matches, or rebuild the venv.
- `food_recipes.csv` has a confirmed **8,009** recipes (direct count,
  matches `index_store/records.json`). The resume says "8K+", which is
  accurate.

## Why this matters (context, not code)

This project backs a "ChefFri — Agentic RAG System" section on Harshit's
resume, actively tailored for AI Platform Engineer / backend roles. The
LangGraph upgrade was specifically done to make the "agentic AI" claim
technically accurate and defensible in interviews. **If you make further
architecture changes here, flag anything that would change how this project
should be described on the resume** (e.g. agent count, orchestration
framework, new capabilities worth a stronger metric) — don't just change the
code and leave the resume/README stale.
