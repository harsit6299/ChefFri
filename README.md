# ChefFri - An Agentic RAG based Agent

This project builds an agentic RAG system on top of `food_recipes.csv` to:
- predict the best matching recipe for a user query,
- summarize the recipe into 5-10 actionable steps,
- fetch recipes from an external source when local confidence is low.

![image](https://github.com/harsit6299/ChefFri/blob/422a7055fafdf5c9dbb11cf47ba9cba5b3a62889/Image.png)

## Agent Design (3 LLM agents + 2 supporting components)

Of the 5 stages in the pipeline, 3 are genuine LLM agents — the model makes a
judgment call. The other 2 are deterministic code that the agents rely on,
not agents themselves. Being precise about this distinction matters if
anyone asks "walk me through your agents."

**LLM agents:**

1. Router Agent: parses intent/constraints from user text.
2. Ranker Agent: selects the best recipe candidate and reports a confidence score.
3. Summarizer Agent: converts raw instructions into 5-10 actionable steps.

**Supporting components (no LLM call):**

- Hybrid Retrieval: dense embeddings (Sentence Transformers + FAISS) + BM25 keyword search.
- Data Acquisition (Fallback): fetches from TheMealDB when local retrieval/ranking confidence is low.

## Agentic Orchestration (LangGraph)

`src/graph_pipeline.py` (`RecipeAgenticPipeline`) replaces the fixed
Python sequence in `src/pipeline.py` with a LangGraph `StateGraph`. All 5
stages above (3 agents + 2 supporting components) become graph nodes, and
two independent conditional edges decide the path at runtime instead of one
hardcoded `if`:

- **After retrieval** - if the raw hybrid score is weak, route straight to
  the fallback node before the ranking agent ever sees the candidates.
- **After ranking** - if the LLM's own confidence in its pick is weak,
  even when retrieval looked fine, route to fallback anyway and re-rank
  once TheMealDB results are merged in.

Both edges share one `fallback_attempts` counter (capped by
`MAX_FALLBACK_ATTEMPTS`), so the graph can react to either signal without
looping more than once per query. `api.py`, `cli.py`, and `gradio_app.py`
all run on `RecipeAgenticPipeline` now; `src/pipeline.py` is kept as the
original fixed-sequence reference implementation.

Every response carries a `used_fallback` flag recording whether the graph
actually took the fallback route, so the routing decision is visible from
the API, the CLI, and the Gradio UI (shown there as a badge).

Note that the two edges do **not** use the same threshold, because retrieval
scores are not on one scale. In `hybrid` mode the score is
`0.7 * cosine + 0.3 * minmax(bm25)`, which is a meaningful 0-1 quantity. In
`bm25` mode it is a raw unbounded Okapi sum whose magnitude tracks query
length and term rarity rather than match quality — measured on this corpus,
anything with lexical overlap lands around 15-20 while a total miss scores
0, with no usable middle band. So in `bm25` mode the retrieval edge only
detects a total lexical miss, and quality filtering is carried by the
ranking-confidence edge, where the LLM reports a genuine 0-1 judgement.

Run the smoke test (no Groq key or embedding download needed - it mocks
the LLM/retriever to exercise just the graph wiring):

```bash
python -m src.test_graph_smoke
```

## Tech Stack

- LLM runtime: Groq API
- Retrieval mode: `hybrid` (Sentence Transformers + BM25) or `bm25` (no embedding download)
- Embeddings: Sentence Transformers (`all-MiniLM-L6-v2` by default, only used in `hybrid` mode)
- Dense index: FAISS
- Sparse index: BM25 (`rank-bm25`)
- Serving: FastAPI + Uvicorn

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set:
- `GROQ_API_KEY`
- Optional: `GROQ_MODEL`, `RETRIEVAL_MODE`, `TOP_K`, `USE_FALLBACK`

### Fast alternate mode (recommended if model download is slow)

Set in `.env`:

```env
RETRIEVAL_MODE=bm25
```

This skips Hugging Face embedding model download and uses keyword-based BM25 retrieval only.

4. Build index (optional, auto-build also works on first run):

```bash
python -m src.build_index
```

## Run API

```bash
uvicorn src.api:app --reload
```

Endpoints:
- `GET /health`
- `POST /predict` with JSON body:

```json
{ "query": "quick vegetarian mexican dinner with mushrooms" }
```

## Run CLI

```bash
python -m src.cli "quick vegetarian mexican dinner with mushrooms"
```

## Run Gradio Frontend

```bash
python -m src.gradio_app
```

Then open the local Gradio URL shown in the terminal (usually `http://127.0.0.1:7860`).

## Fallback Behavior

If top retrieval confidence is low, the system calls TheMealDB API, converts results into local schema, merges into runtime index, then retries ranking + summarization.

## What you need to provide

- Groq API key
- Internet access for fallback source calls
- Optional: additional trusted recipe APIs (Spoonacular/Edamam) for richer fallback data
