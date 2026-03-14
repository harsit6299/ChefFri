# Recipe Agentic RAG (Groq)

This project builds an agentic RAG system on top of `food_recipes.csv` to:
- predict the best matching recipe for a user query,
- summarize the recipe into 5-10 actionable steps,
- fetch recipes from an external source when local confidence is low.

## Agent Design (5 agents/components)

1. Router Agent: parses intent/constraints from user text.
2. Retrieval Agent: hybrid retrieval (dense embeddings + BM25).
3. Ranker Agent: selects best recipe candidate and confidence.
4. Summarizer Agent: converts raw instructions into 5-10 steps.
5. Data Acquisition Agent: fallback fetch from TheMealDB when local retrieval is weak.

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
