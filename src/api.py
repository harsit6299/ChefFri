from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import RecipeRAGPipeline


app = FastAPI(title="Recipe Agentic RAG API", version="1.0.0")
pipeline = RecipeRAGPipeline()


class PredictRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        answer = pipeline.run(payload.query)
        return answer.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
