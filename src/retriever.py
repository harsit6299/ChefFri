from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.data_loader import recipe_to_document
from src.schemas import RecipeRecord


@dataclass
class RetrievalResult:
    recipe: RecipeRecord
    score: float


class HybridRetriever:
    def __init__(self, embedding_model_name: str, index_dir: str, retrieval_mode: str = "hybrid") -> None:
        self.retrieval_mode = retrieval_mode
        self.embedding_model: SentenceTransformer | None = None
        if self.retrieval_mode == "hybrid":
            self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index_dir = index_dir
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.records: list[RecipeRecord] = []
        self.documents: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, recipes: list[RecipeRecord]) -> None:
        self.records = recipes
        self.documents = [recipe_to_document(r) for r in recipes]

        if self.retrieval_mode == "hybrid":
            if self.embedding_model is None:
                raise ValueError("Embedding model is not initialized in hybrid mode")
            embeddings = self.embedding_model.encode(self.documents, normalize_embeddings=True)
            matrix = np.array(embeddings, dtype=np.float32)

            self.faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            self.faiss_index.add(matrix)

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def save(self) -> None:
        if self.bm25 is None:
            raise ValueError("Index not built")

        os.makedirs(self.index_dir, exist_ok=True)
        if self.retrieval_mode == "hybrid":
            if self.faiss_index is None:
                raise ValueError("FAISS index is missing for hybrid mode")
            faiss.write_index(self.faiss_index, os.path.join(self.index_dir, "recipes.faiss"))

        with open(os.path.join(self.index_dir, "records.json"), "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self.records], f, ensure_ascii=False)

        with open(os.path.join(self.index_dir, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def load(self) -> None:
        if self.retrieval_mode == "hybrid":
            self.faiss_index = faiss.read_index(os.path.join(self.index_dir, "recipes.faiss"))

        with open(os.path.join(self.index_dir, "records.json"), "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.records = [RecipeRecord(**row) for row in raw]

        with open(os.path.join(self.index_dir, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 8, alpha: float = 0.7) -> list[RetrievalResult]:
        if self.bm25 is None:
            raise ValueError("Retriever index is not loaded or built")

        bm25_scores = self.bm25.get_scores(query.lower().split())
        if self.retrieval_mode != "hybrid":
            sparse_sorted = np.argsort(-bm25_scores)
            results = [
                RetrievalResult(recipe=self.records[int(idx)], score=float(bm25_scores[idx]))
                for idx in sparse_sorted[:top_k]
            ]
            return results

        if self.faiss_index is None or self.embedding_model is None:
            raise ValueError("Hybrid retriever requires FAISS index and embedding model")

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_embedding, dtype=np.float32)

        dense_scores, dense_idx = self.faiss_index.search(query_vec, top_k)
        dense_scores = dense_scores[0]
        dense_idx = dense_idx[0]

        bm25_min = float(np.min(bm25_scores))
        bm25_max = float(np.max(bm25_scores))
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

        results: list[RetrievalResult] = []
        seen = set()

        for rank, idx in enumerate(dense_idx):
            if idx < 0 or idx >= len(self.records):
                continue
            recipe = self.records[idx]
            dense = float(dense_scores[rank])
            sparse_norm = float((bm25_scores[idx] - bm25_min) / bm25_range)
            hybrid_score = alpha * dense + (1.0 - alpha) * sparse_norm
            results.append(RetrievalResult(recipe=recipe, score=hybrid_score))
            seen.add(int(idx))

        # Backfill with top sparse results if needed.
        if len(results) < top_k:
            sparse_sorted = np.argsort(-bm25_scores)
            for idx in sparse_sorted:
                if len(results) >= top_k:
                    break
                if int(idx) in seen:
                    continue
                sparse_norm = float((bm25_scores[idx] - bm25_min) / bm25_range)
                results.append(RetrievalResult(recipe=self.records[int(idx)], score=sparse_norm * (1.0 - alpha)))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
