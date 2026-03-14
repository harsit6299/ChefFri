from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_path: str = os.getenv("DATA_PATH", "./food_recipes.csv")
    index_dir: str = os.getenv("INDEX_DIR", "./index_store")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "8"))
    use_fallback: bool = os.getenv("USE_FALLBACK", "true").lower() == "true"


settings = Settings()
