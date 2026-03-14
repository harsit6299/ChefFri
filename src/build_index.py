from __future__ import annotations

from src.config import settings
from src.data_loader import load_recipes
from src.retriever import HybridRetriever


def main() -> None:
    recipes = load_recipes(settings.data_path)
    retriever = HybridRetriever(
        settings.embedding_model,
        settings.index_dir,
        retrieval_mode=settings.retrieval_mode,
    )
    retriever.build(recipes)
    retriever.save()
    print(f"Index built with {len(recipes)} recipes in {settings.index_dir}")


if __name__ == "__main__":
    main()
