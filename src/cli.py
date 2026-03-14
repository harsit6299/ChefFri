from __future__ import annotations

import argparse
import json

from src.pipeline import RecipeRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Recipe Agentic RAG CLI")
    parser.add_argument("query", type=str, help="Recipe request, e.g. 'high protein vegetarian dinner'")
    args = parser.parse_args()

    pipeline = RecipeRAGPipeline()
    result = pipeline.run(args.query)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
