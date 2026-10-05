import json
from pathlib import Path

from services.embeddings import embed_query
from services.vector_store import load_index, search_index


INDEX_PATH = Path("data/index/guidely.index")
METADATA_PATH = Path("data/index/metadata.json")


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_documents(
    query: str,
    k: int = 3,
) -> list[dict]:

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    index = load_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    query_embedding = embed_query(query)

    scores, indices = search_index(
        index,
        query_embedding,
        k=k,
    )

    results = []

    for score, vector_id in zip(scores, indices):
        if vector_id == -1:
            continue

        chunk = metadata[vector_id]

        result = {
            "score": float(score),
            "vector_id": int(vector_id),
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        }

        results.append(result)

    return results