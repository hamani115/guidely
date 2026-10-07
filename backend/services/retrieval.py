import json
from pathlib import Path

from services.embeddings import embed_query
from services.vector_store import load_index, search_index
from services.reranker import rerank_results

INDEX_PATH = Path("data/index/guidely.index")
METADATA_PATH = Path("data/index/metadata.json")


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def search_documents(
    query: str,
    k: int = 3,
) -> list[dict]:

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    index = load_index(INDEX_PATH)

    metadata = load_metadata(METADATA_PATH)

    query_embedding = embed_query(query)

    # Ask FAISS for a larger candidate pool.
    candidate_k = max(
        k * 10,
        30,
    )

    # Do not request more candidates than exist.
    candidate_k = min(
        candidate_k,
        len(metadata),
    )

    scores, indices = search_index(
        index,
        query_embedding,
        k=candidate_k,
    )

    results = []

    for score, vector_id in zip(
        scores,
        indices,
    ):
        if vector_id == -1:
            continue

        chunk = metadata[vector_id]

        results.append(
            {
                "score": float(score),
                "vector_id": int(vector_id),
                "source": chunk["source"],
                "page": chunk["page"],
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "heading_path": chunk.get(
                    "heading_path",
                    [],
                ),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        )

    reranked_results = rerank_results(
        query,
        results,
    )

    return reranked_results[:k]
