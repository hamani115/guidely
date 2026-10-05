from pathlib import Path

import faiss
import numpy as np


def create_index(embeddings: np.ndarray):
    if len(embeddings) == 0:
        raise ValueError("Cannot create an index with no embeddings")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_index(index, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(path),
    )


def load_index(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    return faiss.read_index(str(path))


def search_index(
    index,
    query_embedding: np.ndarray,
    k: int = 3,
):
    query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(
        query_embedding,
        k,
    )

    return scores[0], indices[0]
