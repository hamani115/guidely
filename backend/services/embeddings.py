import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype("float32")
