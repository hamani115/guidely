import json
from pathlib import Path

from services.embeddings import embed_query
from services.vector_store import (
    load_index,
    search_index,
)


INDEX_PATH = Path("data/index/guidely.index")
METADATA_PATH = Path("data/index/metadata.json")


index = load_index(INDEX_PATH)


with METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)


query = (
    "What are the objectives of the "
    "Master of Science in Cybersecurity program?"
)


query_embedding = embed_query(query)


scores, indices = search_index(
    index,
    query_embedding,
    k=3,
)


print(f"\nQuestion:\n{query}")

print("\nTop 3 results:")


for rank, (score, vector_id) in enumerate(
    zip(scores, indices),
    start=1,
):
    if vector_id == -1:
        continue

    chunk = metadata[vector_id]

    print("\n" + "=" * 70)

    print(f"Rank: {rank}")
    print(f"Similarity: {score:.4f}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk_index']}")
    print(f"Vector ID: {vector_id}")

    print("\nText:")
    print(chunk["text"])