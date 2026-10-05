from services.retrieval import search_documents


# query = (
#     "What are the objectives of the "
#     "Master of Science in Cybersecurity program?"
# )

query = input("Enter your question: ")

results = search_documents(
    query,
    k=3,
)


print(f"\nQuestion:\n{query}")

print("\nTop results:")


for rank, result in enumerate(results, start=1):
    print("\n" + "=" * 70)

    print(f"Rank: {rank}")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Chunk: {result['chunk_index']}")
    print(f"Vector ID: {result['vector_id']}")

    print("\nText:")
    print(result["text"])