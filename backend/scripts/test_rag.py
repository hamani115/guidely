from services.rag import ask_guidely

question = (
    "What are the objectives of the " "Master of Science in Cybersecurity program?"
)


result = ask_guidely(question)


print("\nQUESTION")
print("=" * 70)
print(question)


print("\nANSWER")
print("=" * 70)
print(result["answer"])


print("\nSOURCES")
print("=" * 70)

for number, source in enumerate(
    result["sources"],
    start=1,
):
    print(f"\n[{number}] " f"{source['source']} " f"- page {source['page']}")

    print(f"Score: {source['score']:.4f}")

    print(source["text"])
