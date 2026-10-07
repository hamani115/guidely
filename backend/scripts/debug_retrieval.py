from services.retrieval import search_documents

questions = [
    (
        "What courses are taken in the third semester "
        "of the Master of Business Administration program?"
    ),
    (
        "What are the objectives of the Master of Science "
        "in Engineering Management program?"
    ),
    (
        "What bachelor's degree specializations are accepted "
        "for the Master of Science in Artificial Intelligence Systems?"
    ),
    (
        "What are the admission requirements for Bachelor of Nursing "
        "holders applying to the Master of Science in Adult Health "
        "Advanced Practice Nursing?"
    ),
    (
        "What courses are taken in the second semester of the "
        "Postgraduate Diploma in Emergency Nursing?"
    ),
]


for question in questions:
    print("\n" + "=" * 90)
    print(question)
    print("=" * 90)

    results = search_documents(
        question,
        k=10,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. " f"score={result['score']:.4f}")

        print(f"   source: " f"{result['source']}")

        print(f"   page: " f"{result['page']}")

        print(f"   title: " f"{result.get('title')}")

        print(f"   section: " f"{result.get('section')}")

        print(f"   semantic: " f"{result.get('semantic_score', 0):.4f}")

        print(f"   title match: " f"{result.get('title_score', 0):.4f}")

        print(f"   section match: " f"{result.get('section_score', 0):.4f}")

        print(f"   final: " f"{result.get('rerank_score', 0):.4f}")
