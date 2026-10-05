import json
from pathlib import Path

from services.retrieval import search_documents

QUERIES_PATH = Path("data/evaluation/retrieval_queries.json")


with QUERIES_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    test_cases = json.load(file)


top1_passed = 0
top3_passed = 0


for number, test_case in enumerate(
    test_cases,
    start=1,
):
    query = test_case["query"]
    expected_source = test_case["expected_source"]
    expected_pages = set(test_case["expected_pages"])

    results = search_documents(
        query,
        k=3,
    )

    matching_rank = None

    for rank, result in enumerate(
        results,
        start=1,
    ):
        correct_source = result["source"] == expected_source

        correct_page = result["page"] in expected_pages

        if correct_source and correct_page:
            matching_rank = rank
            break

    if matching_rank == 1:
        top1_passed += 1

    if matching_rank is not None:
        top3_passed += 1

    status = "PASS" if matching_rank is not None else "FAIL"

    print(f"{number}. [{status}] {query}")

    print(f"   Expected: " f"{expected_source}, " f"page(s) {sorted(expected_pages)}")

    if matching_rank is not None:
        print(f"   Correct passage rank: " f"{matching_rank}")
    else:
        print("   Correct passage not found " "in top 3")

    print("   Retrieved:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"      {rank}. "
            f"{result['source']} "
            f"page {result['page']} "
            f"(score={result['score']:.4f})"
        )

    print()


total = len(test_cases)


top1_accuracy = top1_passed / total * 100 if total else 0

top3_accuracy = top3_passed / total * 100 if total else 0


print("=" * 70)

print(f"Top-1 passage accuracy: " f"{top1_passed}/{total} " f"({top1_accuracy:.1f}%)")
print(f"Retrieval@3: " f"{top3_passed}/{total} " f"({top3_accuracy:.1f}%)")
