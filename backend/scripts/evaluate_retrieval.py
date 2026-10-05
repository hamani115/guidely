import json
from pathlib import Path

from services.retrieval import search_documents

QUERIES_PATH = Path("data/evaluation/retrieval_queries.json")


with QUERIES_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    test_cases = json.load(file)


passed = 0


for number, test_case in enumerate(
    test_cases,
    start=1,
):
    query = test_case["query"]
    expected_source = test_case["expected_source"]

    results = search_documents(
        query,
        k=3,
    )

    retrieved_sources = [result["source"] for result in results]

    success = expected_source in retrieved_sources

    if success:
        passed += 1

    status = "PASS" if success else "FAIL"

    print(f"{number}. [{status}] {query}")
    print(f"   Expected: {expected_source}")
    print(f"   Retrieved: {retrieved_sources}")
    print()


total = len(test_cases)

accuracy = passed / total * 100 if total > 0 else 0


print("=" * 70)

print(f"Retrieval@3: " f"{passed}/{total} " f"({accuracy:.1f}%)")
