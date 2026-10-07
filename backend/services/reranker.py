import re


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def word_overlap(
    query: str,
    value: str | None,
) -> float:
    normalized_query = normalize_text(query)
    normalized_value = normalize_text(value)

    if not normalized_value:
        return 0.0

    query_words = set(normalized_query.split())

    value_words = set(normalized_value.split())

    if not value_words:
        return 0.0

    shared_words = query_words & value_words

    return len(shared_words) / len(value_words)


def rerank_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    reranked = []

    for result in results:
        semantic_score = result["score"]

        title_score = word_overlap(
            query,
            result.get("title"),
        )

        section_score = word_overlap(
            query,
            result.get("section"),
        )

        rerank_score = semantic_score + 0.20 * title_score + 0.15 * section_score

        enriched_result = result.copy()

        enriched_result["semantic_score"] = semantic_score

        enriched_result["title_score"] = title_score

        enriched_result["section_score"] = section_score

        enriched_result["rerank_score"] = rerank_score

        reranked.append(enriched_result)

    reranked.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return reranked
