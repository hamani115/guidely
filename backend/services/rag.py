from services.llm import generate_answer
from services.retrieval import search_documents


def ask_guidely(
    question: str,
    k: int = 3,
) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")

    sources = search_documents(
        question,
        k=k,
    )

    if not sources:
        return {
            "answer": (
                "No relevant information was " "found in the indexed documents."
            ),
            "sources": [],
        }

    answer = generate_answer(
        question,
        sources,
    )

    return {
        "answer": answer,
        "sources": sources,
    }
