from ollama import chat

from config import LLM_MODEL_NAME

SYSTEM_PROMPT = """
You are Guidely, an internal knowledge assistant.

Answer the user's question using only the provided sources.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Base every factual claim on the provided sources.
- Pay close attention to section headings and labels in the sources.
- If the question asks about a specific section, such as
  "Program Objectives", use only information belonging to that
  section.
- Do not mix information from neighboring sections such as
  "Program Objectives" and "Program Intended Learning Outcomes".
- A section ends when another heading begins.
- If the provided sources do not contain enough information,
  clearly say that the information was not found in the sources.
- Cite supporting sources using [1], [2], [3], etc.
- Keep the answer clear and concise.
"""


def generate_answer(
    question: str,
    sources: list[dict],
) -> str:
    context_parts = []

    for number, source in enumerate(
        sources,
        start=1,
    ):
        context_parts.append(
            f"[{number}] "
            f"{source['source']}, "
            f"page {source['page']}\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_parts)

    user_prompt = f"""
    Question:
    {question}

    Sources:
    {context}

    Answer the question using only the sources above.

    Use the source headings to determine which information
    actually answers the question. Do not include information
    from a different section merely because it is related.
    """

    response = chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.1,
        },
    )

    return response.message.content.strip()
