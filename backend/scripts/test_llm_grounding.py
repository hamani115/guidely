from ollama import chat

from config import LLM_MODEL_NAME

context = """
[1] Information-Technology.pdf, page 2

Program Objectives

1. Gaining credibility and recognition in the field,
engage successfully in career advancement within the
cybersecurity sector, achieving higher-level positions
and leadership roles, and serving the needs of industry,
academia, or pursuing entrepreneurial ventures.

2. Commit to life-long learning and professional
development, seeking further educational opportunities,
adapting to changes in the cybersecurity landscape.

3. Contribute to the welfare of society and the
advancement of the cybersecurity profession through
responsible and ethical practices.
"""


question = (
    "What are the objectives of the " "Master of Science in Cybersecurity program?"
)


response = chat(
    model=LLM_MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using only the provided source. "
                "Do not add information that is not present."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Source:
{context}
""",
        },
    ],
    options={
        "temperature": 0.1,
    },
)


print(response.message.content)
