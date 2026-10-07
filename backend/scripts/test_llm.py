from ollama import chat

from config import LLM_MODEL_NAME

response = chat(
    model=LLM_MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": ("Explain cybersecurity " "in one sentence."),
        }
    ],
)


print(response.message.content)
