from transformers import AutoTokenizer

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CONTEXT_TOKENS,
    EMBEDDING_MODEL_NAME,
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def count_tokens(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return len(tokens)


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []

    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(tokens):
            break

        start = end - overlap_tokens

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []

    chunk_index = 0

    for page_position, page in enumerate(pages):
        page_chunks = chunk_text(page["text"])

        context = ""

        if page_position > 0:
            previous_page = pages[page_position - 1]

            context = get_tail_tokens(
                previous_page["text"],
                CONTEXT_TOKENS,
            )

        for text in page_chunks:
            if context:
                embedding_text = context + "\n\n" + text
            else:
                embedding_text = text

            chunk = {
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": text,
                "embedding_text": embedding_text,
            }

            chunks.append(chunk)

            chunk_index += 1

    return chunks


def get_tail_tokens(
    text: str,
    token_count: int,
) -> str:
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    tail_tokens = tokens[-token_count:]

    return tokenizer.decode(
        tail_tokens,
        skip_special_tokens=True,
    ).strip()
