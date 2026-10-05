import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[str]:

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(tokens):
            break

        start = end - overlap_tokens

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []

    chunk_index = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for text in page_chunks:
            chunk = {
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": text,
            }

            chunks.append(chunk)

            chunk_index += 1

    return chunks
