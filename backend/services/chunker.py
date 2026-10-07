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

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"]

    if not offsets:
        return []

    chunks = []

    start = 0

    while start < len(offsets):
        end = min(
            start + max_tokens,
            len(offsets),
        )

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        chunk = text[char_start:char_end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(offsets):
            break

        start = end - overlap_tokens

    return chunks


def chunk_pages(
    pages: list[dict],
) -> list[dict]:
    chunks = []
    chunk_index = 0

    heading_path = {}

    for page_position, page in enumerate(pages):
        sections, heading_path = split_page_into_sections(
            page,
            heading_path,
        )

        previous_page_context = ""

        if page_position > 0:
            previous_page_context = get_tail_tokens(
                pages[page_position - 1]["text"],
                CONTEXT_TOKENS,
            )

        for section in sections:
            path = section["heading_path"]

            hierarchy = [path[level] for level in sorted(path)]

            title = path.get(1)

            section_name = None

            if path:
                deepest_level = max(path)
                section_name = path[deepest_level]

            section_chunks = chunk_text(section["text"])

            for text in section_chunks:
                embedding_parts = []

                if previous_page_context:
                    embedding_parts.append(
                        "Previous page context:\n" + previous_page_context
                    )

                if hierarchy:
                    embedding_parts.append(
                        "Document hierarchy:\n" + " > ".join(hierarchy)
                    )

                embedding_parts.append(text)

                embedding_text = "\n\n".join(embedding_parts)

                chunks.append(
                    {
                        "source": page["source"],
                        "page": page["page"],
                        "title": title,
                        "section": section_name,
                        "heading_path": hierarchy,
                        "chunk_index": chunk_index,
                        "text": text,
                        "embedding_text": (embedding_text),
                    }
                )

                chunk_index += 1

    return chunks


def get_tail_tokens(
    text: str,
    token_count: int,
) -> str:
    if token_count <= 0:
        return ""

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"]

    if not offsets:
        return ""

    start = max(
        0,
        len(offsets) - token_count,
    )

    char_start = offsets[start][0]
    char_end = offsets[-1][1]

    return text[char_start:char_end].strip()


def split_page_into_sections(
    page: dict,
    heading_path: dict[int, str],
) -> tuple[list[dict], dict[int, str]]:
    sections = []

    current_path = heading_path.copy()
    current_lines = []
    has_body_text = False

    for element in page["elements"]:
        if element["type"] == "heading":
            # Save the previous section, but only if
            # it actually contained body text.
            if current_lines and has_body_text:
                sections.append(
                    {
                        "heading_path": current_path.copy(),
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            level = element["level"]

            # A new heading replaces headings at the
            # same level and anything below it.
            levels_to_remove = [
                existing_level
                for existing_level in current_path
                if existing_level >= level
            ]

            for existing_level in levels_to_remove:
                del current_path[existing_level]

            current_path[level] = element["text"]

            # Keep the heading itself in the source text.
            current_lines = [element["text"]]

            has_body_text = False

        else:
            current_lines.append(element["text"])

            has_body_text = True

    if current_lines and has_body_text:
        sections.append(
            {
                "heading_path": current_path.copy(),
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections, current_path
