from collections import defaultdict
from pathlib import Path

import pymupdf


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    document = pymupdf.open(pdf_path)

    raw_pages = []
    all_lines = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        page_data = page.get_text("dict")

        lines = []

        for block in page_data["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = [span for span in line["spans"] if span["text"].strip()]

                if not spans:
                    continue

                text = "".join(span["text"] for span in spans).strip()

                if not text:
                    continue

                font_size = round(
                    max(float(span["size"]) for span in spans),
                    1,
                )

                is_bold = any(
                    (span.get("flags", 0) & 16)
                    or (
                        "bold"
                        in span.get(
                            "font",
                            "",
                        ).lower()
                    )
                    for span in spans
                )

                x0, y0, x1, y1 = line["bbox"]

                line_data = {
                    "text": text,
                    "font_size": font_size,
                    "bold": bool(is_bold),
                    "bbox": [
                        float(x0),
                        float(y0),
                        float(x1),
                        float(y1),
                    ],
                }

                lines.append(line_data)
                all_lines.append(line_data)

        lines.sort(
            key=lambda item: (
                round(item["bbox"][1], 1),
                item["bbox"][0],
            )
        )

        raw_pages.append(
            {
                "page": page_number,
                "lines": lines,
            }
        )

    body_font_size = detect_body_font_size(all_lines)

    pages = []

    for raw_page in raw_pages:
        elements = []

        for line in raw_page["lines"]:

            if line["text"].isdigit():
                continue

            heading_level = get_heading_level(
                line,
                body_font_size,
            )

            element_type = "heading" if heading_level is not None else "text"

            elements.append(
                {
                    "type": element_type,
                    "level": heading_level,
                    "text": line["text"],
                    "metadata": {
                        "font_size": line["font_size"],
                        "bold": line["bold"],
                        "bbox": line["bbox"],
                    },
                }
            )

        elements = merge_adjacent_headings(elements)

        plain_text = "\n".join(element["text"] for element in elements)

        pages.append(
            {
                "source": pdf_path.name,
                "page": raw_page["page"],
                "text": plain_text,
                "elements": elements,
            }
        )

    document.close()

    return pages


def detect_body_font_size(
    lines: list[dict],
) -> float:
    size_weights = defaultdict(int)

    for line in lines:
        # Bold text is more likely to be structural,
        # so don't use it when estimating body size.
        if line["bold"]:
            continue

        alphabetic_characters = sum(character.isalpha() for character in line["text"])

        # Ignore page numbers, symbols, etc.
        if alphabetic_characters < 3:
            continue

        size_weights[line["font_size"]] += alphabetic_characters

    if not size_weights:
        raise ValueError("Could not determine body font size")

    return max(
        size_weights,
        key=size_weights.get,
    )


def is_heading(
    line: dict,
    body_font_size: float,
) -> bool:
    font_size = line["font_size"]
    is_bold = line["bold"]

    # Bold text at normal body size or larger.
    bold_heading = is_bold and font_size >= body_font_size

    # Text substantially larger than normal body
    # text can also be a heading even if not bold.
    large_heading = font_size >= body_font_size + 2.0

    return bold_heading or large_heading


def merge_adjacent_headings(
    elements: list[dict],
) -> list[dict]:
    if not elements:
        return []

    merged = []

    for element in elements:
        if not merged:
            merged.append(element)
            continue

        previous = merged[-1]

        both_headings = previous["type"] == "heading" and element["type"] == "heading"

        if not both_headings:
            merged.append(element)
            continue

        previous_metadata = previous["metadata"]
        current_metadata = element["metadata"]

        same_font_size = (
            abs(previous_metadata["font_size"] - current_metadata["font_size"]) < 0.2
        )

        same_weight = previous_metadata["bold"] == current_metadata["bold"]
        same_level = previous["level"] == element["level"]

        previous_bbox = previous_metadata["bbox"]
        current_bbox = current_metadata["bbox"]

        previous_top = previous_bbox[1]
        current_top = current_bbox[1]

        vertical_step = current_top - previous_top

        largest_font_size = max(
            previous_metadata["font_size"],
            current_metadata["font_size"],
        )

        close_vertically = 0 < vertical_step <= largest_font_size * 1.6

        if same_font_size and same_weight and same_level and close_vertically:
            previous["text"] = previous["text"] + " " + element["text"]

            previous_metadata["bbox"] = [
                min(
                    previous_bbox[0],
                    current_bbox[0],
                ),
                min(
                    previous_bbox[1],
                    current_bbox[1],
                ),
                max(
                    previous_bbox[2],
                    current_bbox[2],
                ),
                max(
                    previous_bbox[3],
                    current_bbox[3],
                ),
            ]

        else:
            merged.append(element)

    return merged


def get_heading_level(
    line: dict,
    body_font_size: float,
) -> int | None:
    if not is_heading(
        line,
        body_font_size,
    ):
        return None

    font_size = line["font_size"]

    # Significantly larger than body text:
    # document/program/title-level heading.
    if font_size >= body_font_size + 2.0:
        return 1

    # Bold text at normal body size:
    # section-level heading.
    return 2
