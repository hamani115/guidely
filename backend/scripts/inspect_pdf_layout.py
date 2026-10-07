from pathlib import Path

import pymupdf

pdf_path = Path("data/sample-docs/Information-Technology.pdf")

document = pymupdf.open(pdf_path)

page = document[1]

page_data = page.get_text("dict")


for block in page_data["blocks"]:
    if "lines" not in block:
        continue

    for line in block["lines"]:
        spans = line["spans"]

        text = "".join(span["text"] for span in spans).strip()

        if not text:
            continue

        max_font_size = max(span["size"] for span in spans)

        fonts = [span["font"] for span in spans]

        is_bold = any("bold" in font.lower() for font in fonts)

        bold_marker = "BOLD" if is_bold else "regular"

        print(f"{max_font_size:5.1f} pt | " f"{bold_marker:7} | " f"{text}")
