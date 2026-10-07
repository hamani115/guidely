from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

documents_directory = Path("data/sample-docs")


for pdf_path in sorted(documents_directory.glob("*.pdf")):
    print("\n" + "#" * 80)
    print(f"DOCUMENT: {pdf_path.name}")
    print("#" * 80)

    document = pymupdf.open(pdf_path)

    style_counts = Counter()
    style_examples = defaultdict(list)

    for page in document:
        page_data = page.get_text("dict")

        for block in page_data["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line["spans"]

                text = "".join(span["text"] for span in spans).strip()

                if not text:
                    continue

                max_font_size = round(
                    max(span["size"] for span in spans),
                    1,
                )

                fonts = [span["font"] for span in spans]

                is_bold = any("bold" in font.lower() for font in fonts)

                style = (
                    max_font_size,
                    is_bold,
                )

                style_counts[style] += 1

                if len(style_examples[style]) < 5:
                    style_examples[style].append(text)

    for style, count in style_counts.most_common():
        font_size, is_bold = style

        weight = "BOLD" if is_bold else "regular"

        print(f"\n{font_size:5.1f} pt | " f"{weight:7} | " f"{count} lines")

        for example in style_examples[style]:
            print(f"    {example}")
