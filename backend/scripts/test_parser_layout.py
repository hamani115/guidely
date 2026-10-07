from pathlib import Path

from services.parser import extract_pdf_pages

documents_directory = Path("data/sample-docs")


for pdf_path in sorted(documents_directory.glob("*.pdf")):
    pages = extract_pdf_pages(pdf_path)

    body_font_size = pages[0]["body_font_size"]

    total_lines = sum(len(page["lines"]) for page in pages)

    print(f"{pdf_path.name}")

    print(f"  Pages: {len(pages)}")

    print(f"  Lines: {total_lines}")

    print(f"  Detected body font: " f"{body_font_size:.1f} pt")

    print()
