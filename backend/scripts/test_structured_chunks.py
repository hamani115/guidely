from pathlib import Path

from services.chunker import chunk_pages
from services.parser import extract_pdf_pages

pdf_path = Path("data/sample-docs/Information-Technology.pdf")

pages = extract_pdf_pages(pdf_path)

chunks = chunk_pages(pages)


for chunk in chunks:
    if chunk["page"] != 2:
        continue

    print("\n" + "=" * 70)

    print(f"CHUNK {chunk['chunk_index']}")

    print(f"Title: {chunk['title']}")

    print(f"Section: {chunk['section']}")

    print("Hierarchy: " + " > ".join(chunk["heading_path"]))

    print("-" * 70)

    print(chunk["text"])
