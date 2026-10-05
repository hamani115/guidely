from pathlib import Path

from services.parser import extract_pdf_pages

documents_directory = Path("data/sample-docs")

pdf_files = sorted(documents_directory.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_path in pdf_files:
    pages = extract_pdf_pages(pdf_path)

    print(f"File: {pdf_path.name}")
    print(f"Pages containing text: {len(pages)}")

    total_characters = sum(len(page["text"]) for page in pages)

    print(f"Extracted characters: {total_characters}")
    print("-" * 50)
