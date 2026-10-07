from pathlib import Path

from services.chunker import split_into_sections
from services.parser import extract_pdf_pages

documents_directory = Path("data/sample-docs")


for pdf_path in sorted(documents_directory.glob("*.pdf")):
    print("\n")
    print("#" * 80)
    print(f"DOCUMENT: {pdf_path.name}")
    print("#" * 80)

    pages = extract_pdf_pages(pdf_path)

    total_detected = 0

    for page in pages:
        sections = split_into_sections(page["text"])

        detected_sections = [
            section for section in sections if section["section"] is not None
        ]

        if not detected_sections:
            continue

        print(f"\nPage {page['page']}")

        for section in detected_sections:
            print(f"  -> {section['section']}")

            total_detected += 1

    print(f"\nDetected headings: {total_detected}")
