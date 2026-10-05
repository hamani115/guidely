from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is None:
            text = ""

        text = text.strip()

        if not text:
            continue

        page_data = {
            "source": pdf_path.name,
            "page": page_number,
            "text": text,
        }

        pages.append(page_data)

    return pages
