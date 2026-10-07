from pathlib import Path

from services.parser import extract_pdf_pages

pdf_path = Path("data/sample-docs/Information-Technology.pdf")

pages = extract_pdf_pages(pdf_path)

page = pages[1]


print(f"Source: {page['source']}")
print(f"Page: {page['page']}")
print()


for element in page["elements"]:
    if element["type"] == "heading":
        print(f"HEADING L{element['level']} | " f"{element['text']}")
    else:
        print(f"TEXT       | " f"{element['text']}")
