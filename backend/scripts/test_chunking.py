from pathlib import Path

from services.chunker import chunk_pages, count_tokens
from services.parser import extract_pdf_pages

pdf_path = Path("data/sample-docs/Information-Technology.pdf")

pages = extract_pdf_pages(pdf_path)

chunks = chunk_pages(pages)

print(f"Document: {pdf_path.name}")
print(f"Pages containing text: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    if chunk["page"] == 3:
        print("\n--- DISPLAY TEXT ---")
        print(chunk["text"])

        print("\n--- EMBEDDING TEXT ---")
        print(chunk["embedding_text"])

        break

print("\n--- FIRST 10 CHUNKS ---")

for chunk in chunks[:10]:
    token_count = count_tokens(chunk["text"])

    print(
        f"Chunk {chunk['chunk_index']} | "
        f"Page {chunk['page']} | "
        f"{token_count} tokens"
    )


print("\n--- FIRST CHUNK CONTENT ---")
print(f"Source: {chunks[0]['source']}")
print(f"Page: {chunks[0]['page']}")
print(f"Chunk: {chunks[0]['chunk_index']}")
print()
print(chunks[0]["text"])
