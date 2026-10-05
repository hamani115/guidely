from pathlib import Path

from sentence_transformers import SentenceTransformer

from services.chunker import chunk_pages
from services.parser import extract_pdf_pages


pdf_path = Path(
    "data/sample-docs/Information-Technology.pdf"
)

pages = extract_pdf_pages(pdf_path)

chunks = chunk_pages(pages)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


chunk = chunks[1]

embedding = model.encode(chunk["text"])


print("Source:")
print(chunk["source"])

print("\nPage:")
print(chunk["page"])

print("\nChunk:")
print(chunk["chunk_index"])

print("\nText:")
print(chunk["text"])

print("\nEmbedding dimensions:")
print(len(embedding))

print("\nFirst 10 embedding values:")
print(embedding[:10])