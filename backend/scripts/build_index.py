import json
from pathlib import Path

from services.chunker import chunk_pages
from services.embeddings import embed_texts
from services.parser import extract_pdf_pages
from services.vector_store import create_index, save_index

DOCUMENTS_DIRECTORY = Path("data/sample-docs")

INDEX_DIRECTORY = Path("data/index")

FAISS_INDEX_PATH = INDEX_DIRECTORY / "guidely.index"

METADATA_PATH = INDEX_DIRECTORY / "metadata.json"


pdf_files = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))

if not pdf_files:
    raise RuntimeError("No PDF files found in data/sample-docs")


all_chunks = []


for pdf_path in pdf_files:
    print(f"\nProcessing {pdf_path.name}...")

    pages = extract_pdf_pages(pdf_path)

    chunks = chunk_pages(pages)

    print(f"  Pages: {len(pages)}")
    print(f"  Chunks: {len(chunks)}")

    all_chunks.extend(chunks)


print(f"\nTotal chunks: {len(all_chunks)}")


for vector_id, chunk in enumerate(all_chunks):
    chunk["vector_id"] = vector_id


texts = [chunk["embedding_text"] for chunk in all_chunks]

print("\nCreating embeddings...")

embeddings = embed_texts(texts)


print(f"Embedding matrix shape: {embeddings.shape}")


print("\nCreating FAISS index...")

index = create_index(embeddings)


print(f"Vectors stored in FAISS: {index.ntotal}")


INDEX_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


save_index(
    index,
    FAISS_INDEX_PATH,
)


with METADATA_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("\nSaved:")
print(f"  {FAISS_INDEX_PATH}")
print(f"  {METADATA_PATH}")
