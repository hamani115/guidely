import json
from pathlib import Path

import numpy as np

from services.cache import (
    calculate_cache_signature,
    load_document_cache,
    load_manifest,
    save_document_cache,
    save_manifest,
)
from services.chunker import chunk_pages
from services.embeddings import embed_texts
from services.parser import extract_pdf_pages
from services.vector_store import (
    create_index,
    save_index,
)

DOCUMENTS_DIRECTORY = Path("data/sample-docs")

INDEX_DIRECTORY = Path("data/index")

FAISS_INDEX_PATH = INDEX_DIRECTORY / "guidely.index"

METADATA_PATH = INDEX_DIRECTORY / "metadata.json"


def build_index(verbose: bool = True) -> dict:
    def log(message: str = ""):
        if verbose:
            print(message)

    pdf_files = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError("No PDF files found in " f"{DOCUMENTS_DIRECTORY}")

    manifest = load_manifest()

    all_chunks = []
    all_embeddings = []

    cached_embeddings_count = 0
    new_embeddings_count = 0

    for pdf_path in pdf_files:
        log(f"\nProcessing " f"{pdf_path.name}...")

        signature = calculate_cache_signature(pdf_path)

        cached = load_document_cache(
            pdf_path,
            signature,
            manifest,
        )

        if cached is not None:
            chunks, embeddings = cached

            log("  Cache: HIT")
            log(f"  Chunks: " f"{len(chunks)}")

            cached_embeddings_count += len(chunks)

        else:
            log("  Cache: MISS")

            pages = extract_pdf_pages(pdf_path)

            chunks = chunk_pages(pages)

            texts = [chunk["embedding_text"] for chunk in chunks]

            embeddings = embed_texts(texts)

            save_document_cache(
                pdf_path,
                signature,
                chunks,
                embeddings,
                manifest,
            )

            log(f"  Pages: " f"{len(pages)}")

            log(f"  Chunks: " f"{len(chunks)}")

            new_embeddings_count += len(chunks)

        all_chunks.extend(chunks)

        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")

    log(f"\nTotal chunks: " f"{len(all_chunks)}")

    log("Embedding matrix shape: " f"{embeddings.shape}")

    for vector_id, chunk in enumerate(all_chunks):
        chunk["vector_id"] = vector_id

    log("\nCreating FAISS index...")

    index = create_index(embeddings)

    log("Vectors stored in FAISS: " f"{index.ntotal}")

    INDEX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_index(
        index,
        FAISS_INDEX_PATH,
    )

    save_manifest(manifest)

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

    log("\nSaved:")
    log(f"  {FAISS_INDEX_PATH}")
    log(f"  {METADATA_PATH}")

    total_embeddings = cached_embeddings_count + new_embeddings_count

    if total_embeddings > 0:
        cache_hit_rate = cached_embeddings_count / total_embeddings * 100
    else:
        cache_hit_rate = 0.0

    log("\nCache statistics:")

    log("  Cached embeddings reused: " f"{cached_embeddings_count}")

    log("  New embeddings generated: " f"{new_embeddings_count}")

    log("  Cache hit rate: " f"{cache_hit_rate:.1f}%")

    return {
        "documents_processed": len(pdf_files),
        "total_chunks": len(all_chunks),
        "vectors_stored": int(index.ntotal),
        "embedding_dimension": int(embeddings.shape[1]),
        "cached_embeddings_reused": (cached_embeddings_count),
        "new_embeddings_generated": (new_embeddings_count),
        "cache_hit_rate": round(
            cache_hit_rate,
            1,
        ),
        "index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }
