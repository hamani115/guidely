import hashlib
import json
from pathlib import Path

import numpy as np

from config import (
    CACHE_VERSION,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CONTEXT_TOKENS,
    EMBEDDING_MODEL_NAME,
)
from services.file_utils import calculate_file_hash

CACHE_DIRECTORY = Path("data/cache")

MANIFEST_PATH = CACHE_DIRECTORY / "manifest.json"


def calculate_cache_signature(
    file_path: Path,
) -> str:
    file_hash = calculate_file_hash(file_path)

    cache_data = {
        "file_hash": file_hash,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "cache_version": CACHE_VERSION,
    }

    serialized = json.dumps(
        cache_data,
        sort_keys=True,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(manifest: dict) -> None:
    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )


def get_cache_paths(
    file_path: Path,
) -> tuple[Path, Path]:
    stem = file_path.stem

    chunks_path = CACHE_DIRECTORY / f"{stem}.chunks.json"

    embeddings_path = CACHE_DIRECTORY / f"{stem}.embeddings.npy"

    return chunks_path, embeddings_path


def save_document_cache(
    file_path: Path,
    signature: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    manifest: dict,
) -> None:
    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_path, embeddings_path = get_cache_paths(file_path)

    with chunks_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    np.save(
        embeddings_path,
        embeddings,
    )

    manifest[file_path.name] = {
        "signature": signature,
        "chunk_count": len(chunks),
        "chunks_file": chunks_path.name,
        "embeddings_file": embeddings_path.name,
    }


def load_document_cache(
    file_path: Path,
    signature: str,
    manifest: dict,
):
    entry = manifest.get(file_path.name)

    if entry is None:
        return None

    if entry["signature"] != signature:
        return None

    chunks_path = CACHE_DIRECTORY / entry["chunks_file"]

    embeddings_path = CACHE_DIRECTORY / entry["embeddings_file"]

    if not chunks_path.exists():
        return None

    if not embeddings_path.exists():
        return None

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    if len(chunks) != len(embeddings):
        return None

    return chunks, embeddings
