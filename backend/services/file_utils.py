import hashlib
from pathlib import Path


def calculate_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(8192)

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()
