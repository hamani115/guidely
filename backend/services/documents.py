from pathlib import Path
import shutil
from pathlib import Path
from typing import BinaryIO

import pymupdf

DOCUMENTS_DIRECTORY = Path("data/sample-docs")


def list_documents() -> list[dict]:
    if not DOCUMENTS_DIRECTORY.exists():
        return []

    documents = []

    for path in sorted(DOCUMENTS_DIRECTORY.glob("*.pdf")):
        documents.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            }
        )

    return documents


def save_uploaded_pdf(
    filename: str,
    file_object: BinaryIO,
) -> dict:
    safe_filename = Path(filename).name

    if not safe_filename:
        raise ValueError("Uploaded file must have a filename")

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    DOCUMENTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = DOCUMENTS_DIRECTORY / safe_filename

    if destination.exists():
        raise FileExistsError(f"A document named " f"'{safe_filename}' already exists")

    temporary_path = DOCUMENTS_DIRECTORY / f".{safe_filename}.uploading"

    try:
        with temporary_path.open("wb") as output:
            shutil.copyfileobj(
                file_object,
                output,
            )

        try:
            document = pymupdf.open(temporary_path)

            page_count = document.page_count

            document.close()

        except Exception as error:
            raise ValueError("The uploaded file is not " "a valid PDF") from error

        if page_count <= 0:
            raise ValueError("The uploaded PDF contains " "no pages")

        temporary_path.replace(destination)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return {
        "filename": destination.name,
        "size_bytes": (destination.stat().st_size),
        "pages": page_count,
    }
