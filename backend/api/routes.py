import logging

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from api.schemas import (
    DocumentListResponse,
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    ReindexResponse,
)
  
from services.rag import ask_guidely
from services.documents import (
    list_documents,
    save_uploaded_pdf,
)

from services.indexing import build_index

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "guidely-backend",
    }


@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        return ask_guidely(
            question,
            k=request.k,
        )

    except FileNotFoundError:
        logger.exception("Guidely index files are missing")

        raise HTTPException(
            status_code=503,
            detail=(
                "The document index is not available. "
                "Please index the documents first."
            ),
        )

    except Exception:
        logger.exception("Failed to answer question")

        raise HTTPException(
            status_code=503,
            detail=("Guidely could not complete the request."),
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
def get_documents():
    documents = list_documents()

    return {
        "documents": documents,
        "count": len(documents),
    }


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
)
def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=("Uploaded file must " "have a filename"),
        )

    try:
        document = save_uploaded_pdf(
            file.filename,
            file.file,
        )

    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception("Failed to upload document")

        raise HTTPException(
            status_code=500,
            detail=("The document could not " "be uploaded."),
        )

    return {
        **document,
        "reindex_required": True,
    }


@router.post(
    "/reindex",
    response_model=ReindexResponse,
)
def reindex_documents():
    try:
        result = build_index(verbose=False)

        return result

    except RuntimeError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception("Failed to rebuild document index")

        raise HTTPException(
            status_code=500,
            detail=("The document index could " "not be rebuilt."),
        )
