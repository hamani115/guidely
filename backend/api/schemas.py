from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str

    k: int = Field(default=3, ge=1, le=10)


class SourceResponse(BaseModel):
    source: str
    page: int

    title: str | None = None
    section: str | None = None

    heading_path: list[str]

    chunk_index: int
    text: str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


# document
class DocumentResponse(BaseModel):
    filename: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    count: int


class DocumentUploadResponse(BaseModel):
    filename: str
    size_bytes: int
    pages: int
    reindex_required: bool


class ReindexResponse(BaseModel):
    documents_processed: int
    total_chunks: int
    vectors_stored: int
    embedding_dimension: int
    cached_embeddings_reused: int
    new_embeddings_generated: int
    cache_hit_rate: float
    index_path: str
    metadata_path: str
