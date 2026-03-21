from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ── Ingest ────────────────────────────────────────────


class IngestResponse(BaseModel):
    document_id: str
    status: str  # processing | duplicate | failed
    message: Optional[str] = None


# ── Documents ─────────────────────────────────────────


class DocumentRecord(BaseModel):
    id: str
    file_name: str
    file_size: int
    mime_type: str
    chunk_count: int
    page_count: Optional[int]
    status: str
    error_message: Optional[str]
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


# ── Query ─────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = None


class CitationMeta(BaseModel):
    excerpt_number: int
    file_name: str
    page_number: Optional[int]
    chunk_index: int
    pinecone_id: str
    reranker_score: Optional[float]


# ── Errors ────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
