from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from supabase import create_client
from backend.config.settings import settings
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    from backend.ingestion.indexer import compute_file_hash, is_duplicate

    # Validate file type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use .txt or .pdf."
        )

    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    file_hash = compute_file_hash(file_bytes)

    # Deduplication check
    if await is_duplicate(file_hash, supabase):
        existing = supabase.table("documents") \
            .select("id, status") \
            .eq("file_hash", file_hash) \
            .single() \
            .execute()
        return {
            "document_id": existing.data["id"],
            "status": "duplicate",
            "message": "File already ingested"
        }

    # Create document record
    document_id = str(uuid.uuid4())
    supabase.table("documents").insert({
        "id": document_id,
        "file_name": file.filename,
        "file_hash": file_hash,
        "file_size": len(file_bytes),
        "mime_type": file.content_type,
        "storage_path": f"{document_id}/{file.filename}",
        "status": "processing",
    }).execute()

    # Run remaining pipeline in background
    background_tasks.add_task(
        run_ingestion_pipeline,
        file_bytes, file.filename, file.content_type,
        file_hash, document_id, supabase
    )

    return {"document_id": document_id, "status": "processing"}


async def run_ingestion_pipeline(
    file_bytes, file_name, mime_type,
    file_hash, document_id, supabase
):
    from backend.ingestion.parser import parse_document
    from backend.ingestion.chunker import chunk_pages
    from backend.ingestion.embedder import embed_passages
    from backend.ingestion.indexer import store_file, upsert_to_pinecone

    try:
        # 1. Store raw file
        await store_file(file_bytes, file_name, document_id, supabase)

        # 2. Parse (Text or PDF)
        pages = parse_document(file_bytes, mime_type)

        # 3. Chunk
        chunks = chunk_pages(pages, file_hash)

        # 4. Embed (Batch)
        texts = [c["chunk_text"] for c in chunks]
        embeddings = embed_passages(texts)

        # 5. Index (Pinecone)
        upsert_to_pinecone(chunks, embeddings, document_id, file_name)

        # 6. Rebuild BM25 index (requires retrieval module to be ready)
        # Note: Phase 1 might skip BM25, but we follow the spec.
        try:
            from backend.retrieval.bm25_retriever import rebuild_bm25_index
            rebuild_bm25_index(supabase)
        except ImportError:
            logger.warning("BM25 retriever not implemented yet. Skipping index rebuild.")

        # 7. Mark as ready
        supabase.table("documents").update({
            "status": "ready",
            "chunk_count": len(chunks),
            "page_count": max((c["page_number"] or 1) for c in chunks) if chunks else None,
        }).eq("id", document_id).execute()

    except Exception as e:
        logger.error(f"Ingestion failed for {document_id}: {e}", exc_info=True)
        supabase.table("documents").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", document_id).execute()
