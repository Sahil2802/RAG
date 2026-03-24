from fastapi import APIRouter, HTTPException
from supabase import create_client
from backend.config.settings import settings
from backend.api.models import DocumentRecord, DocumentListResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result = supabase.table("documents") \
        .select("*") \
        .order("created_at", desc=True) \
        .execute()
    return {
        "documents": result.data,
        "total": len(result.data),
    }


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result = supabase.table("documents") \
        .select("*") \
        .eq("id", document_id) \
        .single() \
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")
    return result.data


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str):
    from pinecone import Pinecone

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    # Delete vectors from Pinecone first
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    index.delete(filter={"document_id": {"$eq": document_id}})

    # Delete from Supabase (cascade deletes chunks if they were stored there, 
    # but currently they only live in Pinecone)
    supabase.table("documents").delete().eq("id", document_id).execute()

    # Rebuild BM25 index
    try:
        from backend.retrieval.bm25_retriever import rebuild_bm25_index
        rebuild_bm25_index(supabase)
    except Exception:
        pass
