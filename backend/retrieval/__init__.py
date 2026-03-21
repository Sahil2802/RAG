import logging
from backend.retrieval.vector_retriever import vector_search

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    top_k: int = 20,
    final_top_n: int = 5,
    filter_document_id: str | None = None,
) -> list[dict]:
    """
    Full retrieval pipeline. Phase 1: vector search only.
    All chunk data comes from Pinecone metadata — no Supabase call needed.

    Returns top_n chunks ready for the generation layer, each containing:
        - pinecone_id: str
        - score: float (cosine similarity)
        - chunk_text: str
        - file_name: str
        - page_number: int | None
        - chunk_index: int

    Args:
        query: The user's question.
        top_k: Number of candidates to retrieve from vector search.
        final_top_n: Number of chunks to return for generation.
        filter_document_id: Restrict search to a single document if set.
    """
    # Stage 1: Vector search
    vector_results = vector_search(
        query,
        top_k=top_k,
        filter_document_id=filter_document_id,
    )

    if not vector_results:
        logger.warning(f"No vector results for query: {query[:80]!r}")
        return []

    # Enrich results with metadata fields extracted from Pinecone
    enriched: list[dict] = []
    for result in vector_results[:final_top_n]:
        meta = result.get("metadata", {})
        enriched.append({
            "pinecone_id": result["pinecone_id"],
            "score": result["score"],
            "chunk_text": meta.get("chunk_text", ""),
            "file_name": meta.get("file_name", "unknown"),
            "page_number": meta.get("page_number"),
            "chunk_index": int(meta.get("chunk_index", 0)),
        })

    logger.info(
        f"Retrieval pipeline returned {len(enriched)} chunks "
        f"(from {len(vector_results)} vector candidates)"
    )
    return enriched
