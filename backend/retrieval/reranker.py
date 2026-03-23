import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    chunk_texts: dict[str, str],
    top_n: int = 5,
) -> list[dict]:
    """
    Re-score candidates using cross-encoder for precise relevance scoring.
    It takes the chunks you already retrieved and re-orders them so the most relevant ones come first.
    The cross-encoder sees the full (query, chunk) pair together, unlike
    bi-encoders which embed them separately. This is slower but far more
    accurate for final scoring over a small candidate set.

    Args:
        query: The user's question.
        candidates: RRF-fused results (or vector-only results if BM25 unavailable).
        chunk_texts: Map from pinecone_id to chunk text.
        top_n: Number of results to return after re-ranking.

    Returns:
        Top-n candidates sorted by cross-encoder score, each with
        'reranker_score' added.
    """
    if not candidates:
        return []

    # Build (query, passage) pairs for batch scoring
    pairs = [
        (query, chunk_texts.get(c["pinecone_id"], ""))
        for c in candidates
    ]

    try:
        reranker = _get_reranker()
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error(f"Cross-encoder re-ranking failed: {e}", exc_info=True)
        # Fallback: return candidates in original order without reranker scores
        for c in candidates[:top_n]:
            c["reranker_score"] = None
        return candidates[:top_n]

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["reranker_score"] = float(score)

    sorted_candidates = sorted(
        candidates,
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    logger.info(
        f"Re-ranked {len(candidates)} candidates — "
        f"top score: {sorted_candidates[0]['reranker_score']:.4f}, "
        f"returning top {top_n}"
    )
    return sorted_candidates[:top_n]
