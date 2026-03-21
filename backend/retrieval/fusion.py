import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF combines ranked lists from multiple retrievers without requiring
    score normalization. For each result in each list, its contribution
    is 1 / (k + rank + 1). Results appearing in multiple lists get
    a natural boost.

    Args:
        result_lists: Each list is a ranked list of dicts with 'pinecone_id'.
                      The first item in each list has rank 0 (highest).
        k: RRF constant. 60 is the standard default from the original paper.
           Higher values penalize lower ranks less aggressively.

    Returns:
        Merged list sorted by fused score descending. Each item has:
        {pinecone_id, rrf_score, sources: ['vector', 'bm25']}
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)

    source_names = ["vector", "bm25"]

    for source_name, result_list in zip(source_names, result_lists):
        for rank, result in enumerate(result_list):
            pid = result["pinecone_id"]
            rrf_scores[pid] += 1.0 / (k + rank + 1)
            sources[pid].append(source_name)

    merged = [
        {
            "pinecone_id": pid,
            "rrf_score": score,
            "sources": sources[pid],
        }
        for pid, score in rrf_scores.items()
    ]

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(
        f"RRF merged {sum(len(rl) for rl in result_lists)} candidates "
        f"into {len(merged)} unique results"
    )
    return merged
