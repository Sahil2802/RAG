from qdrant_client import QdrantClient

from embedding.embedder import Embedder
from vectorstore.qdrant_store import COLLECTION


def retrieve(
    query: str,
    client: QdrantClient,
    embedder: Embedder,
    top_k: int = 5,
    min_similarity: float = 0.65,
) -> list[dict]:
    """Retrieve the top-k most similar documents for a query from Qdrant.

    Args:
        query: The search query string.
        client: A QdrantClient connected to the vector store.
        embedder: An Embedder instance used to embed the query.
        top_k: Maximum number of results to return.
        min_similarity: Minimum cosine similarity score threshold.

    Returns:
        A list of dicts with keys: id, content, metadata, similarity_score, rank.
        Results are ordered by descending similarity (rank starts at 1).
    """
    query_vector = embedder.embed_query(query)
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        score_threshold=min_similarity,
    )
    return [
        {
            "id": point.id,
            "content": point.payload["content"],
            "metadata": point.payload["metadata"],
            "similarity_score": round(point.score, 4),
            "rank": rank + 1,
        }
        for rank, point in enumerate(response.points)
    ]
