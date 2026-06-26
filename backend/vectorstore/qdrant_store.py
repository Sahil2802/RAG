import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION: str = "documents"
STORE_DIR: str = "qdrant_storage"

# Namespace for deriving stable, deterministic point IDs from chunk content.
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk: Document) -> str:
    """Stable ID for a chunk: same content -> same ID (idempotent upsert)."""
    paper_id = chunk.metadata.get("paper_id", "")
    key = f"{paper_id}:{chunk.page_content}"
    # Same key -> same UUID every time, so re-ingesting the same chunk
    # overwrites its existing point instead of creating a duplicate.
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def build_and_save(
    embeddings: list[list[float]],
    chunks: list[Document],
    store_dir: str,
) -> None:
    if not embeddings:
        raise ValueError("No embeddings to store")

    client = QdrantClient(path=store_dir)
    try:
        dim = len(embeddings[0])

        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        #  Each point represents a chunk stored in the vector database
        points = [
            PointStruct(
                id=_point_id(chunk),
                vector=vector,       # list[float] of length 384 when using BGE
                # Payloads are other data you want to associate with the  vector:
                # Add payload indexes for any field you'll filter on (e.g. source, doc_id), otherwise filters scan all points
                payload={
                    "doc_id": chunk.metadata.get("paper_id", ""),
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
            for vector, chunk in zip(embeddings, chunks, strict=True)
        ]
        #  Inserts each PointStruct into the specified collection. If a point with the same ID already exists, it will be updated with the new values.
        client.upsert(collection_name=COLLECTION, points=points)
    finally:
        client.close()


def load_store(store_dir: str) -> QdrantClient:
    return QdrantClient(path=store_dir)
