from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION: str = "documents"
STORE_DIR: str = "qdrant_storage"


def build_and_save(
    embeddings: list[list[float]],
    chunks: list[Document],
    store_dir: str,
) -> None:
    client = QdrantClient(path=store_dir)
    dim = len(embeddings[0])

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "content": chunks[i].page_content,
                "metadata": chunks[i].metadata,
            },
        )
        for i in range(len(embeddings))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def load_store(store_dir: str) -> QdrantClient:
    return QdrantClient(path=store_dir)
