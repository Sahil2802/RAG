import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazily initialized to avoid blocking web-service startup.
_model: SentenceTransformer | None = None

# BGE models require a query prefix for retrieval tasks
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""  # No prefix needed for passages/documents


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: BAAI/bge-small-en-v1.5")
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. No prefix needed."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # Required for cosine similarity in Pinecone
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a user query. BGE requires the query prefix for best results."""
    model = _get_model()
    embedding = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    return embedding.tolist()
