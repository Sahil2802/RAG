from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level — expensive to initialize
# BGE-small-en-v1.5 is the locked choice (384-dim)
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# BGE models require a query prefix for retrieval tasks
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""  # No prefix needed for passages/documents


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. No prefix needed."""
    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # Required for cosine similarity in Pinecone
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a user query. BGE requires the query prefix for best results."""
    embedding = _model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )
    return embedding.tolist()
