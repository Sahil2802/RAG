import numpy as np
from sentence_transformers import SentenceTransformer

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model) # SentenceTransformer model for generating embeddings
    
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        # convert docs into a numerical vector (embedding)
        embeddings = self.model.encode(texts,batch_size=batch_size, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        # convert query into a numerical vector (embedding)
        embedding = self.model.encode(_BGE_QUERY_PREFIX + query, convert_to_numpy=True)
        return embedding.tolist()