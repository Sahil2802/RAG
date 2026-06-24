import numpy as np
import faiss
from embedding.embedder import Embedder
from langchain_core.documents import Document

def retrieve(query: str, 
             index: faiss.IndexFlatL2, 
             chunks: list[Document], 
             embedder: Embedder, 
             top_k: int = 5, 
             score_threshold: float = 1.5) -> list[dict]:
    
    query_vector = np.array([embedder.embed_query(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)

    retrieved_docs = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1 or dist > score_threshold:  # idx==-1 means FAISS found fewer results than top_k
            continue
        similarity = round(float(1 / (1 + dist)), 4)  # maps L2 distance → (0,1]: closer = higher score
        retrieved_docs.append({
            "id": int(idx),
            "content": chunks[idx].page_content,
            "metadata": chunks[idx].metadata,
            "similarity_score": similarity,
            "distance": round(float(dist), 4),
            "rank": rank + 1,
        })
    return retrieved_docs