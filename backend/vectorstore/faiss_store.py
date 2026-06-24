import faiss
import pickle
from pathlib import Path
import numpy as np
from langchain_core.documents import Document

STORE_DIR = "faiss_store"

def build_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    vectors = np.array(embeddings, dtype=np.float32)
    dim = vectors.shape[1]
    # creates an empty FAISS index that will store vectors of size dim and use L2 distance (Euclidean distance) to measure similarity between them.
    index = faiss.IndexFlatL2(dim)
    index.add(vectors) # add vectors to the index
    return index

def save_store(index: faiss.IndexFlatL2, chunks: list[Document], store_dir: str = STORE_DIR
) -> None: 
    path = Path(store_dir)
    path.mkdir(exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_store(store_dir: str = STORE_DIR) -> tuple[faiss.IndexFlatL2, list[Document]]:
    path = Path(store_dir)
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks