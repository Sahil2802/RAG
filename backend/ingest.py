from ingestion.loader import load_dir
from ingestion.chunker import chunk_documents
from embedding.embedder import Embedder
from vectorstore.qdrant_store import build_and_save, STORE_DIR

def ingest(source: str, store_dir: str = STORE_DIR) -> None:
    print(f"Step [1/4] Loading documents from '{source}' ...")
    docs, failed = load_dir(source)
    print(f" {len(docs)} pages loaded", end="")
    if failed:
        print(f" | {len(failed)} file(s) failed: {failed}")

    print(f"Step [2/4] Chunking ...")
    chunks = chunk_documents(docs)
    print(f" {len(chunks)} from {len(docs)} pages")
    
    print(f"Step [3/4] Embedding ...")
    embedder = Embedder()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    print(f"      {len(embeddings)} embeddings  (dim={len(embeddings[0])})")
    
    print(f"Step [4/4] Building Qdrant index ...")
    build_and_save(embeddings, chunks, store_dir)
    print(f"      Done. {len(embeddings)} vectors stored")

    
if __name__ == "__main__":
    ingest("data/")