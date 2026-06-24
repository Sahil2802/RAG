from pathlib import Path
from embedding.embedder import Embedder
from vectorstore.faiss_store import load_store, STORE_DIR
from retriever.retriever import retrieve
from generation.generator import generate

def main():
    if not Path(STORE_DIR, "index.faiss").exists():
        print(f"No store found in '{STORE_DIR}'. Run `python ingest.py` first.")
        return

    print("Loading store and embedder ...")
    index, chunks = load_store(STORE_DIR)
    embedder = Embedder()
    print(f"Ready. {index.ntotal} vectors loaded. Ask a question (or 'exit').\n")

    while True:
        query = input("Q: ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break

        docs = retrieve(query, index, chunks, embedder)
        result = generate(query, docs)

        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}\n")

if __name__ == "__main__":
    main()
