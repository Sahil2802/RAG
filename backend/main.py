from pathlib import Path
import observability  # noqa: F401  -- loads .env / enables LangSmith tracing
from embedding.embedder import Embedder
from vectorstore.qdrant_store import load_store, STORE_DIR
from retriever.retriever import retrieve
from generation.generator import generate

def main():
    if not Path(STORE_DIR).exists():
        print(f"No store found in '{STORE_DIR}'. Run `python ingest.py` first.")
        return

    print("Loading store and embedder ...")
    client = load_store(STORE_DIR)
    embedder = Embedder()
    collection_info = client.get_collection("documents")
    print(f"Ready. {collection_info.points_count} vectors loaded. Ask a question (or 'exit').\n")

    while True:
        query = input("Q: ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break

        docs = retrieve(query, client, embedder)
        result = generate(query, docs)

        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}\n")

if __name__ == "__main__":
    main()
