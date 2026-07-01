from contextlib import asynccontextmanager
from pathlib import Path

import observability  # noqa: F401  -- loads .env / enables LangSmith tracing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding.embedder import Embedder
from vectorstore.qdrant_store import load_store, STORE_DIR, COLLECTION
from api.state import engine
from api.routes import chat


# A context manager is anything that runs setup code before you use something, and cleanup code after 
@asynccontextmanager 
async def lifespan(app: FastAPI):
    # The first part of the function, before the yield, will be executed before the application starts.
    # Load the store + embedder ONCE at startup, reuse for every request.
    if Path(STORE_DIR).exists():
        client = load_store(STORE_DIR)
        engine["client"] = client
        engine["embedder"] = Embedder()
        print(f"Engine ready: {client.get_collection(COLLECTION).points_count} vectors loaded.")
    else:
        print(f"No store in '{STORE_DIR}'. Run `python ingest.py` first.")
    yield # the part after the yield will be executed after the application has finished.
    engine.clear()


app = FastAPI(title="RAG API", lifespan=lifespan)

# Allow the Vite dev server (localhost:5173) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
