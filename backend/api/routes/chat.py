import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from api.state import engine
from retriever.retriever import retrieve
from generation.generator import stream_answer, build_sources

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    # Format one Server-Sent Event: an event name + a JSON data line.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/health")
def health():
    loaded = "index" in engine
    return {
        "status": "ok",
        "store_loaded": loaded,
        "vector_count": engine["index"].ntotal if loaded else 0,
    }


@router.post("/chat")
def chat(request: ChatRequest):
    def event_stream():
        if "index" not in engine:
            yield _sse("error", {"message": "No documents indexed. Run `python ingest.py` first."})
            return

        messages = [m.model_dump() for m in request.messages]
        question = messages[-1]["content"]

        try:
            # Fresh retrieval for the latest question.
            docs = retrieve(question, engine["index"], engine["chunks"], engine["embedder"])
            # Only send sources when retrieval found relevant documents.
            if docs:
                yield _sse("sources", build_sources(docs))
            for token in stream_answer(messages, docs):
                yield _sse("token", {"text": token})
            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
