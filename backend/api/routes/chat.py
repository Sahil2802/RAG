import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from api.state import engine
from retriever.retriever import retrieve
from generation.generator import stream_answer, build_sources
from vectorstore.qdrant_store import COLLECTION

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    # Format one Server-Sent Event: an event name + a JSON data line.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/health")
def health():
    # If the qdrant_storage folder didn't exist when the app started, 
    # engine would be empty -> loaded would be False.
    loaded = "client" in engine
    return {
        "status": "ok",
        "store_loaded": loaded,
        "vector_count": engine["client"].get_collection(COLLECTION).points_count if loaded else 0,
    }


@router.post("/chat")
def chat(request: ChatRequest):
    def event_stream():
        if "client" not in engine:
            yield _sse("error", {"message": "No documents indexed. Run `python ingest.py` first."})
            return

        messages = [m.model_dump() for m in request.messages]
        question = messages[-1]["content"]

        try:
            # Fresh retrieval for the latest question.
            docs = retrieve(question, engine["client"], engine["embedder"])
            # Off-topic gate: no chunk cleared the similarity bar, so there's
            # nothing to ground an answer in. Reply plainly instead of letting
            # the model answer from training knowledge.
            if not docs:
                yield _sse("token", {"text": "This information is not in the provided documents."})
                yield _sse("done", {})
                return
            yield _sse("sources", build_sources(docs))
            for token in stream_answer(messages, docs):
                yield _sse("token", {"text": token})
            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    # event_stream() creates a generator object (body doesn't run yet).
    # StreamingResponse iterates it, which drives the body to execute
    # and sends each yielded SSE chunk to the client as it's produced.
    return StreamingResponse(event_stream(), media_type="text/event-stream")
