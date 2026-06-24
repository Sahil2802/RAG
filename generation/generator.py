from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_PROMPT = PromptTemplate.from_template("""You are a helpful assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know".

Context: {context}

Question: {question}

Answer: """)

_SYSTEM_TEMPLATE = """You are a helpful assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know".

Context:
{context}"""

_llm = ChatOllama(model = "llama3.2", temperature=0.1)


# single turn generation
def generate(query: str, docs: list[dict]) -> dict:
    if not docs:
        return {"answer": "I don't Know", "citations": [], "confidence": 0.0}
    
    context = "\n\n".join(doc["content"] for doc in docs)
    prompt = _PROMPT.format(context = context, question = query)

    response = _llm.invoke(prompt)
    answer = response.content

    citations = list({doc["metadata"].get("source_file",  "unknown") for doc in docs})

    confidence = round(sum(doc["similarity_score"] for doc in docs) / len(docs), 4)

    return {"answer": answer, "citations": citations, "confidence": confidence}


def _build_messages(messages: list[dict], docs: list[dict]) -> list:
    # System message carries the freshly-retrieved context; the conversation
    # turns give the model continuity across the chat.
    context = "\n\n".join(doc["content"] for doc in docs)
    chat = [SystemMessage(content=_SYSTEM_TEMPLATE.format(context=context))]
    for m in messages:
        if m["role"] == "user":
            chat.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat.append(AIMessage(content=m["content"]))
    return chat


def stream_answer(messages: list[dict], docs: list[dict]):
    # Yields answer tokens one at a time so the API can stream them.
    chat = _build_messages(messages, docs)
    for chunk in _llm.stream(chat):
        yield chunk.content


def build_sources(docs: list[dict]) -> dict:
    # Citations, confidence, and chunk list are all derived from retrieval,
    # so they're known before generation starts.
    if not docs:
        return {"citations": [], "confidence": 0.0, "chunks": []}

    citations = list({doc["metadata"].get("source_file", "unknown") for doc in docs})
    confidence = round(sum(doc["similarity_score"] for doc in docs) / len(docs), 4)
    chunks = [
        {
            "content": doc["content"],
            "source": doc["metadata"].get("source_file", "unknown"),
            "similarity_score": doc["similarity_score"],
            "rank": doc["rank"],
        }
        for doc in docs
    ]
    return {"citations": citations, "confidence": confidence, "chunks": chunks}