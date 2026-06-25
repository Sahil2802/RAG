from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_PROMPT = PromptTemplate.from_template("""You are a document question-answering assistant.
Answer ONLY using the context chunks provided below.
If the answer is not in the context, say: "This information is not in the provided documents."
Do NOT use any knowledge from your training data.
Do NOT infer or extrapolate beyond what is explicitly stated.

Cite the chunk ID for every claim, inline. Format: "The tax rate is 7.65% [chunk_1]."
If you cannot cite a chunk for a claim, do not make that claim.

Context:
{context}

Question: {question}

Answer: """)

_SYSTEM_TEMPLATE = """You are a document question-answering assistant.
Answer ONLY using the context chunks provided below.
If the answer is not in the context, say: "This information is not in the provided documents."
Do NOT use any knowledge from your training data.
Do NOT infer or extrapolate beyond what is explicitly stated.

Context:
{context}"""

# Appended to the current user turn so grounding is enforced as the model
_CITATION_INSTRUCTION = (
    "\n\nCite the chunk ID for every claim, inline. "
    'Format: "The tax rate is 7.65% [chunk_1]." '
    "If you cannot cite a chunk for a claim, do not make that claim."
)

_llm = ChatOllama(model = "llama3.2", temperature=0.1)


def _label_context(docs: list[dict]) -> str:
    # Prefix each chunk with a stable [chunk_i] ID the model can cite.
    return "\n\n".join(f"[chunk_{i}]: {doc['content']}" for i, doc in enumerate(docs))


# single turn generation
def generate(query: str, docs: list[dict]) -> dict:
    if not docs:
        return {"answer": "This information is not in the provided documents.", "citations": [], "confidence": 0.0}

    context = _label_context(docs)
    prompt = _PROMPT.format(context = context, question = query)

    response = _llm.invoke(prompt)
    answer = response.content

    citations = list({doc["metadata"].get("source_file",  "unknown") for doc in docs})

    confidence = round(sum(doc["similarity_score"] for doc in docs) / len(docs), 4)

    return {"answer": answer, "citations": citations, "confidence": confidence}


def _build_messages(messages: list[dict], docs: list[dict]) -> list:
    # System message carries the freshly-retrieved context, with each chunk
    # labelled by ID so the model can cite it inline.
    context = _label_context(docs)
    # Start by filling chat with a system message that contains the context.
    chat = [SystemMessage(content=_SYSTEM_TEMPLATE.format(context=context))]
    # Append the citation instruction only to the current (last) user turn.
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"),
        default=-1,
    )
    for i, m in enumerate(messages):
        if m["role"] == "user":
            content = m["content"]
            if i == last_user_idx:
                content += _CITATION_INSTRUCTION
            chat.append(HumanMessage(content=content))
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

    # Citations => ['refund_policy.pdf', 'support_guide.pdf']
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