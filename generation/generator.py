from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

_PROMPT = PromptTemplate.from_template("""You are a helpful assistant. Answer the question using only the context below. If the answer is not in the context, say "I don't know".

Context: {context}

Question: {question}
                                       
Answer: """)

_llm = ChatOllama(model = "llama3.2", temperature=0.1)



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