import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# ragas library: dataset containers + the metric-scoring entrypoint
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# ragas metrics: each one prompts the judge LLM (and/or uses embeddings) to score a sample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

# lazily-initialized singletons so we don't reconnect to Ollama / reload the
# embedding model on every score_single() call
_llm_wrapper = None
_emb_wrapper = None
_metrics = None


def _init():
    global _llm_wrapper, _emb_wrapper, _metrics
    if _llm_wrapper is None:
        # local llama3.1:8b via Ollama acts as the ragas judge LLM -- llama3.2:3b
        # was too unreliable at following ragas's structured JSON output format
        _llm_wrapper = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
        # same embedding model as the retriever, used by embedding-based metrics
        _emb_wrapper = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )
        _metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# the main entry point, called once per test question 
def score_single(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    ground_truth_answer: str,
) -> dict:
    _init()
    # package one RAG interaction into ragas's expected sample format
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        response=answer,
        reference=ground_truth_answer,
    )
    # evaluate() expects a dataset even for a single sample
    dataset = EvaluationDataset(samples=[sample])
    try:
        # Perform the evaluation on the dataset with different metrics
        result = evaluate(
            dataset=dataset,
            metrics=_metrics,
            llm=_llm_wrapper,
            embeddings=_emb_wrapper,
            # 180s/job timeout - run them one at a time with more headroom instead
            run_config=RunConfig(timeout=600, max_workers=1),
        )
        # one-sample dataset -> grab row 0 of the pandas result
        row = result.to_pandas().iloc[0]
        return {
            "faithfulness": _safe_float(row.get("faithfulness")),
            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
            "context_precision": _safe_float(row.get("context_precision")),
            "context_recall": _safe_float(row.get("context_recall")),
        }
    except Exception as exc:
        # don't let one bad sample (e.g. Ollama down, malformed judge output)
        # kill the whole eval run over many questions
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "error": str(exc),
        }


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if f != f else round(f, 4)  # NaN check
    except (TypeError, ValueError):
        return None
