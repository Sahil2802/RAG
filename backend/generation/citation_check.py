import logging

logger = logging.getLogger(__name__)

REFUSAL_PHRASES = [
    "i couldn't find sufficient information",
    "the documents provided do not contain",
    "i don't have enough information",
    "not mentioned in the provided",
    "cannot be found in the documents",
]


def is_refusal(answer: str) -> bool:
    """Check if the LLM already refused to answer."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def check_citations_present(answer: str) -> bool:
    """
    Heuristic check: does the answer contain at least one citation bracket?
    Ensures the LLM followed citation instructions.
    """
    return "[Source:" in answer


def enforce_citations(answer: str, chunks: list[dict]) -> tuple[str, bool]:
    """
    Post-generation citation enforcement.

    Returns (final_answer, was_refused).

    If the answer has no citations AND isn't already a refusal, the LLM
    likely drifted into prior knowledge. Replace with a refusal message
    to prevent unsupported answers from reaching the user.
    """
    if is_refusal(answer):
        logger.info("LLM self-refused — answer is a refusal")
        return answer, True

    if not check_citations_present(answer):
        logger.warning(
            "Citation enforcement triggered — LLM answer had no [Source:] citations. "
            "Replacing with refusal."
        )
        refusal = (
            "The documents provided do not contain sufficient information "
            "to answer this question with confidence."
        )
        return refusal, True

    return answer, False
