"""Integration tests for retriever.retrieve() with Qdrant backend."""
import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from embedding.embedder import Embedder
from vectorstore.qdrant_store import build_and_save, load_store
from retriever.retriever import retrieve


TEXTS = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "Dogs are loyal companions and make great pets.",
]

CHUNKS = [
    Document(page_content=text, metadata={"source": f"doc{i}"})
    for i, text in enumerate(TEXTS)
]


@pytest.fixture(scope="module")
def qdrant_client_and_embedder(tmp_path_factory):
    """Build a real Qdrant collection with 3 real embeddings, return (client, embedder)."""
    tmp_path = tmp_path_factory.mktemp("qdrant_store")
    embedder = Embedder()
    embeddings = embedder.embed_documents(TEXTS)
    build_and_save(embeddings, CHUNKS, str(tmp_path))
    client = load_store(str(tmp_path))
    return client, embedder


def test_retrieve_returns_nonempty(qdrant_client_and_embedder):
    """retrieve() with a similar query returns at least one result."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "Where is the Eiffel Tower?",
        client,
        embedder,
        top_k=5,
        min_similarity=0.4,
    )
    assert len(results) > 0


def test_retrieve_result_has_exact_keys(qdrant_client_and_embedder):
    """Each result dict has exactly the keys: id, content, metadata, similarity_score, rank."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "Where is the Eiffel Tower?",
        client,
        embedder,
        top_k=5,
        min_similarity=0.4,
    )
    assert len(results) > 0
    expected_keys = {"id", "content", "metadata", "similarity_score", "rank"}
    for result in results:
        assert set(result.keys()) == expected_keys, (
            f"Keys mismatch: got {set(result.keys())}, expected {expected_keys}"
        )


def test_retrieve_similarity_score_in_range(qdrant_client_and_embedder):
    """Each result's similarity_score is between 0 and 1 (cosine range)."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "Tell me about programming languages",
        client,
        embedder,
        top_k=5,
        min_similarity=0.2,
    )
    assert len(results) > 0
    for result in results:
        score = result["similarity_score"]
        assert 0 <= score <= 1, f"similarity_score {score} out of [0, 1] range"


def test_retrieve_results_ordered_by_rank(qdrant_client_and_embedder):
    """Results are ordered by rank starting from 1."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "Where is the Eiffel Tower?",
        client,
        embedder,
        top_k=5,
        min_similarity=0.2,
    )
    assert len(results) > 0
    ranks = [r["rank"] for r in results]
    assert ranks[0] == 1, f"First rank should be 1, got {ranks[0]}"
    assert ranks == list(range(1, len(results) + 1)), (
        f"Ranks should be consecutive from 1, got {ranks}"
    )


def test_retrieve_high_min_similarity_returns_empty(qdrant_client_and_embedder):
    """retrieve() with min_similarity=0.99 and an unrelated query returns empty list."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "xyzzy frobnicate quantum entanglement of dark matter",
        client,
        embedder,
        top_k=5,
        min_similarity=0.99,
    )
    assert results == [], f"Expected empty list, got {results}"


def test_retrieve_content_matches_chunks(qdrant_client_and_embedder):
    """The top result content for an Eiffel Tower query matches the Eiffel Tower chunk."""
    client, embedder = qdrant_client_and_embedder
    results = retrieve(
        "Where is the Eiffel Tower located?",
        client,
        embedder,
        top_k=1,
        min_similarity=0.4,
    )
    assert len(results) == 1
    assert results[0]["content"] == TEXTS[0]
