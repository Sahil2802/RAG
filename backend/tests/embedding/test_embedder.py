import pytest
from embedding.embedder import Embedder

BGE_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_documents_count(embedder: Embedder):
    texts = ["The attention mechanism.", "Transformers changed NLP."]
    result = embedder.embed_documents(texts)
    assert len(result) == 2


def test_embed_documents_dimension(embedder: Embedder):
    result = embedder.embed_documents(["Sample text."])
    assert len(result[0]) == BGE_DIM


def test_embed_query_dimension(embedder: Embedder):
    result = embedder.embed_query("What is attention?")
    assert len(result) == BGE_DIM


def test_embed_query_differs_from_embed_document(embedder: Embedder):
    text = "What is attention?"
    query_vec = embedder.embed_query(text)
    doc_vec = embedder.embed_documents([text])[0]
    # BGE query prefix makes them different
    assert query_vec != doc_vec


def test_embed_documents_returns_floats(embedder: Embedder):
    result = embedder.embed_documents(["hello world"])
    assert all(isinstance(v, float) for v in result[0])


def test_embed_documents_empty_list(embedder: Embedder):
    result = embedder.embed_documents([])
    assert result == []