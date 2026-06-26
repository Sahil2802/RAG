import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from vectorstore.qdrant_store import build_and_save, load_store


FAKE_EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
]

FAKE_CHUNKS = [
    Document(page_content="Hello world", metadata={"source": "doc1"}),
    Document(page_content="Foo bar baz", metadata={"source": "doc2"}),
]


def test_build_and_save_returns_none(tmp_path):
    result = build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    assert result is None


def test_load_store_returns_qdrant_client(tmp_path):
    build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    client = load_store(str(tmp_path))
    assert isinstance(client, QdrantClient)


def test_load_store_vectors_count(tmp_path):
    build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    client = load_store(str(tmp_path))
    collection_info = client.get_collection("documents")
    assert collection_info.points_count == 2


def test_search_top_result_payload_keys(tmp_path):
    build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    client = load_store(str(tmp_path))
    response = client.query_points(
        collection_name="documents",
        query=FAKE_EMBEDDINGS[0],
        limit=1,
    )
    results = response.points
    assert len(results) == 1
    payload = results[0].payload
    assert "content" in payload
    assert "metadata" in payload


def test_rebuild_does_not_duplicate_vectors(tmp_path):
    build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    build_and_save(FAKE_EMBEDDINGS, FAKE_CHUNKS, str(tmp_path))
    client = load_store(str(tmp_path))
    collection_info = client.get_collection("documents")
    assert collection_info.points_count == 2
