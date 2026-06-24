from langchain_core.documents import Document
from ingestion.chunker import chunk_documents


def _make_doc(content: str, source: str = "/papers/test.pdf") -> Document:
    return Document(
        page_content=content,
        metadata={"source_file": source, "file_type": ".pdf", "page": 0},
    )


def test_long_text_splits_into_multiple_chunks():
    doc = _make_doc("word " * 300)  # ~1500 chars, exceeds chunk_size=800
    chunks = chunk_documents([doc])
    assert len(chunks) > 1


def test_paper_id_derived_from_source_file():
    doc = _make_doc("Some content.", source="/papers/attention_is_all_you_need.pdf")
    chunks = chunk_documents([doc])
    assert chunks[0].metadata["paper_id"] == "attention_is_all_you_need"


def test_chunk_index_is_sequential():
    doc = _make_doc("word " * 300)
    chunks = chunk_documents([doc])
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_existing_metadata_preserved():
    doc = _make_doc("Some content.")
    chunks = chunk_documents([doc])
    assert chunks[0].metadata["file_type"] == ".pdf"
    assert chunks[0].metadata["page"] == 0


def test_short_text_stays_as_single_chunk():
    doc = _make_doc("A short sentence.")
    chunks = chunk_documents([doc])
    assert len(chunks) == 1


def test_overlap_means_adjacent_chunks_share_content():
    # With overlap=150, the tail of chunk N should appear at the head of chunk N+1
    doc = _make_doc("word " * 300)
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
    tail = chunks[0].page_content[-30:]
    assert tail in chunks[1].page_content