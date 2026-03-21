from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Use the same tokenizer as the embedding model for accurate token counts
_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def token_length_function(text: str) -> int:
    return count_tokens(text)


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,           # tokens
    chunk_overlap=100,        # tokens
    length_function=token_length_function,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[dict], file_hash: str) -> list[dict]:
    """
    Takes parsed pages, returns flat list of chunks with full metadata.

    Returns:
        list of {
            pinecone_id: str,
            chunk_index: int,
            chunk_text: str,
            page_number: int | None,
            token_count: int
        }
    """
    chunks = []
    global_index = 0

    for page in pages:
        page_chunks = _splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "pinecone_id": f"{file_hash}_{global_index}",
                "chunk_index": global_index,
                "chunk_text": chunk_text,
                "page_number": page["page_number"],
                "token_count": count_tokens(chunk_text),
            })
            global_index += 1

    return chunks
