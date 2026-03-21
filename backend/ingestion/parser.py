import fitz  # PyMuPDF
from pathlib import Path


def parse_text_file(file_bytes: bytes) -> list[dict]:
    """Returns list of {text, page_number} — page_number is None for text files."""
    text = file_bytes.decode("utf-8", errors="replace")
    return [{"text": text, "page_number": None}]


def parse_pdf(file_bytes: bytes) -> list[dict]:
    """
    Returns list of {text, page_number} — one entry per page.
    Uses PyMuPDF for text extraction. Preserves reading order.
    """
    pages = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # "text" mode preserves reading order
        if text.strip():  # Skip blank pages
            pages.append({"text": text, "page_number": page_num})
    doc.close()
    return pages


def parse_document(file_bytes: bytes, mime_type: str) -> list[dict]:
    if mime_type == "application/pdf":
        return parse_pdf(file_bytes)
    elif mime_type == "text/plain":
        return parse_text_file(file_bytes)
    else:
        raise ValueError(f"Unsupported mime type: {mime_type}")
