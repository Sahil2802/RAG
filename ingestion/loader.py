import logging
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader

logger = logging.getLogger(__name__)

# maps file extension to its LangChain loader class
LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

# looks up the right loader for a given file extension, raises if unsupported
def _resolve_loader(ext: str):
    loader_cls = LOADER_MAP.get(ext.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls


# loads a single file using whichever loader matches its extension
def _load_document(file_path: str) -> list:
    path = Path(file_path)
    loader_type = _resolve_loader(path.suffix)
    logger.info("Loading file: %s", file_path)
    docs = loader_type(file_path).load()  # e.g. PyMuPDFLoader("file.pdf").load()
    for doc in docs:
        doc.metadata["source_file"] = str(path)
        doc.metadata["file_type"] = path.suffix.lower()
    logger.debug("Loaded %d pages from %s", len(docs), file_path)
    return docs


# single entry point 
# accepts a file or a directory
# if directory: walks it, feeds each supported file to load_document one by one
# if file: passes it directly to load_document
def load_dir(source: str, glob: str = "**/*") -> tuple[list, list]:
    path = Path(source)
    if path.is_dir():
        docs = []
        failed_docs = []
        for file in path.glob(glob):
            if file.suffix.lower() in LOADER_MAP:  # skip unsupported types silently
                try:
                    docs.extend(_load_document(str(file)))  # extend keeps list flat
                except Exception as e:
                    logger.warning("Failed to load %s: %s", file, e)
                    failed_docs.append(str(file))
        return docs, failed_docs
    return _load_document(source), []

