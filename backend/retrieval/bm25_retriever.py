import pickle
import io
import logging
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_pc = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX_NAME)

STORAGE_BUCKET = "system"
STORAGE_KEY = "bm25_index.pkl"


def rebuild_bm25_index(supabase_client) -> None:
    """
    Fetch all chunk texts from Pinecone metadata, build BM25 index,
    persist to Supabase Storage. Called after every successful ingestion.
    """
    try:
        pinecone_ids = []
        chunk_texts = []

        # Pinecone list() paginates through all vector IDs
        for page in _index.list():
            ids_batch = page
            if not ids_batch:
                continue

            # Fetch metadata for this batch
            fetch_response = _index.fetch(ids=ids_batch)
            for vid, vector in fetch_response.vectors.items():
                pinecone_ids.append(vid)
                chunk_texts.append(vector.metadata.get("chunk_text", ""))

        if not chunk_texts:
            logger.warning("No chunks found in Pinecone. Skipping BM25 rebuild.")
            return

        # Tokenization fallback for simple BM25
        tokenized = [text.lower().split() for text in chunk_texts]
        bm25 = BM25Okapi(tokenized)

        # Serialize and upload to Supabase Storage
        payload = pickle.dumps({"bm25": bm25, "pinecone_ids": pinecone_ids})

        # Ensure bucket exists (handled in DB schema but this is safer)
        supabase_client.storage.from_(STORAGE_BUCKET).upload(
            path=STORAGE_KEY,
            file=payload,
            file_options={"upsert": "true"},
        )
        logger.info(f"Successfully rebuilt BM25 index with {len(chunk_texts)} chunks.")

    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}", exc_info=True)


def load_bm25_index(supabase_client) -> tuple[BM25Okapi, list[str]]:
    """Download BM25 pickle from Supabase Storage on startup."""
    try:
        data = supabase_client.storage.from_(STORAGE_BUCKET).download(STORAGE_KEY)
        loaded = pickle.loads(data)
        return loaded["bm25"], loaded["pinecone_ids"]
    except Exception as e:
        logger.warning(f"Could not load BM25 index: {e}")
        return None, []
