"""
rebuild_index.py
----------------
Builds a FAISS vector index from the processed events CSV.
Uses Mistral AI cloud embeddings to avoid local model download issues.
"""

import logging
import os
import sys
import pandas as pd
import httpx
import urllib3
import warnings
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable XetHub and other HF-related stuff
os.environ["HF_HUB_DISABLE_XET"] = "1"

# --- ZSCALER / PROXY WORKAROUNDS ---
import httpx


def patch_httpx_ssl(client_class):
    """
    Surgically patches httpx classes to ignore SSL verification.
    This is necessary to bypass Zscaler/Corporate VPN inspection.
    """
    original_init = client_class.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        original_init(self, *args, **kwargs)

    client_class.__init__ = patched_init


# Apply the patch to both Client and AsyncClient independently
patch_httpx_ssl(httpx.Client)
patch_httpx_ssl(httpx.AsyncClient)

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

PROCESSED_DATA_PATH = "data/processed/events_structured.csv"
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_NAME = "mistral-embed"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Functions ──────────────────────────────────────────────────────────────────


def load_events(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.dropna(subset=["content"], inplace=True)
    logger.info(f"  Loaded {len(df)} events.")
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row["content"]),
            metadata={
                "uid": str(row.get("uid", "")),
                "title": str(row.get("title", "")),
                "city": str(row.get("city", "")),
                "url": str(row.get("url", "")),
            },
        )
        docs.append(doc)
    logger.info(f"  Created {len(docs)} documents.")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"  Split into {len(chunks)} chunks.")
    return chunks


def build_faiss_index(chunks: list[Document], index_path: str) -> None:
    logger.info(
        f"  [DEBUG] Starting indexing with Mistral model: {EMBEDDING_MODEL_NAME}"
    )
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY not found.")
        raise ValueError("MISTRAL_API_KEY not found.")

    # Initialize embeddings
    embeddings = MistralAIEmbeddings(
        mistral_api_key=api_key, model=EMBEDDING_MODEL_NAME
    )

    logger.info("  [DEBUG] Sending chunks to Mistral API (SSL check disabled)...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        logger.info(f"  ✅ FAISS index saved to '{index_path}'.")
    except Exception as e:
        logger.error(f"  ❌ Error during Mistral API call: {e}")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RUNNING REBUILD INDEX (FIXED PATCH)")
    logger.info("=" * 60)

    df = load_events(PROCESSED_DATA_PATH)
    documents = build_documents(df)
    chunks = split_documents(documents)
    build_faiss_index(chunks, FAISS_INDEX_PATH)

    logger.info("=" * 60)
    logger.info("✅ Done.")
