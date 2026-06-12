"""
rebuild_index.py
----------------
Builds a FAISS vector index from the processed events CSV.
Uses Mistral AI cloud embeddings to avoid local model download issues.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

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
        kwargs['verify'] = False
        original_init(self, *args, **kwargs)
    client_class.__init__ = patched_init

# Apply the patch to both Client and AsyncClient independently
patch_httpx_ssl(httpx.Client)
patch_httpx_ssl(httpx.AsyncClient)

# Suppress warnings
import urllib3 # noqa: E402
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings # noqa: E402
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter # noqa: E402
from langchain_mistralai import MistralAIEmbeddings # noqa: E402
from langchain_community.vectorstores import FAISS # noqa: E402
from langchain_core.documents import Document # noqa: E402

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
    print(f"  Loaded {len(df)} events.")
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
    print(f"  Created {len(docs)} documents.")
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks.")
    return chunks

def build_faiss_index(chunks: list[Document], index_path: str) -> None:
    print(f"  [DEBUG] Starting indexing with Mistral model: {EMBEDDING_MODEL_NAME}")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found.")
    
    # Initialize embeddings
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key, model=EMBEDDING_MODEL_NAME)

    print("  [DEBUG] Sending chunks to Mistral API (SSL check disabled)...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        print(f"  ✅ FAISS index saved to '{index_path}'.")
    except Exception as e:
        print(f"  ❌ Error during Mistral API call: {e}")
        sys.exit(1)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING REBUILD INDEX (FIXED PATCH)")
    print("=" * 60)
    
    df = load_events(PROCESSED_DATA_PATH)
    documents = build_documents(df)
    chunks = split_documents(documents)
    build_faiss_index(chunks, FAISS_INDEX_PATH)
    
    print("=" * 60)
    print("✅ Done.")
