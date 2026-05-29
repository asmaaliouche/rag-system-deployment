"""
rebuild_index.py
----------------
Builds a FAISS vector index from the processed events CSV.

Pipeline:
    1. Load processed events from data/processed/events_structured.csv
    2. Split each event's 'content' field into overlapping text chunks
    3. Embed each chunk using a HuggingFace sentence-transformer model (local, no API key needed)
    4. Store the resulting vectors in a FAISS index
    5. Persist the index to disk at data/faiss_index/

Usage:
    python scripts/rebuild_index.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

# Disable XetHub to avoid 403 errors and suppress SSL warnings
os.environ["HF_HUB_DISABLE_XET"] = "1"
import httpx
original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init
# Also suppress warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

load_dotenv()
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

# ── Configuration ──────────────────────────────────────────────────────────────

PROCESSED_DATA_PATH = "data/processed/events_structured.csv"
FAISS_INDEX_PATH = "data/faiss_index"

# Model used to generate embeddings (runs locally, no API key required)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings: each text is split into chunks of ~500 characters,
# with a 50-character overlap to preserve context across chunk boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ── Functions ──────────────────────────────────────────────────────────────────

def load_events(csv_path: str) -> pd.DataFrame:
    """
    Loads the processed events CSV file into a DataFrame.
    Drops rows where the 'content' column is empty, as they cannot be vectorized.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed data not found at '{csv_path}'. "
            "Please run scripts/fetch_data.py and scripts/process_data.py first."
        )
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    df.dropna(subset=["content"], inplace=True)
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  [Warning] Dropped {dropped} rows with missing 'content' field.")
    print(f"  Loaded {len(df)} events from '{csv_path}'.")
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Converts each row of the DataFrame into a LangChain Document object.
    Metadata (uid, title, city, url) is stored alongside the text so that
    search results can be traced back to their source event.
    """
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
    print(f"  Created {len(docs)} LangChain Document objects.")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Splits each document into smaller chunks using a recursive character splitter.
    This ensures that large descriptions don't exceed the embedding model's context window
    and that each chunk remains semantically meaningful.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # Try to split at natural boundaries
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_faiss_index(chunks: list[Document], index_path: str) -> None:
    """
    Embeds each chunk and stores the resulting vectors in a FAISS index.
    The index is then saved to disk so it can be loaded by the API without
    re-running the full embedding pipeline every time.
    """
    print(f"  Loading embedding model: '{EMBEDDING_MODEL_NAME}' ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("  Building FAISS index (this may take a moment)...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"  FAISS index saved to '{index_path}'.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1/4 — Loading processed events...")
    df = load_events(PROCESSED_DATA_PATH)

    print("Step 2/4 — Building LangChain documents...")
    documents = build_documents(df)

    print("Step 3/4 — Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("Step 4/4 — Embedding chunks and building FAISS index...")
    build_faiss_index(chunks, FAISS_INDEX_PATH)

    print("=" * 60)
    print("✅ Index successfully built and saved.")
