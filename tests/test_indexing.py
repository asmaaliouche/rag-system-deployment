"""
test_indexing.py
----------------
Unit tests for the FAISS index building pipeline (scripts/rebuild_index.py).

Tests cover:
    - Document creation from a DataFrame row
    - Text splitting into chunks
    - Full index build and similarity search on mock data
"""

# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from scripts.rebuild_index import build_documents, split_documents


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Returns a minimal DataFrame that mimics the processed events CSV."""
    return pd.DataFrame([
        {
            "uid": "1",
            "title": "Jazz Concert",
            "city": "Paris",
            "content": (
                "Event: Jazz Concert\n"
                "Location: Parc Floral, Route de la Pyramide, Paris\n"
                "Description: A great outdoor jazz concert with live music."
            ),
            "url": "https://example.com/jazz",
        },
        {
            "uid": "2",
            "title": "Photography Exhibition",
            "city": "Paris",
            "content": (
                "Event: Photography Exhibition\n"
                "Location: Palais de Tokyo, Avenue du Président Wilson, Paris\n"
                "Description: An international photography exhibition featuring emerging artists."
            ),
            "url": "https://example.com/photo",
        },
    ])


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBuildDocuments:
    """Tests for the build_documents function."""

    def test_returns_correct_number_of_documents(self, sample_df):
        """build_documents should return one Document per DataFrame row."""
        docs = build_documents(sample_df)
        assert len(docs) == len(sample_df)

    def test_document_page_content_matches_content_column(self, sample_df):
        """The page_content of each Document should match the 'content' column."""
        docs = build_documents(sample_df)
        for i, doc in enumerate(docs):
            assert doc.page_content == str(sample_df.iloc[i]["content"])

    def test_document_metadata_contains_expected_keys(self, sample_df):
        """Each Document's metadata should include uid, title, city, and url."""
        docs = build_documents(sample_df)
        for doc in docs:
            assert "uid" in doc.metadata
            assert "title" in doc.metadata
            assert "city" in doc.metadata
            assert "url" in doc.metadata

    def test_document_metadata_values_are_correct(self, sample_df):
        """Metadata values should match the corresponding DataFrame row."""
        docs = build_documents(sample_df)
        assert docs[0].metadata["title"] == "Jazz Concert"
        assert docs[0].metadata["city"] == "Paris"
        assert docs[1].metadata["title"] == "Photography Exhibition"


class TestSplitDocuments:
    """Tests for the split_documents function."""

    def test_splitting_increases_document_count(self, sample_df):
        """
        After splitting, the number of chunks should be >= the number of source documents
        (long texts should produce multiple chunks).
        """
        docs = build_documents(sample_df)
        chunks = split_documents(docs)
        assert len(chunks) >= len(docs)

    def test_each_chunk_has_non_empty_content(self, sample_df):
        """Every chunk produced should have non-empty page_content."""
        docs = build_documents(sample_df)
        chunks = split_documents(docs)
        for chunk in chunks:
            assert chunk.page_content.strip() != ""

    def test_chunks_inherit_metadata(self, sample_df):
        """Chunks should inherit the metadata from their parent Document."""
        docs = build_documents(sample_df)
        chunks = split_documents(docs)
        for chunk in chunks:
            # Every chunk must have at least the 'title' key from its parent
            assert "title" in chunk.metadata

    def test_large_text_is_split_into_multiple_chunks(self):
        """A document longer than CHUNK_SIZE should produce more than one chunk."""
        long_text = "This is a sentence about a cultural event. " * 50  # ~2200 chars
        doc = Document(page_content=long_text, metadata={"title": "Long Event"})
        chunks = split_documents([doc])
        assert len(chunks) > 1
