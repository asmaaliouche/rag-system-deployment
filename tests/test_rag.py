"""
test_rag.py
-----------
Unit tests for the RAG system logic.
Uses mocks to avoid making actual API calls to Mistral or requiring a built FAISS index.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.rag_system import RAGSystem


@patch("src.rag_system.ChatMistralAI")
@patch("src.rag_system.FAISS.load_local")
@patch("src.rag_system.MistralAIEmbeddings")
@patch("os.path.exists", return_value=True)
def test_rag_ask_method(mock_exists, mock_embeddings, mock_faiss, mock_llm):
    """
    Tests that the RAGSystem successfully invokes the LangChain pipeline
    and returns the expected answer.
    """
    # Mock the RAG chain invoke method
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {
        "answer": "Il y a un concert de jazz ce soir au Parc Floral."
    }

    with patch("src.rag_system.create_retrieval_chain", return_value=mock_rag_chain):
        rag = RAGSystem(mistral_api_key="fake-key")
        answer = rag.ask("Quels sont les événements de jazz ?")

        # Verify the chain was called with the right input
        mock_rag_chain.invoke.assert_called_once_with(
            {"input": "Quels sont les événements de jazz ?"}
        )
        # Verify the answer
        assert "concert de jazz" in answer


def test_rag_missing_api_key():
    """
    Tests that a ValueError is raised if the Mistral API key is missing or invalid.
    """
    with pytest.raises(ValueError, match="Valid MISTRAL_API_KEY is not set"):
        RAGSystem(mistral_api_key="your_mistral_api_key_here")
