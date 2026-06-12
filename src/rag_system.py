"""
rag_system.py
-------------
Defines the RAGSystem class that integrates FAISS for retrieval and Mistral for generation.
"""

import logging
import os
import httpx
import urllib3
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable XetHub
os.environ["HF_HUB_DISABLE_XET"] = "1"


def patch_httpx_ssl(client_class):
    """
    Surgically patches httpx classes to ignore SSL verification.
    """
    original_init = client_class.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        original_init(self, *args, **kwargs)

    client_class.__init__ = patched_init


patch_httpx_ssl(httpx.Client)
patch_httpx_ssl(httpx.AsyncClient)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_NAME = "mistral-embed"


class RAGSystem:
    def __init__(self, index_path=FAISS_INDEX_PATH, mistral_api_key=None):
        api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key or api_key == "your_mistral_api_key_here":
            raise ValueError("Valid MISTRAL_API_KEY is not set in .env")

        # 1. Initialize Mistral embeddings (Cloud-based, no local model download)
        self.embeddings = MistralAIEmbeddings(
            mistral_api_key=api_key, model=EMBEDDING_MODEL_NAME
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Please run scripts/rebuild_index.py first."
            )

        self.vector_store = FAISS.load_local(
            index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # 2. Initialize Mistral LLM
        self.llm = ChatMistralAI(
            mistral_api_key=api_key, model="open-mistral-nemo", temperature=0.2
        )

        # 3. Create the RAG chain
        system_prompt = (
            "You are an intelligent cultural event assistant for the company Puls-Events. "
            "Use the following pieces of retrieved context to answer the user's question about cultural events. "
            "If you don't know the answer or if there are no matching events, say that you don't have information about that right now, but do not make up events. "
            "Keep the answer concise, informative, and friendly. Answer in the same language as the user's question (default to French)."
            "\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system and get a generated response based on the FAISS index.
        """
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]


if __name__ == "__main__":
    try:
        logger.info("Initializing RAG System...")
        rag = RAGSystem()
        test_question = "Quels sont les événements musicaux ou concerts prévus ?"
        logger.info(f"\nQuestion: {test_question}")
        logger.info("Thinking...")
        answer = rag.ask(test_question)
        logger.info(f"\nAnswer: {answer}")
    except Exception as e:
        logger.error(f"Error: {e}")
