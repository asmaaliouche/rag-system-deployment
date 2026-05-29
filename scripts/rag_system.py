"""
rag_system.py
-------------
Defines the RAGSystem class that integrates FAISS for retrieval and Mistral for generation.

Pipeline:
    1. Load local FAISS index.
    2. Retrieve top-k events based on the user's question.
    3. Pass the events context to the Mistral LLM to generate a personalized answer.
"""

import os
from dotenv import load_dotenv

# Disable XetHub to avoid 403 errors and suppress SSL warnings for local environment
os.environ["HF_HUB_DISABLE_XET"] = "1"
import httpx
original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Using langchain_classic as per environment structure
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGSystem:
    def __init__(self, index_path=FAISS_INDEX_PATH, mistral_api_key=None):
        api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key or api_key == "your_mistral_api_key_here":
            raise ValueError("Valid MISTRAL_API_KEY is not set in .env")
        
        # 1. Load local embeddings and FAISS index
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run scripts/rebuild_index.py first.")
            
        self.vector_store = FAISS.load_local(
            index_path, 
            self.embeddings,
            allow_dangerous_deserialization=True # Required by FAISS to load local pickles securely
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 2. Initialize Mistral LLM
        self.llm = ChatMistralAI(
            mistral_api_key=api_key,
            model="open-mistral-nemo", # Lightweight model for POC
            temperature=0.2
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
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
        print("Initializing RAG System...")
        rag = RAGSystem()
        test_question = "Quels sont les événements musicaux ou concerts prévus ?"
        print(f"\nQuestion: {test_question}")
        print("Thinking...")
        answer = rag.ask(test_question)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
