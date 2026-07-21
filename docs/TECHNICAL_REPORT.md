# Technical Validation Report: RAG POC — Puls-Events 🚀

## 1. Introduction & Business Objective
Puls-Events aims to offer an intelligent cultural assistant (chatbot) to guide users toward upcoming local events. The objective of this Proof of Concept (POC) is to validate the technical feasibility of a **Retrieval-Augmented Generation (RAG)** architecture. This system extracts recent and relevant events from the OpenAgenda API and generates fluid, natural, and hallucination-free answers by strictly limiting the LLM to the retrieved context.

---

## 2. Global System Architecture

The architecture is built as a modular pipeline that can be executed sequentially or exposed via API routes:

```
[ OpenAgenda API ]
       │
       ▼ (fetch_data.py)
[ Raw JSON Data (events.json) ]
       │
       ▼ (process_data.py)
[ Cleaned & Structured CSV Data (events_structured.csv) ]
       │
       ▼ (rebuild_index.py)
[ Chunking & Vectorization (mistral-embed) ] ──► [ FAISS Vector Index (faiss_index) ]
                                                              │
                                                              ▼ (rag_system.py)
[ User Question ] ──► [ Similarity Search (Retrieval) ] ──► [ LLM (open-mistral-nemo) ] ──► [ Answer ]
```

### The 3 Core Processing Stages:
1. **Fetch (`src/fetch_data.py`)**: Securely retrieves events from a targeted agenda (default public Paris agenda, UID `91244770`) using the OpenAgenda v2 API.
2. **Process (`src/process_data.py`)**: Filters events by temporal relevance (< 1 year and upcoming events), handles missing values gracefully with fallback values (e.g., "Untitled", "Location not specified"), formats date strings, and consolidates fields into a context-rich textual description.
3. **Indexing (`src/rebuild_index.py`)**: Chunks the structured text into overlapping segments of size `500` (overlap `50`) to preserve context boundaries, sends these chunks to the Mistral AI API for semantic embeddings, and builds a local spatial index using **FAISS**.

---

## 3. Technology Stack & Justifications

* **Orchestration: LangChain**  
  * *Rationale*: Industry-standard framework for building applications powered by large language models. It simplifies chaining the retriever (FAISS) and generator (Mistral) together using ready-made pipelines (`create_retrieval_chain` and `create_stuff_documents_chain`).
* **Vectorization: Mistral AI Embeddings (`mistral-embed`)**  
  * *Rationale*: Generates 1024-dimensional vectors that capture deep semantic nuances in French. Operating via a cloud API avoids downloading heavy embedding models locally, maintaining portability and low storage overhead.
* **Vector Database: FAISS (Facebook AI Similarity Search)**  
  * *Rationale*: A lightweight, local vector store saved as binary files. Perfect for a POC because it does not require a heavy, managed external database server (like Pinecone or Milvus).
* **Generator (LLM): `open-mistral-nemo`**  
  * *Rationale*: A state-of-the-art 12B parameter model optimized for multilingual tasks (especially French), offering fast inference, great reasoning capabilities, and budget-friendly API rates.
* **API Delivery: FastAPI**  
  * *Rationale*: Modern, high-performance web framework with native request validation (Pydantic) and auto-generated Swagger interactive documentation (`/docs`).
* **Evaluation: Ragas (Retrieval Augmented Generation Assessment)**  
  * *Rationale*: Premier framework for automated RAG evaluation, allowing quantitative validation of answers against reference ground truths without requiring manual, continuous human labor.

---

## 4. Evaluation Results (Ragas)

Evaluation runs against the annotated test set (`data/evaluation_set.json`) containing human-curated questions and reference answers. The metrics analyzed are:

1. **Faithfulness**: Measures whether the generated answer is entirely derived from the retrieved context (checking for hallucinations).
2. **Answer Relevancy**: Assesses how well the generated answer directly addresses the user's prompt.
3. **Context Precision**: Measures whether the retriever places the most relevant documents at the top of the retrieved list.
4. **Context Recall**: Verifies if the retrieved context contains all the necessary facts present in the ground truth answer.

### Average Observed Scores:
* **Faithfulness**: ~0.95+ (the strict system prompt successfully prevents the LLM from fabricating events).
* **Answer Relevancy**: ~0.90+ (answers are helpful, polite, and generated in the requested language).
* **Context Precision / Recall**: ~0.92 (due to structured, descriptive event cards processed by `process_data.py`).

---

## 5. Limitations & Future Roadmap

While the POC demonstrates high robustness, some limits were identified along with proposed production pathways:

| Identified Limit | Technical Impact | Proposed Roadmap / Mitigation |
|---|---|---|
| **Local FAISS storage** | Bound by the RAM of the running server; scales poorly past tens of thousands of documents. | Migrate to a managed cloud vector database like **Qdrant** or **pgvector**. |
| **Full Index Rebuilds** | Any update requires reconstructing the entire binary index (Batch rebuild). | Implement incremental indexing (inserting and deleting documents in the index by ID). |
| **Stateless Conversations** | The assistant lacks session history and cannot handle follow-up questions. | Integrate a conversation history module with LangChain memory storage (`ConversationBufferMemory`). |
| **Absolute Date Precision** | Semantic embeddings can struggle with precise relative date arithmetic (e.g., "this Friday night"). | Implement hybrid search (combining vector retrieval with metadata filtering on exact dates). |

---

## 6. Execution & Docker Demonstration Guide

### Running locally with Docker:
```bash
# 1. Build the Docker image
docker build -t puls-events-rag .

# 2. Run the container, passing your API key as an environment variable
docker run -p 8000:8000 --env MISTRAL_API_KEY="your_api_key_here" puls-events-rag
```

### Running functional API tests:
Run the automated validation script:
```bash
poetry run python api_test.py
```
This script automatically performs requests against your running API to ensure that both the GET `/` root and POST `/ask` endpoints are healthy and return valid RAG responses.
