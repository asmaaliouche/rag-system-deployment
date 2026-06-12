# Puls-Events RAG POC 🚀

This project is a Proof of Concept (POC) for an intelligent cultural assistant. It uses **RAG (Retrieval-Augmented Generation)** to recommend events from the Open Agenda platform based on semantic search.

Built for the **OpenClassrooms** AI Engineer path.

---

## 🛠 Tech Stack

- **Language:** Python 3.10
- **Framework:** FastAPI (REST API)
- **Orchestration:** LangChain
- **LLM:** Mistral AI (Open Mistral Nemo)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Environment:** Poetry & Docker
- **Evaluation:** Ragas

---

## 📋 Prerequisites

- **Python 3.10+**
- **Poetry** (Dependency manager)
- **Docker** (for containerized deployment)
- **Mistral API Key** (Get one at [console.mistral.ai](https://console.mistral.ai/))

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone <https://github.com/asmaaliouche/rag-system-deployment.git>
cd puls-events-rag-poc
cp .env.template .env
```
*Edit `.env` and add your `MISTRAL_API_KEY`.*

### 2. Install Dependencies
```bash
poetry install
```

### 3. Initialize the Vector Index
Before running the system, you must fetch the data and build the FAISS index:
```bash
# 1. Fetch data from OpenAgenda
poetry run python scripts/fetch_data.py

# 2. Process data
poetry run python scripts/process_data.py

# 3. Build the index
poetry run python scripts/rebuild_index.py
```

---

## 🖥 Usage

### Running Locally
```bash
poetry run uvicorn api.main:app --reload
```
The API will be available at `http://localhost:8000`.

### Running with Docker
```bash
# Build the image
docker build -t puls-events-rag .

# Run the container
docker run -p 8000:8000 --env-file .env puls-events-rag
```

---

## 📡 API Documentation (Swagger)

FastAPI automatically generates documentation. Once the API is running, go to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### How to test the chatbot:
1. Locate the **`POST /ask`** endpoint.
2. Click **"Try it out"**.
3. In the Request Body, replace `"string"` with a full question:
   ```json
   {
     "question": "Quels sont les concerts de jazz prévus à Paris ?"
   }
   ```
   *(Note: The RAG system requires full questions rather than single keywords like "concert" to provide the best, most contextualized answers).*
4. Click **"Execute"** to see the AI's response.

---

## 📊 Evaluation

To measure the quality of the RAG system (Faithfulness, Relevancy, etc.) using **Ragas**:
```bash
poetry run python scripts/evaluate_rag.py
```
Results will be saved in `data/evaluation_results.csv`.

---

## 🧪 Testing
Run unit tests with:
```bash
poetry run pytest
```

---

## 📂 Project Structure

- `api/`: FastAPI routes and logic.
- `data/`: Local storage for datasets and FAISS index.
- `docs/`: Technical documentation and presentation slides for the POC.
- `scripts/`: Core logic (Data fetching, Processing, Indexing, RAG system, Evaluation).
- `tests/`: Unit tests.
- `Dockerfile`: Container configuration.
- `pyproject.toml`: Poetry dependencies.