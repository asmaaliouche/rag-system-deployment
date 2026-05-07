# Puls-Events RAG POC

This project is a Proof of Concept (POC) for an intelligent assistant capable of recommending cultural events using the RAG (Retrieval-Augmented Generation) technique.

## Project Structure

- `api/`: REST API code (FastAPI).
- `data/`: Raw and processed data (Open Agenda).
- `docs/`: Technical report and presentation.
- `scripts/`: Scripts for data cleaning, vectorization, and building the index.
- `tests/`: Unit tests and system evaluation.
- `pyproject.toml`: Dependency management with Poetry.

## Installation

1. Clone the repository.
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
3. Activate the virtual environment:
   ```bash
   poetry shell
   ```
4. Set up environment variables in a `.env` file:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

## Usage

(To be completed during development)

- Rebuild index: `python scripts/rebuild_index.py`
- Start the API: `uvicorn api.main:app --reload`
- Run tests: `pytest`


---Tester le modele: en changement le format des dates, et rajouter des erreurs---
---verifier les vigilences: les vides--