from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Puls-Events RAG API",
    description="API for the cultural event RAG assistant.",
    version="1.0.0",
)

# Initialize RAG system
# We do this globally so it's loaded once when the app starts
try:
    rag = RAGSystem()
except Exception as e:
    print(f"Warning: RAG system could not be initialized: {e}")
    rag = None


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
async def root():
    return {
        "message": "Welcome to Puls-Events RAG API. Use /ask to query the assistant."
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized. Please ensure the FAISS index exists.",
        )

    try:
        answer = rag.ask(request.question)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_index():
    """
    Triggers the indexing pipeline to rebuild the FAISS index from current data.
    """
    try:
        # Import and run the rebuild logic
        # We run it as a subprocess to keep the API responsive or just import the functions
        from scripts.rebuild_index import (
            load_events,
            build_documents,
            split_documents,
            build_faiss_index,
            PROCESSED_DATA_PATH,
            FAISS_INDEX_PATH,
        )

        df = load_events(PROCESSED_DATA_PATH)
        documents = build_documents(df)
        chunks = split_documents(documents)
        build_faiss_index(chunks, FAISS_INDEX_PATH)

        # Reload the RAG system
        global rag
        rag = RAGSystem()

        return {"message": "Index successfully rebuilt and RAG system reloaded."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to rebuild index: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
