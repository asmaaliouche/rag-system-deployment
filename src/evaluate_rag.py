"""
evaluate_rag.py
---------------
Automated evaluation script for the RAG system using the Ragas framework.
It tests the generated answers against a ground truth dataset and computes
metrics like Faithfulness, Answer Relevancy, Context Precision, and Recall.
"""

import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path so we can import 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from scripts.rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()


# We'll use the existing RAGSystem to generate answers and contexts
def run_evaluation():
    logger.info("Loading RAG system...")
    rag = RAGSystem()

    logger.info("Loading evaluation set...")
    with open("data/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    logger.info(f"Running {len(eval_set)} questions through the RAG system...")
    for item in eval_set:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get answer and retrieved documents
        response = rag.rag_chain.invoke({"input": question})

        questions.append(question)
        answers.append(response["answer"])
        contexts.append([doc.page_content for doc in response["context"]])
        ground_truths.append(ground_truth)

    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    logger.info("Starting Ragas evaluation (using Mistral)...")

    evaluator_llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-large-latest"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed")
    )

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    logger.info("\nEvaluation Results:")
    logger.info(result)

    # Save results to CSV
    df = result.to_pandas()
    df.to_csv("data/evaluation_results.csv", index=False)
    logger.info("\nResults saved to data/evaluation_results.csv")


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
