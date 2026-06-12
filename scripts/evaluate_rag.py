import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from scripts.rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()

# We'll use the existing RAGSystem to generate answers and contexts
def run_evaluation():
    print("Loading RAG system...")
    rag = RAGSystem()
    
    print("Loading evaluation set...")
    with open("data/evaluation_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print(f"Running {len(eval_set)} questions through the RAG system...")
    for item in eval_set:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # Get answer and retrieved documents
        # Note: We need the context used for the answer. 
        # In rag_system.py, the rag_chain returns 'answer' and 'context'
        response = rag.rag_chain.invoke({"input": question})
        
        questions.append(question)
        answers.append(response["answer"])
        # context should be a list of strings
        contexts.append([doc.page_content for doc in response["context"]])
        ground_truths.append(ground_truth)
    
    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("Starting Ragas evaluation...")
    # This will use the Mistral LLM via LangChain if configured, 
    # but Ragas often defaults to OpenAI. 
    # We might need to configure Ragas to use Mistral for evaluation too.
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("\nEvaluation Results:")
    print(result)
    
    # Save results to CSV
    df = result.to_pandas()
    df.to_csv("data/evaluation_results.csv", index=False)
    print("\nResults saved to data/evaluation_results.csv")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Error during evaluation: {e}")
