import sys
import requests

BASE_URL = "http://localhost:8000"


def run_test_root():
    print("Testing GET / ...")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}"
        data = response.json()
        assert (
            "Welcome to Puls-Events RAG API" in data["message"]
        ), "Unexpected response message"
        print("✅ GET / passed successfully.")
    except Exception as e:
        print(f"❌ GET / failed: {e}")
        return False
    return True


def run_test_ask():
    print("\nTesting POST /ask ...")
    # We use a French question because the ingested cultural event data from OpenAgenda is in French
    payload = {"question": "Où se déroule le concert de jazz ?"}
    try:
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "question" in data, "Response missing 'question'"
        assert "answer" in data, "Response missing 'answer'"
        print(f"Question sent: {payload['question']}")
        print(f"Response answer: {data['answer']}")
        print("✅ POST /ask passed successfully.")
    except Exception as e:
        print(f"❌ POST /ask failed: {e}")
        return False
    return True


if __name__ == "__main__":
    print("====================================================")
    print("Puls-Events RAG API - Functional Test Script")
    print("Ensure the API is running locally (e.g. uvicorn or Docker)")
    print(f"Target URL: {BASE_URL}")
    print("====================================================\n")

    success = True
    success &= run_test_root()
    success &= run_test_ask()

    if success:
        print("\n🎉 All functional tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please verify that the API is running and healthy.")
        sys.exit(1)
