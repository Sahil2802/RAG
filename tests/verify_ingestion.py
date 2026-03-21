import httpx
import time
import sys

BASE_URL = "http://localhost:8000/api"


def verify_ingestion():
    print("🚀 Starting Ingestion Verification...")

    # 1. Health check
    try:
        response = httpx.get(f"{BASE_URL.replace('/api', '')}/health")
        if response.status_code == 200:
            print("✅ Server is UP")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        return

    # 2. Create test file
    test_content = "The capital of France is Paris. It is known for the Eiffel Tower."
    test_filename = "test_ingest.txt"
    with open(test_filename, "w") as f:
        f.write(test_content)

    # 3. Trigger Ingest
    print(f"📤 Uploading {test_filename}...")
    with open(test_filename, "rb") as f:
        files = {"file": (test_filename, f, "text/plain")}
        # Increase timeout as the first request might trigger model loading
        response = httpx.post(f"{BASE_URL}/ingest", files=files, timeout=60.0)

    if response.status_code not in [200, 202]:
        print(f"❌ Ingestion request failed: {response.text}")
        return

    data = response.json()
    doc_id = data["document_id"]
    status = data["status"]
    print(f"📄 Document ID: {doc_id} (Initial status: {status})")

    if status == "duplicate":
        print("⚠️ File already exists, skipping polling.")
        return

    # 4. Poll for status
    print("⏳ Polling for 'ready' status...")
    max_retries = 20
    for i in range(max_retries):
        resp = httpx.get(f"{BASE_URL}/documents/{doc_id}")
        if resp.status_code == 200:
            doc_data = resp.json()
            current_status = doc_data["status"]
            print(f"   [{i+1}/{max_retries}] Status: {current_status}")
            if current_status == "ready":
                print("✅ Ingestion SUCCESSFUL!")
                print(f"   Chunks: {doc_data.get('chunk_count')}")
                return
            if current_status == "failed":
                print(f"❌ Ingestion FAILED: {doc_data.get('error_message')}")
                return
        else:
            print(f"❌ Failed to get status: {resp.text}")
            return
        time.sleep(2)

    print("❌ Ingestion timed out (still processing).")


if __name__ == "__main__":
    verify_ingestion()
