import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def test_gcp_connection():
    project_id = os.getenv("GCP_PROJECT_ID")
    bucket_name = os.getenv("GCP_BUCKET")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    print("--- GCP Storage & Service Account Test ---")
    print(f"Project ID : {project_id}")
    print(f"Bucket     : {bucket_name}")
    print(f"Credentials: {creds_path}")
    print("-" * 42)

    try:
        # Initialize the GCS client (picks up GOOGLE_APPLICATION_CREDENTIALS automatically)
        client = storage.Client(project=project_id)

        # 1. Verify bucket existence and permissions
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Successfully connected to bucket: {bucket.name}")

        # 2. Test uploading a temporary file
        blob_name = "connection-test.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string("Hello from the Meta generator backend test!")
        print(f"✅ Successfully uploaded test object: {blob_name}")

        # 3. Test reading the file back
        content = blob.download_as_text()
        print(f"✅ Successfully read back object content: '{content}'")

        # 4. Clean up test file
        blob.delete()
        print(f"✅ Successfully cleaned up test object: {blob_name}")

        print("\n🎉 All GCP storage checks passed successfully! Your backend is fully wired.")

    except Exception as e:
        print(f"\n❌ GCP Connection Test Failed: {e}")

if __name__ == "__main__":
    test_gcp_connection()