"""Cloud Run job entrypoint: train one character LoRA, upload adapter.

Env:
  DATASET_GCS  gs://bucket/datasets/<character>.zip (000.png/000.txt)
  OUTPUT_GCS   gs://bucket/adapters/<character>/  (adapter destination)
  TRIGGER      the character's trigger word, e.g. norman_tok
"""
import os
import subprocess
import sys
import zipfile

from google.cloud import storage

BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), uri
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


def main() -> None:
    dataset_uri = os.environ["DATASET_GCS"]
    output_uri = os.environ["OUTPUT_GCS"]
    trigger = os.environ["TRIGGER"]
    client = storage.Client()

    bucket_name, blob_name = parse_gs(dataset_uri)
    zip_path = "/tmp/dataset.zip"
    client.bucket(bucket_name).blob(blob_name).download_to_filename(zip_path)
    data_dir = "/tmp/data"
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)

    out_dir = "/tmp/out"
    cmd = [
        "accelerate", "launch",
        "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py",
        f"--pretrained_model_name_or_path={BASE}",
        f"--instance_data_dir={data_dir}",
        f"--instance_prompt=a photo of {trigger}",
        "--resolution=1024", "--train_batch_size=1",
        "--gradient_accumulation_steps=4", "--learning_rate=1e-4",
        "--lr_scheduler=constant", "--max_train_steps=800", "--rank=16",
        f"--output_dir={out_dir}", "--seed=42",
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    bucket_name, prefix = parse_gs(output_uri)
    bucket = client.bucket(bucket_name)
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            local = os.path.join(root, name)
            rel = os.path.relpath(local, out_dir)
            bucket.blob(f"{prefix}{rel}").upload_from_filename(local)
    print(f"adapter uploaded to {output_uri}")


if __name__ == "__main__":
    sys.exit(main())
