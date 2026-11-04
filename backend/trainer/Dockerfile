# LoRA trainer image for a Google Cloud Run GPU job.
# Build + deploy (from backend/):
#   gcloud builds submit trainer --tag gcr.io/$PROJECT/lora-trainer
#   gcloud run jobs deploy lora-trainer --image gcr.io/$PROJECT/lora-trainer \
#     --gpu 1 --gpu-type nvidia-l4 --memory 32Gi --cpu 8 \
#     --task-timeout 3600 --region $REGION
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 diffusers==0.32.2 \
    transformers==4.47.1 accelerate==1.2.1 peft==0.14.0 safetensors \
    Pillow google-cloud-storage

RUN git clone --depth 1 https://github.com/huggingface/diffusers /opt/diffusers

COPY train_entry.py /app/train_entry.py
WORKDIR /app
ENTRYPOINT ["python3", "train_entry.py"]
