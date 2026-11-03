"""LoRA training orchestration.

The default Modal path sends a dataset zip to the deployed
`backend/modal_lora_app.py` service, queues GPU training, polls its call
id, and generates through the stored adapter. `TRAINING_PROVIDER=gcp`
preserves the older path: dataset zip -> Cloud Storage -> Cloud Run GPU
job (the trainer container in backend/trainer/) -> adapter files back in
Cloud Storage. A missing provider configuration refuses honestly with
setup steps instead of pretending to train.

Modal configuration (backend/.env or environment):
  MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, MODAL_APP_NAME, MODAL_VOLUME_NAME,
  MODAL_BASE_MODEL, and one-time `modal deploy modal_lora_app.py`.

GCP configuration (backend/.env or environment):
  GCP_PROJECT_ID, GCP_REGION (default us-central1), GCP_BUCKET,
  GOOGLE_APPLICATION_CREDENTIALS (service-account JSON path), and the
  one-time `gcloud run jobs deploy lora-trainer ...` from trainer/.
"""
import os
from pathlib import Path

from . import dataset, modal_backend
from .config import settings


class TrainingNotConfigured(Exception):
    pass


ModalTrainingError = modal_backend.ModalTrainingError


def training_provider() -> str:
    return _setting("TRAINING_PROVIDER", "modal").lower() or "modal"


def _setting(name: str, default: str = "") -> str:
    from_env = os.environ.get(name, "").strip()
    if from_env:
        return from_env
    return str(getattr(settings, name.lower(), default) or default).strip()


def gcp_configured() -> bool:
    creds = _setting("GOOGLE_APPLICATION_CREDENTIALS")
    creds_ok = bool(creds and Path(creds).exists())
    return bool(_setting("GCP_PROJECT_ID") and _setting("GCP_BUCKET")
                and creds_ok)


def submit_training(asset: dict, zip_bytes: bytes, options: dict | None = None) -> dict:
    """Submit training to the configured provider (Modal by default)."""
    provider = training_provider()
    if provider == "modal":
        try:
            return modal_backend.submit_training(asset, zip_bytes, options)
        except modal_backend.ModalNotConfigured as e:
            raise TrainingNotConfigured(str(e)) from e
    if provider == "gcp":
        return _submit_gcp(asset, zip_bytes)
    raise TrainingNotConfigured(
        f"Unsupported TRAINING_PROVIDER={provider!r}; use 'modal' or 'gcp'.")


def _submit_gcp(asset: dict, zip_bytes: bytes) -> dict:
    """Upload the dataset and execute the Cloud Run trainer job."""
    if not gcp_configured():
        raise TrainingNotConfigured(
            "Google Cloud is not connected. Set GCP_PROJECT_ID, GCP_BUCKET "
            "and GOOGLE_APPLICATION_CREDENTIALS (service-account JSON) in "
            "backend/.env, create the bucket, deploy the trainer once "
            "(see trainer/Dockerfile header), and retry.")
    try:
        from google.cloud import run_v2, storage
    except ImportError as e:
        raise TrainingNotConfigured(
            "Google Cloud libraries are not installed in backend/.venv "
            "(`pip install google-cloud-run google-cloud-storage`).") from e

    creds = _setting("GOOGLE_APPLICATION_CREDENTIALS")
    if creds and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    project = _setting("GCP_PROJECT_ID")
    region = _setting("GCP_REGION", "us-central1")
    bucket_name = _setting("GCP_BUCKET")
    name = asset.get("name", "character")
    slug = dataset.trigger_for(name).removesuffix("_tok")
    trigger = dataset.trigger_for(name)

    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    dataset_blob = f"datasets/{slug}.zip"
    bucket.blob(dataset_blob).upload_from_string(
        zip_bytes, content_type="application/zip")
    adapter_prefix = f"adapters/{slug}/"

    jobs = run_v2.JobsClient()
    job_name = f"projects/{project}/locations/{region}/jobs/lora-trainer"
    request = run_v2.RunJobRequest(
        name=job_name,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[
                    run_v2.EnvVar(name="DATASET_GCS",
                                  value=f"gs://{bucket_name}/{dataset_blob}"),
                    run_v2.EnvVar(name="OUTPUT_GCS",
                                  value=f"gs://{bucket_name}/{adapter_prefix}"),
                    run_v2.EnvVar(name="TRIGGER", value=trigger),
                ])]),
    )
    operation = jobs.run_job(request=request)
    return {"status": "training",
            "adapter_ref": f"gs://{bucket_name}/{adapter_prefix}",
            "trigger": trigger,
            "operation": getattr(operation.operation, "name", "")}


def training_status(asset: dict, call_id: str = "") -> dict:
    """Poll the configured provider for an in-flight training job."""
    if training_provider() != "modal":
        return {"status": asset.get("train_status", "none"), "call_id": call_id}
    return modal_backend.training_status(call_id)


def generate_with_adapter(adapter_ref: str, prompt: str, seed: int = 42,
                          steps: int = 30, guidance_scale: float = 7.0,
                          lora_scale: float = 1.0,
                          negative_prompt: str = "") -> dict:
    """Generate one image from a stored adapter on the configured provider."""
    if training_provider() != "modal":
        raise TrainingNotConfigured(
            "Adapter generation requires TRAINING_PROVIDER=modal.")
    return modal_backend.generate_with_adapter(
        adapter_ref, prompt, seed=seed, steps=steps,
        guidance_scale=guidance_scale, lora_scale=lora_scale,
        negative_prompt=negative_prompt)
