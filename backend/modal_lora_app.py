"""Deployable Modal training service for per-character LoRAs.

Manual deployment only (this module never spends credits on import)::

    modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface-secret HF_TOKEN=<huggingface-read-token>
    cd backend
    modal deploy modal_lora_app.py

The FastAPI backend calls the deployed ``start_training`` function, which
stores the uploaded dataset on the Modal Volume and spawns ``train_character``
on a GPU. Generation later loads the stored adapter in ``generate_character``.

The trainer itself is the public Ostris ai-toolkit repository, pinned below.
Our local zip format (``000.png``/``000.txt`` pairs) is exactly the folder
layout ai-toolkit expects, so no caption conversion is needed.
"""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import modal
import yaml

APP_NAME = os.getenv("MODAL_APP_NAME", "meta-lora-studio")
VOLUME_NAME = os.getenv("MODAL_VOLUME_NAME", "lora-studio")
BASE_MODEL = os.getenv(
    "MODAL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
)
TRAIN_GPU = os.getenv("MODAL_TRAIN_GPU", "A100")
GENERATE_GPU = os.getenv("MODAL_GENERATE_GPU", "A10")
HF_SECRET_NAME = os.getenv("MODAL_HF_SECRET", "huggingface-secret")
AI_TOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_REV = "e9c72b762748e4d31837fbeb3b87d01a5b143321"
AI_TOOLKIT_ROOT = "/opt/ai-toolkit"
VOLUME_ROOT = Path("/vol")
DATASET_ROOT = VOLUME_ROOT / "datasets"
ADAPTER_ROOT = VOLUME_ROOT / "adapters"
MAX_DATASET_ZIP_BYTES = 25 * 1024 * 1024
MAX_TRAIN_STEPS = 2000
MIN_TRAIN_STEPS = 100

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_TRIGGER_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*_tok$")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_TEXT_EXT = ".txt"


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
light_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pyyaml"
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "huggingface_hub",
        "pyyaml",
        "Pillow",
    )
    .run_commands(
        f"git clone {AI_TOOLKIT_REPO} {AI_TOOLKIT_ROOT}",
        f"cd {AI_TOOLKIT_ROOT} && git checkout {AI_TOOLKIT_REV}",
        # NOTE: never use sys.executable here — at deploy time it is the
        # *local* interpreter path (e.g. C:\... on Windows), which does not
        # exist inside the Linux build container.
        # uv's resolver is orders of magnitude faster than pip's on the torch
        # ecosystem, avoiding pathological backtracking on this large file.
        "python3 -m pip install uv",
        f"python3 -m uv pip install --system -r {AI_TOOLKIT_ROOT}/requirements_base.txt",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "DISABLE_TELEMETRY": "YES",
            "HF_HOME": "/vol/models/hf",
            "HUGGINGFACE_HUB_CACHE": "/vol/models/hf",
        }
    )
)
app = modal.App(APP_NAME)


def check_slug(slug: str) -> str:
    cleaned = (slug or "").strip().lower()
    if not _SLUG_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid Modal dataset slug: {slug!r}")
    return cleaned


def check_trigger(trigger: str) -> str:
    cleaned = (trigger or "").strip().lower()
    if not _TRIGGER_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid LoRA trigger word: {trigger!r}")
    return cleaned


def safe_dataset_files(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extract safe ``(filename, bytes)`` pairs from an uploaded zip."""
    if not zip_bytes:
        raise ValueError("The uploaded dataset zip is empty")
    if len(zip_bytes) > MAX_DATASET_ZIP_BYTES:
        raise ValueError("The uploaded dataset zip is larger than 25 MB")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded dataset is not a valid zip file") from exc

    with archive:
        files: dict[str, bytes] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if not name or name.startswith("."):
                continue
            stem, ext = os.path.splitext(name)
            if not stem or ext.lower() not in _IMAGE_EXTS | {_TEXT_EXT}:
                continue
            # Later same-named entries win; this mirrors a flat dataset folder.
            files[f"{stem}{ext.lower()}"] = archive.read(info.filename)

    images = {name for name in files if os.path.splitext(name)[1] in _IMAGE_EXTS}
    if not images:
        raise ValueError("The dataset zip contains no .png/.jpg/.jpeg images")
    missing = sorted(
        f"{os.path.splitext(name)[0]}{_TEXT_EXT}"
        for name in images
        if f"{os.path.splitext(name)[0]}{_TEXT_EXT}" not in files
    )
    if missing:
        raise ValueError(
            "Every training image needs a same-named caption file: "
            + ", ".join(missing)
        )
    return sorted(files.items())


def store_dataset_bytes(zip_bytes: bytes, slug: str) -> dict:
    slug = check_slug(slug)
    dataset_dir = DATASET_ROOT / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in dataset_dir.iterdir():
        if name.is_file():
            name.unlink()
    for name, data in safe_dataset_files(zip_bytes):
        (dataset_dir / name).write_bytes(data)
    volume.commit()
    images = sorted(
        name for name in os.listdir(dataset_dir)
        if os.path.splitext(name)[1] in _IMAGE_EXTS
    )
    return {
        "slug": slug,
        "dataset_dir": str(dataset_dir),
        "images": images,
        "image_count": len(images),
        "dataset_bytes": len(zip_bytes),
    }


def clamp_train_params(params: dict | None) -> dict:
    params = dict(params or {})
    steps = int(params.get("steps", 400))
    rank = int(params.get("rank", 16))
    learning_rate = float(params.get("learning_rate", 1e-4))
    resolution = int(params.get("resolution", 1024))
    return {
        "steps": max(MIN_TRAIN_STEPS, min(MAX_TRAIN_STEPS, steps)),
        "rank": max(4, min(64, rank)),
        "learning_rate": max(1e-6, min(1e-3, learning_rate)),
        "resolution": 1024 if resolution < 768 else min(1536, resolution),
        "seed": int(params.get("seed", 42)),
        "validation_prompt": str(params.get("validation_prompt") or "").strip()[:500],
    }


def training_config(
    slug: str,
    trigger: str,
    dataset_dir: str,
    params: dict | None = None,
    base_model: str = BASE_MODEL,
) -> dict:
    """Build an ai-toolkit ``sd_trainer`` config for SDXL character LoRAs."""
    slug = check_slug(slug)
    trigger = check_trigger(trigger)
    options = clamp_train_params(params)
    train = {
        "batch_size": 1,
        "steps": options["steps"],
        "gradient_accumulation_steps": 1,
        "train_unet": True,
        # Text-encoder training is what binds colour adjectives (red hat,
        # yellow shirt, green overalls) to the trigger word. UNet-only
        # LoRAs memorise silhouette but scramble attributes.
        "train_text_encoder": True,
        "gradient_checkpointing": True,
        "noise_scheduler": "ddpm",
        "optimizer": "adamw8bit",
        "lr": options["learning_rate"],
        "seed": options["seed"],
        "ema_config": {"use_ema": True, "ema_decay": 0.99},
        "dtype": "bf16",
    }
    if options["validation_prompt"]:
        sample = {
            "sampler": "ddpm",
            "sample_every": options["steps"],
            "sample_start_step": 0,
            "width": 1024,
            "height": 1024,
            "prompts": [f"[trigger], {options['validation_prompt']}"],
            "seed": options["seed"],
            "guidance_scale": 7.0,
            "sample_steps": 25,
        }
    else:
        sample = None

    process = {
        "type": "sd_trainer",
        "training_folder": str(ADAPTER_ROOT / slug),
        "device": "cuda:0",
        "trigger_word": trigger,
        "network": {
            "type": "lora",
            "linear": options["rank"],
            "linear_alpha": options["rank"],
        },
        "save": {
            "dtype": "float16",
            "save_every": options["steps"],
            "max_step_saves_to_keep": 1,
        },
        "datasets": [
            {
                "folder_path": dataset_dir,
                "caption_ext": "txt",
                "caption_dropout_rate": 0.05,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": [options["resolution"]],
            }
        ],
        "train": train,
        "model": {"name_or_path": base_model, "is_xl": True},
    }
    if sample is None:
        process["train"]["disable_sampling"] = True
    else:
        process["sample"] = sample
    return {
        "job": "extension",
        "config": {"name": f"{slug}-lora", "process": [process]},
    }


def find_adapter_file(adapter_dir: Path) -> Path:
    candidates = sorted(
        (path for path in adapter_dir.rglob("*.safetensors") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(
            f"Training finished but no .safetensors adapter was found in {adapter_dir}"
        )
    return candidates[-1]


def _run_ai_toolkit(config: dict) -> None:
    config_path = Path(tempfile.mkdtemp(prefix="lora-config-")) / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, f"{AI_TOOLKIT_ROOT}/run.py", str(config_path)],
        cwd=AI_TOOLKIT_ROOT,
        check=False,
        timeout=6 * 60 * 60,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ai-toolkit training failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


@app.function(
    image=light_image,
    volumes={"/vol": volume},
    cpu=2,
    memory=2048,
    timeout=600,
)
def store_dataset(dataset_zip: bytes, slug: str) -> dict:
    return store_dataset_bytes(dataset_zip, slug)


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu=TRAIN_GPU,
    timeout=6 * 60 * 60,
    secrets=[hf_secret],
)
def train_character(
    slug: str,
    trigger: str,
    params: dict | None = None,
    base_model: str = BASE_MODEL,
) -> dict:
    slug = check_slug(slug)
    trigger = check_trigger(trigger)
    dataset_dir = DATASET_ROOT / slug
    if not dataset_dir.is_dir():
        raise RuntimeError(f"No Modal dataset found for slug {slug!r}")
    options = clamp_train_params(params)
    config = training_config(slug, trigger, str(dataset_dir), options, base_model)
    _run_ai_toolkit(config)
    volume.commit()
    adapter_file = find_adapter_file(ADAPTER_ROOT / slug)
    relative = adapter_file.relative_to(VOLUME_ROOT).as_posix()
    return {
        "status": "ready",
        "slug": slug,
        "trigger": trigger,
        "adapter_ref": f"modal://{VOLUME_NAME}/{relative}",
        "adapter_file": adapter_file.name,
        "steps": options["steps"],
        "rank": options["rank"],
        "learning_rate": options["learning_rate"],
        "resolution": options["resolution"],
        "seed": options["seed"],
        "base_model": base_model,
    }


@app.function(
    image=light_image,
    volumes={"/vol": volume},
    cpu=2,
    memory=2048,
    timeout=600,
)
def start_training(
    dataset_zip: bytes,
    slug: str,
    trigger: str,
    params: dict | None = None,
    base_model: str = BASE_MODEL,
) -> dict:
    """Store one dataset, then queue GPU training without blocking on it."""
    stored = store_dataset_bytes(dataset_zip, slug)
    call = train_character.spawn(slug, trigger, params or {}, base_model)
    return {
        "status": "training",
        "slug": stored["slug"],
        "trigger": check_trigger(trigger),
        "train_call_id": call.object_id,
        "adapter_dir": f"modal://{VOLUME_NAME}/adapters/{stored['slug']}/",
        "image_count": stored["image_count"],
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu=GENERATE_GPU,
    timeout=1800,
    secrets=[hf_secret],
)
def generate_character(
    adapter_ref: str,
    prompt: str,
    seed: int = 42,
    steps: int = 30,
    guidance_scale: float = 7.0,
    base_model: str = BASE_MODEL,
    lora_scale: float = 1.0,
    negative_prompt: str = "",
) -> dict:
    """Generate one PNG with a stored LoRA and return it as base64."""
    from diffusers import StableDiffusionXLPipeline
    import torch

    if not adapter_ref.startswith(f"modal://{VOLUME_NAME}/"):
        raise ValueError(f"Unsupported adapter_ref: {adapter_ref!r}")
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise ValueError("A prompt is required for adapter generation")
    adapter_path = VOLUME_ROOT / adapter_ref.split(f"modal://{VOLUME_NAME}/", 1)[1]
    adapter_file = (
        adapter_path
        if adapter_path.is_file()
        else find_adapter_file(adapter_path)
        if adapter_path.is_dir()
        else None
    )
    if adapter_file is None:
        raise RuntimeError(f"Adapter not found: {adapter_ref}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(
        str(adapter_file.parent), weight_name=adapter_file.name
    )
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    scale = max(0.0, min(1.5, float(lora_scale)))
    output = pipe(
        cleaned_prompt,
        negative_prompt=(negative_prompt or "").strip() or None,
        num_inference_steps=max(10, min(75, int(steps))),
        guidance_scale=max(1.0, min(15.0, float(guidance_scale))),
        generator=generator,
        cross_attention_kwargs={"scale": scale},
    )
    buffer = io.BytesIO()
    output.images[0].save(buffer, format="PNG")
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "model": base_model,
        "source": "modal-lora",
        "seed": int(seed),
    }
