"""Modal training client used by the FastAPI backend.

This module only contacts Modal when it is called: importing it never spends
credits. Deployment/authentication remain manual::

    modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface-secret HF_TOKEN=<huggingface-read-token>
    cd backend
    modal deploy modal_lora_app.py

The deployed app's ``start_training`` function stores one dataset and queues
GPU training. Its returned ``train_call_id`` is polled later with a zero
timeout, exactly as Modal documents for background execution.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from . import dataset
from .config import settings


class ModalNotConfigured(Exception):
    pass


class ModalTrainingError(Exception):
    pass


def _setting(name: str, default: str = "") -> str:
    from_env = os.environ.get(name, "").strip()
    if from_env:
        return from_env
    return str(getattr(settings, name.lower(), default) or default).strip()


def _section_has_credentials(section: object) -> bool:
    if not isinstance(section, dict):
        return False
    token = bool(section.get("token_id") and section.get("token_secret"))
    oauth = bool(
        section.get("oauth_refresh_token")
        and section.get("oauth_client_id")
        and (section.get("oauth_client_secret") or section.get("oauth_jwt_key"))
    )
    return token or oauth


def _toml_credentials_present(path: Path, profile: str) -> bool:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    # `modal setup` names the section after the Modal username (e.g. the
    # `[don-martirez]` section), so accept any profile section holding
    # credentials, preferring MODAL_PROFILE when it exists.
    sections = [data.get(profile)] + [
        section for name, section in data.items() if name != profile
    ]
    return any(_section_has_credentials(section) for section in sections)


def modal_credentials_present() -> bool:
    """Detect Modal auth without contacting Modal or spending credits."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    if (
        os.environ.get("MODAL_OAUTH_REFRESH_TOKEN")
        and os.environ.get("MODAL_OAUTH_CLIENT_ID")
        and (
            os.environ.get("MODAL_OAUTH_CLIENT_SECRET")
            or os.environ.get("MODAL_OAUTH_JWT_KEY")
        )
    ):
        return True
    configured_path = os.environ.get("MODAL_CONFIG_PATH")
    if configured_path:
        config_path = Path(configured_path)
    else:
        try:
            config_path = Path.home() / ".modal.toml"
        except RuntimeError:
            return False
    profile = os.environ.get("MODAL_PROFILE", "default")
    return _toml_credentials_present(config_path, profile)


def modal_settings() -> dict:
    return {
        "app": _setting("MODAL_APP_NAME", "meta-lora-studio"),
        "starter": _setting("MODAL_STARTER_FUNCTION", "start_training"),
        "generate": _setting("MODAL_GENERATE_FUNCTION", "generate_character"),
        "volume": _setting("MODAL_VOLUME_NAME", "lora-studio"),
        "base_model": _setting(
            "MODAL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        "submit_timeout": float(_setting("MODAL_SUBMIT_TIMEOUT_SECONDS", "180") or 180),
        "max_steps": int(_setting("MODAL_MAX_TRAIN_STEPS", "400") or 400),
    }


def modal_configured() -> bool:
    config = modal_settings()
    if _setting("TRAINING_PROVIDER", "modal").lower() != "modal":
        return False
    if not (config["app"] and config["starter"] and config["generate"]):
        return False
    return modal_credentials_present()


def _require_modal():
    try:
        import modal
    except ImportError as exc:
        raise ModalNotConfigured(
            "The Modal SDK is not installed in backend/.venv "
            "(`pip install -r requirements.txt`)."
        ) from exc
    return modal


def _train_options(options: dict | None) -> dict:
    config = modal_settings()
    options = dict(options or {})
    steps = int(options.get("steps", config["max_steps"]))
    rank = int(options.get("rank", 16))
    learning_rate = float(options.get("learning_rate", 1e-4))
    resolution = int(options.get("resolution", 1024))
    return {
        "steps": max(100, min(config["max_steps"], steps)),
        "rank": max(4, min(64, rank)),
        "learning_rate": max(1e-6, min(1e-3, learning_rate)),
        "resolution": 1024 if resolution < 768 else min(1536, resolution),
        "seed": int(options.get("seed", 42)),
        "validation_prompt": str(options.get("validation_prompt") or "")[:500],
    }


def _lookup(modal, function_name: str):
    config = modal_settings()
    try:
        return modal.Function.from_name(config["app"], function_name)
    except modal.exception.NotFoundError as exc:
        raise ModalTrainingError(
            f"Modal app {config['app']!r} / function {function_name!r} was not "
            "found. Deploy it first (`modal deploy modal_lora_app.py`) and "
            "check MODAL_APP_NAME."
        ) from exc


def submit_training(asset: dict, zip_bytes: bytes, options: dict | None = None) -> dict:
    """Upload one dataset to Modal and queue GPU LoRA training.

    The remote starter returns after the dataset is stored and training is
    queued; the long GPU run continues under ``train_call_id``. No GPU result
    is awaited here, so submitting does not block on training.
    """
    if not modal_configured():
        raise ModalNotConfigured(
            "Modal is not connected. Set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET "
            "(or run `modal setup`), deploy backend/modal_lora_app.py, and retry. "
            "Nothing has been spent."
        )
    if asset.get("kind") != "character":
        raise ValueError("Only character assets can be trained")
    if not zip_bytes:
        raise ValueError("A dataset zip is required for Modal training")

    modal = _require_modal()
    config = modal_settings()
    trigger = dataset.namespaced_trigger(asset)
    slug = trigger.removesuffix("_tok")
    params = _train_options(options)
    starter = _lookup(modal, config["starter"])
    try:
        call = starter.spawn(
            zip_bytes, slug, trigger, params, config["base_model"]
        )
        queued = call.get(timeout=config["submit_timeout"])
    except modal.exception.NotFoundError as exc:
        raise ModalTrainingError(
            "The deployed Modal starter disappeared. Redeploy "
            "backend/modal_lora_app.py before retrying."
        ) from exc
    except modal.exception.AuthError as exc:
        raise ModalTrainingError(
            "Modal rejected these credentials. Run `modal token set` again "
            "and retry; no training was queued."
        ) from exc
    except modal.exception.RequestSizeError as exc:
        raise ModalTrainingError(
            "The dataset zip exceeded Modal's request limits. Add fewer or "
            "smaller reference images and retry."
        ) from exc
    except TimeoutError as exc:
        raise ModalTrainingError(
            "Modal did not acknowledge dataset upload before the local "
            "timeout. Check training status before retrying, because the "
            "upload may still have been queued."
        ) from exc
    except ModalTrainingError:
        raise
    except Exception as exc:
        raise ModalTrainingError(f"Modal training submission failed: {exc}") from exc

    if not isinstance(queued, dict) or not queued.get("train_call_id"):
        raise ModalTrainingError(
            "Modal accepted the dataset but did not return a training call id."
        )
    return {
        "status": "training",
        "adapter_ref": str(
            queued.get("adapter_dir") or f"modal://{config['volume']}/adapters/{slug}/"
        ),
        "call_id": str(queued["train_call_id"]),
        "slug": slug,
        "trigger": trigger,
        "image_count": int(queued.get("image_count", 0)),
        "steps": params["steps"],
    }


def training_status(call_id: str) -> dict:
    """Poll one Modal training call without blocking on GPU work."""
    if not call_id:
        raise ValueError("A Modal training call id is required for status")
    if not modal_configured():
        raise ModalNotConfigured(
            "Modal credentials are missing, so training status cannot be polled."
        )
    modal = _require_modal()
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return {"status": "training", "call_id": call_id}
    except modal.exception.OutputExpiredError as exc:
        raise ModalTrainingError(
            "Modal finished this call so long ago that its result expired. "
            "Check the Modal dashboard/logs for the adapter."
        ) from exc
    except Exception as exc:
        raise ModalTrainingError(f"Modal training failed: {exc}") from exc
    if not isinstance(result, dict) or result.get("status") != "ready":
        raise ModalTrainingError(
            "Modal training returned an unexpected result shape."
        )
    return {
        "status": "ready",
        "call_id": call_id,
        "adapter_ref": str(result.get("adapter_ref", "")),
        "trigger": str(result.get("trigger", "")),
        "steps": int(result.get("steps", 0)),
    }


def generate_with_adapter(
    adapter_ref: str,
    prompt: str,
    seed: int = 42,
    steps: int = 30,
    guidance_scale: float = 7.0,
    lora_scale: float = 1.0,
    negative_prompt: str = "",
) -> dict:
    """Generate one PNG through a stored Modal LoRA adapter."""
    if not adapter_ref.startswith("modal://"):
        raise ValueError(f"Unsupported adapter_ref: {adapter_ref!r}")
    if not (prompt or "").strip():
        raise ValueError("A prompt is required for adapter generation")
    if not modal_configured():
        raise ModalNotConfigured(
            "Modal credentials are missing, so the stored adapter cannot be used."
        )
    modal = _require_modal()
    config = modal_settings()
    generate = _lookup(modal, config["generate"])
    try:
        result = generate.remote(
            adapter_ref,
            prompt.strip(),
            int(seed),
            int(steps),
            float(guidance_scale),
            config["base_model"],
            float(lora_scale),
            (negative_prompt or "").strip(),
        )
    except modal.exception.NotFoundError as exc:
        raise ModalTrainingError(
            "The deployed Modal generation function was not found. Redeploy "
            "backend/modal_lora_app.py before retrying."
        ) from exc
    except modal.exception.AuthError as exc:
        raise ModalTrainingError(
            "Modal rejected these credentials. Run `modal token set` again "
            "and retry; no image was generated."
        ) from exc
    except Exception as exc:
        raise ModalTrainingError(f"Modal adapter generation failed: {exc}") from exc
    if not isinstance(result, dict) or not result.get("image_base64"):
        raise ModalTrainingError("Modal returned an empty adapter image.")
    return {
        "image": f"data:image/png;base64,{result['image_base64']}",
        "model": str(result.get("model") or config["base_model"]),
        "source": "modal-lora",
    }
