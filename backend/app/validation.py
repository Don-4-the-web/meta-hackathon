"""Adapter acceptance gate: prove a fresh LoRA's likeness before marking ready.

A training job finishing is not the same as a usable adapter — small-data
LoRAs can collapse to busts, duplicate figures, or scramble outfit colours.
After training completes, three probe renders (neutral full-body, action,
close-up) are generated through the new adapter and checked locally:

- exactly one dominant figure (no sticker-sheet duplicates),
- full body visible where the probe asks for it,
- the DNA's outfit colours actually present in the figure.

Only a passing adapter becomes ``ready``; anything else is ``needs_review``
with per-probe reasons, so a bad adapter can never silently replace a good
one. Probe inference burns a few cents of Modal GPU per validation run.
"""
import asyncio
import base64
import io

import numpy as np
from PIL import Image

from . import dataset, imagegen, training

PROBES = (
    {"name": "neutral", "seed": 11,
     "action": "standing facing forward, arms at sides",
     "framing": "full body, head to toe", "needs_body": True},
    {"name": "action", "seed": 22,
     "action": "waving hello",
     "framing": "full body, head to toe", "needs_body": True},
    {"name": "portrait", "seed": 33,
     "action": "", "framing": "close-up portrait", "needs_body": False},
)

_COLOR_TOLERANCE = 60.0
_BODY_MIN_HEIGHT = 0.55
_DOMINANT_MIN_SHARE = 0.55


def _hex_rgb(value: object) -> tuple[int, int, int] | None:
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
        except ValueError:
            return None
    return None


def _figure_mask(img: Image.Image):
    """Cleaned subject mask, or None when nothing coherent is found."""
    mask = imagegen._segment_mask(img.convert("RGB"))
    w, h = img.size
    if mask is None or imagegen._mask_bbox(mask, w, h) is None:
        return None
    cleaned = imagegen._fill_mask_holes(
        imagegen._significant_components(mask))
    ok, _ = imagegen._mask_quality(cleaned, w, h)
    return cleaned if ok else None


def _region_shares(mask: Image.Image) -> list[float]:
    """Fraction of kept pixels per connected region, largest first."""
    from collections import deque
    small = mask.resize((160, 160), Image.NEAREST)
    arr = np.asarray(small) > 128
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    shares: list[float] = []
    total = int(arr.sum()) or 1
    for y in range(h):
        for x in range(w):
            if arr[y, x] and not seen[y, x]:
                queue = deque([(y, x)])
                seen[y, x] = True
                size = 0
                while queue:
                    cy, cx = queue.popleft()
                    size += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                                   (cy, cx - 1), (cy, cx + 1)):
                        if (0 <= ny < h and 0 <= nx < w
                                and arr[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                shares.append(size / total)
    return sorted(shares, reverse=True)


def _figure_colors(img: Image.Image, mask: Image.Image) -> list[tuple[int, int, int]]:
    small = img.convert("RGB").resize((48, 48))
    msmall = mask.resize((48, 48)).point(lambda v: 255 if v > 128 else 0)
    px, mp = small.load(), msmall.load()
    colors = []
    for y in range(48):
        for x in range(48):
            if mp[x, y]:
                colors.append(px[x, y])
    return colors


def _missing_colors(img: Image.Image, mask: Image.Image,
                    dna: dict,
                    min_fraction: float = 0.02) -> list[str]:
    """DNA outfit colours with no real presence in the figure.

    Presence means a share of the figure's pixels, not a single pixel:
    highlights and edge fringes can land within tolerance of any colour
    and would otherwise let a bust (or a wrong outfit) pass.
    """
    wanted = {}
    for label, key in (("shirt", "outfit_top"), ("bottoms", "outfit_bottom")):
        rgb = _hex_rgb(dna.get(key))
        if rgb:
            wanted[label] = rgb
    if not wanted:
        return []
    colors = _figure_colors(img, mask)
    if not colors:
        return list(wanted)
    need = max(8, int(min_fraction * len(colors)))
    missing = []
    for label, target in wanted.items():
        tol2 = _COLOR_TOLERANCE ** 2
        hits = sum(1 for r, g, b in colors
                   if (r - target[0]) ** 2 + (g - target[1]) ** 2
                   + (b - target[2]) ** 2 <= tol2)
        if hits < need:
            missing.append(label)
    return missing


def _top_band_colors(img: Image.Image, mask: Image.Image,
                    fraction: float = 0.30) -> list[tuple[int, int, int]]:
    """Saturated figure colours in the top band (headwear territory)."""
    bbox = mask.getbbox()
    if not bbox:
        return []
    w, h = img.size
    box = (0, max(0, bbox[1] - 4),
           w, min(h, bbox[1] + int((bbox[3] - bbox[1]) * fraction)))
    band = img.crop(box)
    mband = mask.crop(box).resize(band.size).point(
        lambda v: 255 if v > 128 else 0)
    bp, mp = band.load(), mband.load()
    colors = []
    for y in range(band.size[1]):
        for x in range(band.size[0]):
            if mp[x, y]:
                r, g, b = bp[x, y]
                if max(r, g, b) - min(r, g, b) > 40:
                    colors.append((r, g, b))
    return colors


def check_probe_image(img: Image.Image, dna: dict,
                      needs_body: bool) -> list[str]:
    """Local likeness checks for one probe render; empty means pass."""
    reasons: list[str] = []
    w, h = img.size
    mask = _figure_mask(img)
    if mask is None:
        return ["no coherent subject isolated"]
    shares = _region_shares(mask)
    if not shares or shares[0] < _DOMINANT_MIN_SHARE:
        reasons.append("multiple figures instead of one character")
    if needs_body:
        bbox = mask.getbbox()
        height = (bbox[3] - bbox[1]) / h if bbox else 0
        if height < _BODY_MIN_HEIGHT:
            reasons.append("full body not visible (bust/close-up)")
    for label in _missing_colors(img, mask, dna or {}):
        reasons.append(f"outfit colour missing: {label}")
    accent = _hex_rgb((dna or {}).get("accent"))
    if accent:
        top = _top_band_colors(img.convert("RGB"), mask)
        if top:
            best = min(
                (r - accent[0]) ** 2 + (g - accent[1]) ** 2
                + (b - accent[2]) ** 2 for r, g, b in top) ** 0.5
            if best > _COLOR_TOLERANCE:
                reasons.append("headwear colour wrong or missing")
        else:
            reasons.append("headwear colour wrong or missing")
    return reasons


def _probe_prompt(trigger: str, asset: dict, probe: dict) -> str:
    core = (f"{asset.get('name', 'the figure')}: "
            f"{asset.get('description', '')}").strip()
    action = probe["action"].strip()
    action_bit = f" {action.rstrip('.')}. " if action else " "
    return (f"{trigger}, {core}{action_bit}calm expression, "
            f"{probe['framing']}, single character only, no duplicates, "
            f"t-shirt graphic, isolated on a plain white background, "
            f"everything fully inside the frame.")


async def validate_adapter(asset: dict, adapter_ref: str,
                           lora_scale: float = 0.85) -> dict:
    """Render the probe set through a fresh adapter and grade it.

    Never raises for a bad adapter: failures come back as per-probe
    reasons and ``passed: False``. Transport errors raise so callers can
    surface them distinctly from a failed likeness check.
    """
    trigger = dataset.namespaced_trigger(asset)
    probes = []
    for probe in PROBES:
        entry: dict = {"name": probe["name"], "passed": False, "reasons": []}
        try:
            generated = await asyncio.to_thread(
                training.generate_with_adapter, adapter_ref,
                _probe_prompt(trigger, asset, probe), probe["seed"],
                25, 7.0, lora_scale, imagegen.ADAPTER_NEGATIVE)
            raw = base64.b64decode(
                generated["image"].split(",", 1)[1], validate=True)
            img = Image.open(io.BytesIO(raw))
            img.load()
        except (training.ModalTrainingError,
                training.TrainingNotConfigured):
            raise
        except Exception as e:
            entry["reasons"] = [f"probe render failed: {e}"]
            probes.append(entry)
            continue
        entry["reasons"] = check_probe_image(
            img.convert("RGB"), asset.get("dna") or {},
            probe["needs_body"])
        entry["passed"] = not entry["reasons"]
        probes.append(entry)
    passed = all(p["passed"] for p in probes)
    return {"passed": passed, "adapter_ref": adapter_ref,
            "trigger": trigger, "probes": probes}
