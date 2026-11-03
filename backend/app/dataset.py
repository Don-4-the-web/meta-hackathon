"""Training dataset preparation for per-character LoRA fine-tunes.

Turns an Asset Bible entry's uploaded reference images into a trainer-
ready set: one image per file, each captioned with the character's
stable trigger word plus the encoded traits (the same DNA the Meta
vision encoder produced). A trainer consumes the zip from
`training_zip` (000.png / 000.txt pairs).
"""
import base64
import hashlib
import io
import re
import zipfile

from PIL import Image, ImageStat

_VIEWS = ("standing, full body", "close-up portrait", "side view, full body",
          "three-quarter view, full body", "smiling, full body",
          "walking, full body")


def trigger_for(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return f"{slug or 'character'}_tok"


def namespace_for(asset: dict) -> str:
    """Per-user Modal namespace so two users' characters never collide."""
    uid = asset.get("user_id") or ""
    return f"u{uid.replace('-', '')[:8]}" if uid else ""


def namespaced_trigger(asset: dict) -> str:
    """Trigger word actually used in captions, config, and generation."""
    trigger = trigger_for(asset.get("name", ""))
    namespace = namespace_for(asset)
    return f"{namespace}_{trigger}" if namespace else trigger


def image_fingerprint(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()[:16]


def asset_images(asset: dict) -> list[str]:
    images = [i for i in (asset.get("reference_images") or [])
              if isinstance(i, str) and i.startswith("data:image/")]
    first = asset.get("reference_image") or ""
    if first.startswith("data:image/") and first not in images:
        images.insert(0, first)
    return images


def _caption(asset: dict, trigger: str, index: int) -> str:
    d = asset.get("dna", {}) or {}
    bits = [f"a reference image of {trigger}"]
    if asset.get("description"):
        bits.append(str(asset["description"]).rstrip(".").lower())
    if d.get("accessory") not in (None, "", "none"):
        bits.append(f"wearing a {d['accessory']}")
    if d.get("facial_hair") not in (None, "", "none"):
        bits.append(f"with a {d['facial_hair']}")
    bits.append(_VIEWS[index % len(_VIEWS)])
    return ", ".join(bits)


def build_dataset(asset: dict) -> dict:
    images = asset_images(asset)
    if asset.get("kind") != "character":
        raise ValueError("Only character assets can be trained")
    if not images:
        raise ValueError("Upload at least one reference image first")
    trigger = namespaced_trigger(asset)
    overrides = asset.get("caption_overrides") or {}
    items = []
    for i, img in enumerate(images):
        try:
            png = _png_bytes(img)
        except Exception:
            # Decodability is enforced with a clear error at zip time;
            # previews still work with a content hash as the fingerprint.
            png = b""
        fingerprint = (image_fingerprint(png) if png
                       else hashlib.sha256(img.encode()).hexdigest()[:16])
        caption = overrides.get(fingerprint) or _caption(asset, trigger, i)
        items.append({"index": i, "caption": caption, "image": img,
                      "fingerprint": fingerprint,
                      "custom": fingerprint in overrides})
    return {"trigger": trigger, "count": len(items), "items": items,
            "recommended_minimum": 10, "ready": len(items) >= 1}


def _band_mean(png_bytes: bytes, box: tuple) -> tuple:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    region = img.crop((int(w * box[0]), int(h * box[1]),
                       int(w * box[2]), int(h * box[3])))
    return tuple(round(v) for v in ImageStat.Stat(
        region.resize((24, 24))).mean)


def grade_dataset(asset: dict) -> dict:
    """Free, local pre-flight check. No GPU, no spend, no upload.

    Returns blocking errors plus warnings. Anything error-level means
    training will waste credits and should be fixed first.
    """
    images = asset_images(asset)
    if asset.get("kind") != "character":
        raise ValueError("Only character assets can be trained")
    checks: list[dict] = []
    pngs: list[bytes] = []
    # Duplicates are checked against the raw upload list (asset_images
    # dedupes exact matches, which would hide them from this report).
    first = asset.get("reference_image") or ""
    raw = (([first] if first.startswith("data:image/") else [])
           + [i for i in (asset.get("reference_images") or [])
              if isinstance(i, str) and i.startswith("data:image/")])
    raw_pngs: list[bytes] = []
    for img in raw:
        try:
            raw_pngs.append(_png_bytes(img))
        except Exception:
            pass
    seen_raw: dict[str, int] = {}
    for i, png in enumerate(raw_pngs):
        fingerprint = image_fingerprint(png)
        if fingerprint in seen_raw:
            checks.append({"level": "warning", "code": "duplicate",
                           "message": f"Image {i + 1} is an exact duplicate of "
                                      f"image {seen_raw[fingerprint] + 1}. Duplicates "
                                      "teach the LoRA nothing new."})
        else:
            seen_raw[fingerprint] = i
    for i, img in enumerate(images):
        try:
            pngs.append(_png_bytes(img))
        except Exception:
            checks.append({"level": "error", "code": "unreadable",
                           "message": f"Image {i + 1} could not be decoded. "
                                      "Re-upload it."})
    seen: dict[str, int] = {}
    for png in pngs:
        seen.setdefault(image_fingerprint(png), len(seen))
    for i, png in enumerate(pngs):
        with Image.open(io.BytesIO(png)) as probe:
            if min(probe.size) < 256:
                checks.append({"level": "error", "code": "too_small",
                               "message": f"Image {i + 1} is only "
                                          f"{probe.size[0]}x{probe.size[1]}px. "
                                          "Use at least 256px on the short side."})
    blank = [i + 1 for i, png in enumerate(pngs)
             if len(Image.open(io.BytesIO(png)).convert("RGB").getcolors(1 << 20) or []) == 1]
    for i in blank:
        checks.append({"level": "error", "code": "blank",
                       "message": f"Image {i} is a single flat colour. "
                                  "It teaches nothing — replace it."})
    distinct = len(seen)
    if distinct < 3:
        checks.append({"level": "error", "code": "too_few",
                       "message": f"Only {distinct} distinct image(s). "
                                  "Use at least 3 very different poses."})
    elif distinct < 10:
        checks.append({"level": "warning", "code": "below_minimum",
                       "message": f"Only {distinct} distinct image(s); "
                                  "10+ varied poses train far better."})
    # Wardrobe consistency: compare torso-band means across images. A wide
    # spread means the captions' single outfit claim is probably wrong for
    # some images — exactly what scrambles colours in the LoRA.
    means = [_band_mean(png, (0.30, 0.42, 0.70, 0.75)) for png in pngs]
    spread = max(
        (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
         for a in means for b in means),
        default=0,
    ) if means else 0
    if spread > 120:
        checks.append({"level": "warning", "code": "wardrobe_spread",
                       "message": "Outfit colours vary a lot between images, "
                                  "but every caption claims one outfit. Edit "
                                  "the captions (or drop the odd ones out) or "
                                  "colours will scramble."})
    errors = [c for c in checks if c["level"] == "error"]
    return {"ready": not errors, "count": len(images), "distinct": distinct,
            "recommended_minimum": 10, "trigger": namespaced_trigger(asset),
            "checks": checks}


def _png_bytes(data_url: str) -> bytes:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def training_zip(asset: dict) -> bytes:
    ds = build_dataset(asset)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in ds["items"]:
            stem = f"{item['index']:03d}"
            z.writestr(f"{stem}.png", _png_bytes(item["image"]))
            z.writestr(f"{stem}.txt", item["caption"])
    return buf.getvalue()
