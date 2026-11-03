"""True-likeness graphic generation via the Meta image model.

The graphic generator does NOT draw shapes: it calls muse-image-1.0.
When the asset carries a reference image, generation is conditioned on
that image (/images/edits) so the output reproduces the actual figure;
without one it generates from the encoded DNA description
(/images/generations) and the result is labelled accordingly.

Image output is non-deterministic by nature: output_hash identifies the
returned image, it does not promise a repeat generation is identical.
Upstream failures raise LLMError — there is no silent shape fallback.
"""
import asyncio
import base64
import binascii
import io

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from . import dataset, training
from .config import settings
from .encode import encode_reference
from .llm import LLMError
from .pipeline import canonical_hash, freeze_bible

PROFILES = {
    "tshirt": (
        " For a t-shirt graphic: a single character only, no duplicates, "
        "no sticker sheet, no extra figures; the design isolated on a "
        "plain solid white background, everything fully inside the frame "
        "with generous empty margin on every side, nothing cropped or "
        "touching the edges, no backdrop scene beyond the design itself, "
        "no text, no letters."),
    "storyboard": (
        " Full-bleed cinematic storyboard panel filling the whole frame, "
        "environment fully visible, continuity with neighbouring panels."),
    "video": (
        " 16:9 widescreen short-video keyframe filling the frame, leave "
        "headroom above and around the subject for camera motion, subject "
        "complete and uncropped, no text."),
}

_POSE = {"stand": "standing", "walk": "walking", "reach": "reaching out",
         "leap": "leaping"}
# What the LoRA must never draw: the collapse modes a small-data adapter
# falls into (busts instead of bodies, cropped figures, sticker sheets).
ADAPTER_NEGATIVE = ("close-up, bust, head only portrait, cropped, cut off, "
                    "multiple characters, duplicates, sticker sheet, "
                    "watermark, text, blurry, deformed")
_SHOT = {"wide": "wide full-body shot", "medium": "medium full-body shot",
         "close": "close-up portrait"}


def build_likeness_prompt(asset: dict, extras: list[dict], pose: str,
                          expression: str, shot: str, action: str = "") -> str:
    d = asset.get("dna", {})
    traits = []
    if d.get("accessory") not in (None, "", "none"):
        traits.append(f"wearing a {d['accessory']}")
    if d.get("facial_hair") not in (None, "", "none"):
        traits.append(f"with a {d['facial_hair']}")
    for label, key in (("hair", "hair"), ("top/shirt", "outfit_top"),
                       ("trousers/bottoms", "outfit_bottom")):
        if str(d.get(key, "")).startswith("#"):
            traits.append(f"{label} colour {d[key]}")
    scene = ""
    if extras:
        names = ", ".join(f"{e['name']} ({e.get('description', '')})"
                          for e in extras)
        scene = f" Scene includes: {names}."
    desc = asset.get("description", "")
    # A generic pose word ("standing pose") next to a specific action makes
    # the model pick the pose and drop the action — props and all. When the
    # user describes an action it becomes the bossy sentence and the stock
    # pose label is dropped.
    pose_bit = "" if action else f"{_POSE.get(pose, 'standing')} pose, "
    doing = (f" Action (depict this clearly, with any named objects): "
             f"{action}.") if action else ""
    return (f"{_SHOT.get(shot, 'medium full-body shot')} of "
            f"{asset.get('name', 'the figure')}: {desc} "
            f"{'; '.join(traits)}.{doing} {pose_bit}"
            f"{expression} expression.{scene}")


def _reference_png(data_url: str) -> bytes:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, IndexError, Exception) as e:
        raise LLMError(f"Reference image could not be decoded: {e}") from e
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode_image(data_url: str) -> Image.Image:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except (binascii.Error, IndexError, Exception) as e:
        raise LLMError(f"Generated image could not be decoded: {e}") from e


_SEGMENT_SESSION = None


def _segment_mask(img: Image.Image):
    """Return an L-mode subject mask via learned segmentation, or None.

    Returns None when `rembg` (or its model) is unavailable so callers can
    fall back to flood-fill. Never raises: isolation must degrade, not
    fail, when the segmentation backend is missing.
    """
    global _SEGMENT_SESSION
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    try:
        if _SEGMENT_SESSION is None:
            _SEGMENT_SESSION = new_session("u2net")
        mask = remove(img, session=_SEGMENT_SESSION, only_mask=True)
    except Exception:
        return None
    return mask.convert("L").point(lambda v: 255 if v > 128 else 0)


def _mask_bbox(mask, w: int, h: int):
    """Bounding box of a mask, or None when the mask fails open/closed."""
    bbox = mask.getbbox()
    if not bbox:
        return None
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area > 0.98 * w * h:
        # Mask covers the whole frame: segmentation failed open, which
        # would leave a full-bleed background. Treat as unusable.
        return None
    return bbox


def _significant_components(mask: Image.Image) -> Image.Image:
    """Drop segmentation speckles but keep every real design part.

    A t-shirt design is often several disconnected pieces (figure, held
    tool, dirt pile, decorative leaves) — keeping only the single largest
    region amputates them, which reads as fragmentation. So this keeps
    every region bigger than a speckle (half a percent of the frame, or 2%
    of the biggest region) and drops only the small fry. Labelling runs on
    a downsampled copy for speed; the result is ANDed with the
    full-resolution mask so surviving edges keep their original fidelity.
    A light closing first reconnects thin breaks (a shovel handle with a
    one-pixel gap) before anything is measured.
    """
    import numpy as np
    from collections import deque
    # 384, not 160: at 160px a 6x downsample erases thin structures (tool
    # handles, stems) from the label map, and the AND below then slices
    # them off the artwork — clean cuts that read as fragmentation.
    closed = mask.filter(ImageFilter.MaxFilter(3)).filter(
        ImageFilter.MinFilter(3))
    small = closed.resize((384, 384), Image.NEAREST)
    arr = np.asarray(small) > 128
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list] = []
    for y in range(h):
        for x in range(w):
            if arr[y, x] and not seen[y, x]:
                queue = deque([(y, x)])
                seen[y, x] = True
                cells = [(y, x)]
                while queue:
                    cy, cx = queue.popleft()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                                   (cy, cx - 1), (cy, cx + 1)):
                        if (0 <= ny < h and 0 <= nx < w
                                and arr[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                            cells.append((ny, nx))
                regions.append(cells)
    if not regions:
        return Image.new("L", mask.size, 0)
    biggest = max(len(c) for c in regions)
    keep_min = max(0.005 * w * h, 0.02 * biggest)
    # Area alone would kill thin design parts (a 500px handle is a few
    # hundred cells): also keep anything long enough to be deliberate.
    keep_span = 0.08 * w
    region = Image.new("L", (w, h), 0)
    px = region.load()
    for cells in regions:
        if len(cells) >= keep_min:
            keep = True
        else:
            xs = [c[1] for c in cells]
            ys = [c[0] for c in cells]
            keep = max(max(xs) - min(xs), max(ys) - min(ys)) >= keep_span
        if keep:
            for cy, cx in cells:
                px[cx, cy] = 255
    # Bilinear (not nearest) so kept regions' edges stay smooth instead of
    # snapping to the labelling grid — that snap reads as jaggies
    # on the finished cutout.
    grown = region.resize(mask.size, Image.BILINEAR)
    grown = grown.point(lambda v: 255 if v > 128 else 0)
    # Tolerate a few pixels around each kept region before gating: thin
    # structures can still be a pixel off between the label map and the
    # full-resolution mask, and without this slack the AND slices them.
    # Distant speckles are far outside the slack, so they still drop out.
    grown = grown.filter(ImageFilter.MaxFilter(7))
    gated = ImageChops.logical_and(mask.convert("1"), grown.convert("1"))
    return gated.convert("L")


def _fill_mask_holes(mask: Image.Image) -> Image.Image:
    """Fill background-coloured holes enclosed inside a binary mask.

    Segmentation often punches holes (an eye, a gap between arm and body)
    that would show the t-shirt through the artwork. Anything not
    connected to the frame border becomes subject.
    """
    if mask.getpixel((0, 0)) != 0:
        return mask
    filled = mask.copy()
    ImageDraw.floodfill(filled, (0, 0), 128)
    return filled.point(lambda v: 0 if v == 128 else 255)


def _mask_quality(mask, w: int, h: int) -> tuple[bool, float]:
    """Coherence check: is this mask one solid subject, not fragments?

    Returns (ok, removed_fraction). A sliver, a mostly-empty frame, or a
    holey patch (low fill inside its own bounding box) means the segmenter
    only caught part of the figure — using it would print a fragmented,
    incomplete character.
    """
    bbox = mask.getbbox()
    if not bbox:
        return False, 0.0
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    white = mask.histogram()[255]
    if not area:
        return False, 0.0
    coverage = white / (w * h)
    fill = white / area
    ok = 0.03 <= coverage <= 0.98 and fill >= 0.35
    return ok, 1.0 - coverage


def _floodfill_mask(img: Image.Image, w: int, h: int) -> Image.Image:
    """Background key via corner flood-fill (works on plain backdrops)."""
    work = img.copy()
    sentinel = (255, 0, 255)
    for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        ImageDraw.floodfill(work, corner, sentinel, thresh=28)
    key = Image.new("RGB", work.size, sentinel)
    return ImageChops.difference(work, key).convert("L").point(
        lambda v: 0 if v == 0 else 255)


def isolate_subject(data_url: str) -> tuple[str, list[str]]:
    """Cut the subject off its background for t-shirt use.

    Prefers learned segmentation (clean alpha on any background), falling
    back to corner flood-fill on plain backgrounds when segmentation is
    unavailable or fails open. The subject is then re-centred at 72% of a
    transparent square canvas so containment is guaranteed, not hoped
    for. Returns (transparent PNG data URL, failed check labels).
    """
    img = _decode_image(data_url).convert("RGB")
    w, h = img.size
    solid = Image.new("RGB", img.size, img.getpixel((0, 0)))
    if ImageChops.difference(img, solid).getbbox() is None:
        raise LLMError("Could not isolate a subject from the generated image")
    uncertain = False
    alpha = None
    mask = _segment_mask(img)
    if mask is not None and _mask_bbox(mask, w, h):
        # Segmentation masks are often holey or catch only part of an
        # illustrated figure; using one raw punches holes in the artwork.
        cleaned = _fill_mask_holes(_significant_components(mask))
        ok, _ = _mask_quality(cleaned, w, h)
        if ok:
            alpha = cleaned
    if alpha is None:
        alpha = _floodfill_mask(img, w, h)
        ok, removed = _mask_quality(alpha, w, h)
        # Flood-fill on a textured backdrop only nibbles the corners, so
        # the "subject" would still be the whole scene. Say so instead of
        # silently shipping a full-bleed cutout as a clean isolation.
        uncertain = not ok or removed < 0.10
    bbox = alpha.getbbox()
    if not bbox:
        raise LLMError("Could not isolate a subject from the generated image")
    # Feather the cut edge: a hard-thresholded mask stair-steps on every
    # diagonal outline (the visible "jaggies"). A one-pixel soft falloff
    # reads as a clean print edge instead.
    alpha = alpha.filter(ImageFilter.GaussianBlur(1))
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    subject = rgba.crop(bbox)
    canvas = 1024
    subject.thumbnail((int(canvas * 0.72), int(canvas * 0.72)),
                      Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.alpha_composite(subject,
                        ((canvas - subject.width) // 2,
                         (canvas - subject.height) // 2))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    final = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    checks = _tshirt_checks(out)
    if uncertain and "subject isolation is uncertain" not in checks:
        checks.append("subject isolation is uncertain")
    return final, checks


def _tshirt_checks(img: Image.Image) -> list[str]:
    checks = []
    alpha = img.getchannel("A")
    w, h = img.size
    if any(alpha.getpixel(p) != 0
           for p in ((2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3))):
        checks.append("background is not blank in every corner")
    bbox = alpha.getbbox()
    if not bbox:
        checks.append("no subject found")
    else:
        if (bbox[0] < w * 0.10 or bbox[1] < h * 0.10
                or bbox[2] > w * 0.90 or bbox[3] > h * 0.90):
            checks.append("subject is not contained within a safe margin")
    return checks


_COMPOSITIONS = {
    "figure": "one isolated character only, bold and clean",
    "vignette": ("a dense, richly detailed t-shirt design: the character in "
                 "dynamic action above a detailed ground vignette, "
                 "decorative leaf sprigs framing the sides, motion and "
                 "texture throughout, thick outlines and a bold limited "
                 "colour palette like a screen-printed t-shirt graphic — "
                 "graphic and full, never bare"),
    "badge": ("a dense circular badge-framed t-shirt design: the character "
              "centred inside a bold circular border, decorative leaves "
              "and stars filling the ring, thick outlines and a bold "
              "limited colour palette like a screen-printed emblem"),
}
_FONT_CANDIDATES = ("C:/Windows/Fonts/comicbd.ttf",
                    "C:/Windows/Fonts/arialbd.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_INK = (51, 36, 26)  # warm dark brown, matches storybook outlines


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _sample_accent(img: Image.Image) -> tuple[int, int, int]:
    """Most saturated colour in the artwork, for phrase lettering."""
    small = img.convert("RGBA").resize((64, 64))
    best, best_score = (227, 168, 43), -1.0
    for r, g, b, a in small.getdata():
        if a < 200:
            continue
        hi, lo = max(r, g, b), min(r, g, b)
        score = (hi - lo) * (1 - abs(hi - 150) / 255)
        if hi > 90 and score > best_score:
            best, best_score = (r, g, b), score
    return best


def _styled_text_layer(text: str, font, fill) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    box = probe.textbbox((0, 0), text, font=font)
    stroke = max(2, font.size // 12)
    pad = stroke + 8
    layer = Image.new("RGBA",
                      (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad),
                      (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    xy = (pad - box[0], pad - box[1])
    d.text((xy[0] + max(2, font.size // 18), xy[1] + max(2, font.size // 18)),
           text, font=font, fill=(*_INK, 110))
    d.text(xy, text, font=font, fill=(*fill, 255),
           stroke_width=stroke, stroke_fill=(*_INK, 255))
    return layer


def _arc_warp(layer: Image.Image, bend: int) -> Image.Image:
    """Arch the lettering: edges drop, centre rises (classic tee type)."""
    if bend <= 0:
        return layer
    out = Image.new("RGBA", (layer.width, layer.height + bend), (0, 0, 0, 0))
    for x in range(layer.width):
        u = (2 * (x + 0.5) / layer.width) - 1
        out.alpha_composite(layer.crop((x, 0, x + 1, layer.height)),
                            (x, int(bend * u * u)))
    return out


def compose_phrase(data_url: str, phrase: str,
                   placement: str = "below") -> tuple[str, list[str]]:
    """Render the exact phrase locally (image models misspell text).

    The artwork stays text-free; the words the user typed are drawn with
    a real font onto the transparent canvas: below, above, or on a dark
    banner across the lower design. Returns (PNG data URL, checks).
    """
    img = _decode_image(data_url).convert("RGBA")
    text = (phrase or "").strip()
    if not text:
        return data_url, _tshirt_checks(img)
    if placement not in ("above", "below", "banner"):
        placement = "below"
    size = max(24, img.width // 11)
    while size > 10:
        font = _load_font(size)
        probe = ImageDraw.Draw(img)
        box = probe.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= img.width * 0.86:
            break
        size = int(size * 0.88)
    font = _load_font(size)
    if placement == "banner":
        out = img
        layer = _styled_text_layer(text, font, (255, 248, 235))
        bx = (out.width - layer.width) // 2
        by = int(out.height * 0.86)
        d = ImageDraw.Draw(out)
        d.rounded_rectangle((bx - 18, by - 10, bx + layer.width + 18,
                             by + layer.height + 12),
                            radius=18, fill=(*_INK, 255))
        out.alpha_composite(layer, (bx, by))
    else:
        layer = _arc_warp(_styled_text_layer(text, font, _sample_accent(img)),
                          int(font.size * 0.45))
        band = layer.height + 48
        out = Image.new("RGBA", (img.width, img.height + band), (0, 0, 0, 0))
        out.alpha_composite(img, (0, band if placement == "above" else 0))
        tx = (out.width - layer.width) // 2
        ty = 16 if placement == "above" else img.height + 16
        out.alpha_composite(layer, (tx, ty))
    pad = int(out.width * 0.12)
    padded = Image.new("RGBA", (out.width + 2 * pad, out.height + 2 * pad),
                       (0, 0, 0, 0))
    padded.alpha_composite(out, (pad, pad))
    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    final = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return final, _tshirt_checks(padded)


def _crop_widescreen(data_url: str) -> str:
    img = _decode_image(data_url)
    w, h = img.size
    target_h = int(w * 9 / 16)
    if target_h >= h:
        return data_url
    top = (h - target_h) // 2
    cropped = img.crop((0, top, w, top + target_h))
    buf = io.BytesIO()
    cropped.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _image_data_url(b64: str) -> str:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, Exception) as e:
        raise LLMError(f"Image model returned undecodable image data: {e}") from e
    if raw.startswith(b"RIFF"):
        mime = "image/webp"
    elif raw.startswith(b"\x89PNG"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


async def _post_image(payload_kind: str, prompt: str,
                      reference_png: bytes | None) -> str:
    """POST one image request, retrying transient (5xx / network) failures.

    Meta occasionally returns a bare 500 `server_error`; failing the whole
    graphic on one flaky call wastes the user's time, so 5xx and connection
    errors get three attempts with backoff. Anything else (bad key, bad
    request) raises immediately.
    """
    model = settings.meta_image_model
    base = settings.meta_api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.meta_api_key}"}
    last: LLMError | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                if payload_kind == "edit":
                    resp = await client.post(
                        base + "/images/edits", headers=headers,
                        files={"image": ("reference.png", reference_png,
                                         "image/png")},
                        data={"model": model, "prompt": prompt})
                else:
                    resp = await client.post(
                        base + "/images/generations",
                        headers={**headers, "Content-Type": "application/json"},
                        json={"model": model, "prompt": prompt})
            if resp.status_code == 200:
                try:
                    b64 = resp.json()["data"][0]["b64_json"]
                except Exception as e:
                    raise LLMError(
                        f"META image API returned an unexpected shape: {e}") from e
                if not isinstance(b64, str) or not b64.strip():
                    raise LLMError("META image API returned an empty image")
                return _image_data_url(b64)
            body = (resp.text or "")[:300]
            last = LLMError(f"META image API returned {resp.status_code}: {body}")
            if resp.status_code < 500:
                raise last
        except LLMError:
            raise
        except Exception as e:
            last = LLMError(
                f"Could not reach META image API: {type(e).__name__}: {e}")
        await asyncio.sleep(2 ** attempt)
    raise last  # type: ignore[misc]


async def _generate_adapter_checked(asset: dict, adapter_ref: str,
                                    prompt: str, seed: int,
                                    lora_scale: float, profile: str,
                                    shot: str) -> tuple[str, str, str, int,
                                                        list[str]]:
    """Generate through the LoRA, retrying seeds until it follows direction.

    One seed is a lottery with a small-data adapter (busts, duplicates,
    colour swaps). So render, check locally with the acceptance-gate
    probes (single figure, full body, outfit colours), and retry with
    deterministic alternate seeds. Returns the first passing render, or
    the last attempt with its failure reasons surfaced. Caps at three
    paid inferences.
    """
    from .validation import check_probe_image
    needs_body = profile == "tshirt" and shot != "close"
    attempts = [seed, seed + 1009, seed + 2011]
    last = None
    for attempt in (attempts if profile == "tshirt" else attempts[:1]):
        generated = await asyncio.to_thread(
            training.generate_with_adapter, adapter_ref, prompt, attempt,
            30, 7.0, lora_scale, ADAPTER_NEGATIVE)
        image = generated["image"]
        if profile != "tshirt":
            return (image, generated["source"], generated["model"],
                    attempt, [])
        try:
            raw = base64.b64decode(image.split(",", 1)[1], validate=True)
            probe = Image.open(io.BytesIO(raw)).convert("RGB")
            reasons = check_probe_image(
                probe, asset.get("dna") or {}, needs_body)
        except Exception as e:
            reasons = [f"could not check render: {e}"]
        last = (image, generated["source"], generated["model"], attempt,
                reasons)
        if not reasons:
            return last
    assert last is not None
    image, source, model, used, reasons = last
    return (image, source, model, used,
            ["direction not fully followed after 3 tries: " + "; ".join(reasons)]
            if reasons else [])


async def generate_graphic(asset: dict, extras: list[dict], seed: int = 42,
                           pose: str = "stand", expression: str = "calm",
                           shot: str = "medium", action: str = "",
                           profile: str = "storyboard",
                           composition: str = "vignette", elements: str = "",
                           phrase: str = "",
                           phrase_placement: str = "below",
                           adapter_ref: str | None = None,
                           lora_scale: float = 1.0) -> dict:
    """Generate one true-likeness graphic for `asset` in a format profile.

    Profiles: tshirt (isolated subject, transparent PNG, contained),
    storyboard (full-bleed panel), video (16:9 keyframe with headroom).
    A stored Modal adapter generates directly from the character's trigger
    word. Otherwise this uses reference-conditioned (/images/edits) Meta
    generation when a reference exists, or DNA prompt-only generation.
    Raises LLMError on any upstream failure.
    """
    if profile not in PROFILES:
        raise LLMError(f"Unknown format profile: {profile}")
    if adapter_ref is None and not settings.llm_live:
        raise LLMError("META_API_KEY is not set in backend/.env")
    prompt_extras = [] if profile == "tshirt" else extras
    prompt = build_likeness_prompt(asset, prompt_extras, pose, expression,
                                   shot, action)
    prompt += PROFILES[profile]
    if profile == "tshirt":
        comp = _COMPOSITIONS.get(composition, _COMPOSITIONS["vignette"])
        prompt += f" Design: {comp}."
        if action.strip():
            # Action scenes sprawl (gardens, ground, props everywhere) and
            # the cutout step then has to saw the design apart. Keep the
            # whole action compact so it survives isolation intact.
            prompt += (" Keep the action compact and fully visible: the "
                       "character, their tools, and any mound or pile "
                       "directly around their feet as one connected group, "
                       "no wide landscape or sprawling scenery.")
        if elements.strip():
            prompt += f" Decorative elements to include: {elements.strip()}."
    if adapter_ref:
        # SDXL text encoders silently drop everything past ~77 tokens, so
        # the long Meta prompt would lose the action, the containment
        # instructions, and the composition — the adapter would then draw a
        # generic full-bleed standing figure. Send it a compact prompt that
        # fits: trigger first, then identity, action, and format.
        trigger = dataset.namespaced_trigger(asset)
        core = (f"{asset.get('name', 'the figure')}: "
                f"{asset.get('description', '')}").strip()
        action_bit = (f" {action.strip().rstrip('.')}. "
                      if action.strip() else " ")
        # Without a framing word the adapter falls back to the close-up
        # busts that dominated the old training set.
        framing = ("close-up portrait" if shot == "close"
                   else "full body, head to toe")
        pose_bit = ("" if action.strip()
                    else f"{_POSE.get(pose, 'standing')}, ")
        style = ("single character only, no duplicates, no sticker sheet, "
                 "t-shirt graphic, isolated on a plain white background, "
                 "everything fully inside the frame")
        if profile == "tshirt" and elements.strip():
            style += f", with {elements.strip()}"
        prompt = (f"{trigger}, {core}{action_bit}{pose_bit}"
                  f"{expression} expression, {framing}, {style}.")
        # Modal's blocking SDK call runs in a worker thread so the API stays
        # responsive while paid GPU inference runs remotely.
        image, source, model, used_seed, dir_reasons = \
            await _generate_adapter_checked(
                asset, adapter_ref, prompt, seed, lora_scale,
                profile, shot)
    else:
        reference = asset.get("reference_image") or ""
        if reference.startswith("data:image/"):
            if action.strip():
                prompt += (" Match the reference image's exact art style, "
                           "proportions, face, colours, and outfit; change "
                           "the pose as needed to depict the requested "
                           "action with its objects.")
            else:
                prompt += (" Match the reference image's exact art style, proportions, "
                           "face, colours, and outfit; change only the pose, expression, "
                           "and scene.")
            image = await _post_image("edit", prompt, _reference_png(reference))
            source = "meta-image-edit"
        else:
            prompt += " Detailed illustration, clean background."
            image = await _post_image("generation", prompt, None)
            source = "meta-image-generation"
        model = settings.meta_image_model
        used_seed, dir_reasons = seed, []
    checks: list[str] = list(dir_reasons)
    if profile == "tshirt":
        image, iso_checks = isolate_subject(image)
        checks = checks + [c for c in iso_checks if c not in checks]
        if phrase.strip():
            image, phrase_checks = compose_phrase(image, phrase,
                                                  phrase_placement)
            checks = checks + [c for c in phrase_checks if c not in checks]
    elif profile == "video":
        image = _crop_widescreen(image)
    _bible, bible_hash = freeze_bible([asset, *extras])
    return {"image": image, "prompt": prompt, "model": model,
            "source": source, "profile": profile, "checks": checks,
            "bible_hash": bible_hash,
            "output_hash": canonical_hash([image]), "seed": used_seed}


async def generate_panel_graphic(asset: dict, extras: list[dict],
                                 panel: dict,
                                 profile: str = "storyboard") -> dict:
    """Generate one panel's image from its beat and staging."""
    layout = panel.get("layout", {})
    camera = panel.get("camera", {})
    return await generate_graphic(
        asset, extras, pose=layout.get("pose", "stand"),
        expression=layout.get("expression", "calm"),
        shot=camera.get("shot", "medium"),
        action=panel.get("action", ""), profile=profile)


def _hex_rgb(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        return None
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError:
        return None


async def score_likeness(image: str, bible_dna: dict) -> tuple[float, list[str]]:
    """Vision-check a generated image against the frozen Bible DNA.

    Re-encodes the image and compares the traits that carry identity:
    identity colours, facial hair, and the signature accessory.
    """
    try:
        encoded = (await encode_reference(image, "character"))["dna"]
    except Exception as e:
        return 0.0, [f"Likeness check unavailable: {e}"]
    checks: list[tuple[bool, str]] = []
    for field in ("skin", "hair", "outfit_top", "outfit_bottom", "accent"):
        want, got = _hex_rgb(bible_dna.get(field)), _hex_rgb(encoded.get(field))
        if want is None:
            continue
        close = got is not None and (
            sum(abs(a - b) for a, b in zip(want, got)) / 3 <= 45)
        checks.append((close, f"{field} drifted (Bible {bible_dna.get(field)} "
                              f"vs image {encoded.get(field)})"))
    if str(bible_dna.get("facial_hair", "none")).lower() not in ("", "none"):
        checks.append((str(encoded.get("facial_hair", "")).lower()
                       == str(bible_dna["facial_hair"]).lower(),
                       "facial hair missing or changed"))
    acc = str(bible_dna.get("accessory", "")).lower()
    if acc not in ("", "none"):
        words = {w for w in acc.replace(",", " ").split() if len(w) > 2}
        got_words = set(str(encoded.get("accessory", "")).lower().split())
        checks.append((bool(words & got_words),
                       f"accessory '{acc}' not recognised in image"))
    if not checks:
        return 100.0, []
    score = round(100.0 * sum(1 for ok, _ in checks if ok) / len(checks), 1)
    return score, [label for ok, label in checks if not ok]
