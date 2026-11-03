"""Reference image -> Asset Bible DNA encoding.

Uploading a turnaround/reference image pre-fills the asset's DNA form:
the Meta vision model reads the image and returns structured DNA, which
is validated against the same enums/colour fields the form and renderer
use. The user reviews and edits before saving — nothing is persisted
here. If vision is unavailable or returns unusable output, a local
palette fallback extracts dominant colours by image band (head / torso /
legs for characters) so the form still starts from the actual image.
Every result is labelled with its source so a guess never looks like a
reading.
"""
import base64
import binascii
import io
import re

from PIL import Image

from .config import settings
from .llm import LLMError, call_meta_vision_json

# field -> (kind, allowed enum values or None, default)
DNA_SPEC: dict[str, dict[str, tuple]] = {
    "character": {
        "form": ("enum", ("human", "fox", "cat", "creature"), "human"),
        "build": ("enum", ("slim", "average", "broad"), "average"),
        "skin": ("color", None, "#e8b08a"),
        "hair": ("color", None, "#3b2a20"),
        "hair_style": ("enum", ("short", "long", "bald"), "short"),
        "facial_hair": ("enum", ("none", "beard", "moustache"), "none"),
        "headwear_shape": ("enum", ("auto", "none", "brim", "pointed", "hood"), "auto"),
        "ear_shape": ("enum", ("auto", "none", "pointed", "round", "animal"), "auto"),
        "eyes": ("color", None, "#2b2b2b"),
        "outfit_top": ("color", None, "#256d5a"),
        "outfit_bottom": ("color", None, "#33404f"),
        "accent": ("color", None, "#d9a03c"),
        "accessory": ("text", None, "none"),
        "marking": ("text", None, "none"),
    },
    "object": {
        "shape": ("enum", ("orb", "crystal", "box", "book", "sword", "key"), "orb"),
        "material": ("text", None, ""),
        "primary": ("color", None, "#b08a3c"),
        "secondary": ("color", None, "#6e5426"),
        "accent": ("color", None, "#ffe9a8"),
        "marking": ("text", None, "none"),
        "glow": ("enum", ("none", "soft", "bright"), "none"),
    },
    "environment": {
        "biome": ("text", None, ""),
        "time": ("enum", ("day", "dusk", "night"), "day"),
        "sky": ("color", None, "#a8c8e0"),
        "ground": ("color", None, "#7a8a5a"),
        "primary": ("color", None, "#5a7a9a"),
        "secondary": ("color", None, "#3c4c5e"),
        "accent": ("color", None, "#d9a03c"),
        "landmark": ("enum", ("towers", "mountains", "forest"), "towers"),
        "weather": ("text", None, "clear"),
    },
}

_HEX = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
MAX_IMAGE_CHARS = 2_800_000

SYSTEM = ("You are the Asset Bible encoder for a consistent multi-scene story "
          "engine. You read ONE reference image and return structured DNA as "
          "JSON. Colours must be hex like #b91c1c sampled from the image, not "
          "invented. Reply with ONLY the JSON object requested.")


def _norm_hex(value: object, default: str) -> str:
    if not isinstance(value, str):
        return default
    m = _HEX.match(value.strip())
    if not m:
        return default
    digits = m.group(1).lower()
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return f"#{digits}"


def _coerce(raw: object, kind: str) -> dict:
    """Keep only known fields, with valid enums/hex values throughout."""
    spec = DNA_SPEC[kind]
    source = raw if isinstance(raw, dict) else {}
    dna = {}
    for field, (ftype, allowed, default) in spec.items():
        value = source.get(field)
        if ftype == "color":
            dna[field] = _norm_hex(value, default)
        elif ftype == "enum":
            text = str(value).strip().lower() if value is not None else ""
            dna[field] = text if text in allowed else default
        else:
            text = str(value).strip()[:60] if value is not None else ""
            dna[field] = text or default
    return dna


def _prompt(kind: str) -> str:
    fields = []
    for field, (ftype, allowed, _default) in DNA_SPEC[kind].items():
        if ftype == "color":
            fields.append(f'"{field}": "<hex colour from the image>"')
        elif ftype == "enum":
            fields.append(f'"{field}": "<one of: {", ".join(allowed)}>"')
        else:
            fields.append(f'"{field}": "<short phrase from the image>"')
    extra = (' For "accessory", name the single most recognisable item and keep '
             "the word hat, cap, goggles, glasses, or scarf in it when one is "
             "worn, since the renderer keys off that word."
             if kind == "character" else "")
    return (f"This is a {kind} reference image. Return a JSON object of the form "
            f'{{"description": "<one short sentence describing it>", '
            f'"dna": {{{", ".join(fields)}}}}}.{extra}')


def _decode(image: str) -> Image.Image:
    if not isinstance(image, str) or not image.startswith("data:image/"):
        raise ValueError("Reference image must be an image data URL")
    if len(image) > MAX_IMAGE_CHARS:
        raise ValueError("Reference image is too large")
    try:
        raw = base64.b64decode(image.split(",", 1)[1], validate=True)
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (binascii.Error, IndexError, Exception) as e:
        raise ValueError(f"Reference image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _dominant(img: Image.Image) -> tuple[int, int, int]:
    small = img.resize((48, 48))
    colours = small.quantize(colors=6, method=Image.MEDIANCUT)
    palette = colours.getpalette() or []
    counts = sorted(colours.getcolors() or [], reverse=True)
    if not counts:
        return (128, 128, 128)
    idx = counts[0][1]
    return tuple(palette[idx * 3: idx * 3 + 3])  # type: ignore[return-value]


def palette_fallback(image: str, kind: str) -> tuple[dict, list[str]]:
    """DNA guessed from dominant colours. Bands assume an upright figure."""
    img = _decode(image)
    w, h = img.size
    centre = img.crop((int(w * 0.2), 0, int(w * 0.8), h))
    cw, ch = centre.size

    def band(y0: float, y1: float) -> tuple[int, int, int]:
        return _dominant(centre.crop((0, int(ch * y0), cw, max(int(ch * y1), 1))))

    flat = img.resize((64, 64)).quantize(colors=8, method=Image.MEDIANCUT)
    pal = flat.getpalette() or []
    palette = [_hex(tuple(pal[i * 3: i * 3 + 3]))  # type: ignore[arg-type]
               for i in range(min(8, len(pal) // 3))]
    top, mid, low = band(0.05, 0.35), band(0.35, 0.65), band(0.65, 0.98)
    if kind == "character":
        dna = {"hair": _hex(top), "outfit_top": _hex(mid), "outfit_bottom": _hex(low)}
    elif kind == "environment":
        dna = {"sky": _hex(top), "primary": _hex(mid), "ground": _hex(low)}
    else:
        dna = {"primary": _hex(mid), "secondary": _hex(low), "accent": _hex(top)}
    return _coerce(dna, kind), palette


async def encode_reference(image: str, kind: str) -> dict:
    """Encode one reference image into suggested DNA for `kind`.

    Returns {dna, description, source, model, fallback, fallback_reason,
    palette}. Raises ValueError for a payload that is not a decodable
    image data URL.
    """
    if kind not in DNA_SPEC:
        raise ValueError(f"Unknown asset kind: {kind}")
    _decode(image)  # fail fast on garbage before spending a model call
    try:
        raw = await call_meta_vision_json(SYSTEM, _prompt(kind), image)
        raw_dna = raw.get("dna") if isinstance(raw, dict) else None
        if not isinstance(raw_dna, dict) or not raw_dna:
            raise LLMError("Vision model returned no DNA object")
        description = raw.get("description")
        return {"dna": _coerce(raw_dna, kind),
                "description": description.strip()[:300] if isinstance(description, str) else "",
                "source": "meta-vision", "model": settings.meta_model,
                "fallback": False, "fallback_reason": "", "palette": []}
    except LLMError as e:
        dna, palette = palette_fallback(image, kind)
        return {"dna": dna, "description": "", "source": "palette",
                "model": settings.meta_model, "fallback": True,
                "fallback_reason": str(e), "palette": palette}
