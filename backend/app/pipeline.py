"""Five-agent pipeline for consistent multi-scene stories.

Agents (always in this order):
  1. Canon Keeper       freezes asset DNA into a versioned, hashed Bible
  2. Story Architect    breaks the premise into panel beats on a story arc
                        (Meta model; template fallback when offline/failing)
  3. Layout Director    assigns camera, pose, expression, blocking per panel
                        (Meta model from allowed enums; fallback likewise)
  4. Panel Renderer     draws each panel as SVG strictly from the Bible
  5. Consistency Critic verifies every panel against the Bible, reports drift

Determinism contract: Canon, Renderer, and Critic are deterministic — the
same frozen Bible always renders and scores identically, and with the same
seed the offline story/layout fallback is byte-identical (same output_hash).
When the Meta model is live, beats/layout come from the model, so narrative
content varies; Bible, rendering, scoring, and hashing do not.
"""
import hashlib
import html
import json
import random
import re

from .config import settings
from .llm import LLMError, call_meta_json

# --------------------------------------------------------------------------
# Deterministic primitives
# --------------------------------------------------------------------------

def seeded_rng(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def canonical_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def short_hash(h: str, n: int = 10) -> str:
    return h[:n] if h else ""


def _keywords(text: str, limit: int = 8) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "but", "with", "into", "from", "that",
            "this", "when", "then", "they", "them", "their", "must", "will"}
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
    seen: list[str] = []
    for w in words:
        lw = w.lower()
        if lw not in stop and lw not in seen:
            seen.append(lw)
    return seen[:limit]


# --------------------------------------------------------------------------
# Agent 1 — Canon Keeper
# --------------------------------------------------------------------------

CHARACTER_DEFAULTS = {
    "form": "human", "build": "average", "skin": "#e8b08a", "hair": "#3b2a20",
    "hair_style": "short", "eyes": "#2b2b2b", "outfit_top": "#256d5a",
    "outfit_bottom": "#33404f", "primary": "#256d5a", "secondary": "#33404f",
    "accent": "#d9a03c", "accessory": "none", "marking": "none",
}
OBJECT_DEFAULTS = {
    "shape": "orb", "material": "brass", "primary": "#b08a3c", "secondary": "#6e5426",
    "accent": "#ffe9a8", "marking": "none", "glow": "none",
}
ENV_DEFAULTS = {
    "biome": "city", "time": "day", "sky": "#a8c8e0", "ground": "#7a8a5a",
    "primary": "#5a7a9a", "secondary": "#3c4c5e", "accent": "#d9a03c",
    "landmark": "towers", "weather": "clear",
}
_DEFAULTS = {"character": CHARACTER_DEFAULTS, "object": OBJECT_DEFAULTS,
             "environment": ENV_DEFAULTS}


def freeze_bible(assets: list[dict]) -> tuple[dict, str]:
    """Freeze asset DNA into the Bible. Missing DNA fields get stable
    defaults derived from the asset name, so a half-filled form still
    renders identically on every run."""
    entries = {}
    for a in assets:
        defaults = dict(_DEFAULTS.get(a["kind"], {}))
        name_rng = seeded_rng("defaults", a["kind"], a["name"])
        # Stable per-name accent so unnamed palettes still differ per asset.
        if "accent" not in (a.get("dna") or {}):
            defaults["accent"] = "#%02x%02x%02x" % (
                name_rng.randint(140, 230), name_rng.randint(90, 180), name_rng.randint(40, 110))
        dna = {**defaults, **{k: v for k, v in (a.get("dna") or {}).items() if v not in ("", None)}}
        entry = {"id": a["id"], "kind": a["kind"], "name": a["name"],
                 "description": a.get("description", ""), "version": a.get("version", 1),
                 "dna": dna, "dna_hash": canonical_hash(dna)}
        entries[a["id"]] = entry
    bible = {"version": 1, "assets": entries,
             "asset_order": [a["id"] for a in assets]}
    return bible, canonical_hash(bible)


# --------------------------------------------------------------------------
# Agent 2 — Story Architect
# --------------------------------------------------------------------------

ARC = [
    (0.00, "Setup", "Establish the world and the hero's ordinary moment."),
    (0.18, "Spark", "Something unusual interrupts the routine."),
    (0.36, "Discovery", "The hero investigates and learns the stakes."),
    (0.54, "Escalation", "The problem grows; a first plan fails."),
    (0.70, "Turning point", "A choice or revelation changes everything."),
    (0.85, "Climax", "The hero acts decisively at the highest stakes."),
    (1.01, "Resolution", "The world settles into its new state."),
]
ACTION_TEMPLATES = [
    "{hero} studies the {kw1}, unaware the day is about to change.",
    "{hero} follows the trail of the {kw1} deeper in.",
    "A sudden shift around the {kw2} forces {hero} to improvise.",
    "{hero} confronts the truth about the {kw1}.",
    "With the {kw2} at stake, {hero} makes the decisive move.",
    "{hero} looks back — the {kw1} will never be the same.",
]
DIALOGUE_TEMPLATES = [
    "\"We start with the {kw1}. No shortcuts.\"",
    "\"Did you see that? The {kw1} moved.\"",
    "\"If the {kw2} fails, everything fails.\"",
    "\"I know what the {kw1} really is now.\"",
    "\"Now. Together — for the {kw2}.\"",
    "\"It's over. And it worked.\"",
]


def _arc_phase(fraction: float) -> tuple[str, str]:
    for limit, title, desc in ARC:
        if fraction < limit:
            return title, desc
    return ARC[-1][1], ARC[-1][2]


def _build_beats_offline(premise: str, panel_count: int, seed: int, bible: dict) -> list[dict]:
    rng = seeded_rng("story", seed, premise, panel_count)
    kws = _keywords(premise) or ["signal", "journey"]
    kw1, kw2 = kws[0], kws[min(1, len(kws) - 1)]
    heroes = [e["name"] for e in bible["assets"].values() if e["kind"] == "character"]
    hero = heroes[0] if heroes else "The hero"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", premise or "") if s.strip()]
    beats = []
    for i in range(panel_count):
        fraction = i / max(panel_count - 1, 1) if panel_count > 1 else 0.0
        title, desc = _arc_phase(fraction + 0.001)
        if i < len(sentences):
            action = sentences[i]
        else:
            tpl = ACTION_TEMPLATES[min(int(fraction * len(ACTION_TEMPLATES)),
                                       len(ACTION_TEMPLATES) - 1)]
            action = tpl.format(hero=hero, kw1=kw1, kw2=kw2)
        dialogue = ""
        if i % 2 == 1 or i == panel_count - 1:
            tpl = DIALOGUE_TEMPLATES[min(int(fraction * len(DIALOGUE_TEMPLATES)),
                                         len(DIALOGUE_TEMPLATES) - 1)]
            dialogue = tpl.format(kw1=kw1, kw2=kw2)
        narration = desc if i in (0, panel_count - 1) else ""
        beats.append({"index": i, "beat_title": title, "action": action,
                      "dialogue": dialogue, "narration": narration,
                      "_phase": round(fraction, 3), "_rng": rng.random()})
    return beats


def _stage_meta(source: str, fallback_reason: str | None = None) -> dict:
    """StageRun.detail provenance for the two Meta-backed agents."""
    fallback = source != "meta_model"
    return {"source": source,
            "generated_by": "meta_model" if not fallback else "offline_fallback",
            "model": settings.meta_model,
            "fallback": fallback,
            "fallback_used": fallback,
            "fallback_reason": fallback_reason}


def _validate_beats(data: dict, panel_count: int) -> list[dict]:
    """Strict-schema check on Meta beats. Raises ValueError on any drift."""
    raw = data.get("beats") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != panel_count:
        raise ValueError(f"expected {panel_count} beats")
    beats = []
    for pos, b in enumerate(sorted(raw, key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0)):
        if not isinstance(b, dict) or int(b.get("index", -1)) != pos:
            raise ValueError("beat indexes must be 0..panel_count-1")
        fields = {k: b.get(k) for k in ("beat_title", "action", "dialogue", "narration")}
        if not all(isinstance(v, str) for v in fields.values()):
            raise ValueError("beat fields must be strings")
        if not fields["beat_title"].strip() or not fields["action"].strip():
            raise ValueError("beat_title and action must be non-empty")
        fraction = pos / max(panel_count - 1, 1) if panel_count > 1 else 0.0
        beats.append({"index": pos,
                      "beat_title": fields["beat_title"].strip(),
                      "action": fields["action"].strip(),
                      "dialogue": fields["dialogue"].strip(),
                      "narration": fields["narration"].strip(),
                      "_phase": round(fraction, 3), "_rng": 0.0})
    return beats


async def _build_beats_with_meta(premise: str, panel_count: int, seed: int,
                                 bible: dict) -> tuple[list[dict], dict]:
    if not settings.llm_live:
        return _build_beats_offline(premise, panel_count, seed, bible), \
            _stage_meta("offline_fallback", "llm_not_live")
    chars = [{"id": e["id"], "name": e["name"]}
             for e in bible["assets"].values() if e["kind"] == "character"]
    others = [{"id": e["id"], "name": e["name"], "kind": e["kind"]}
              for e in bible["assets"].values() if e["kind"] != "character"]
    arc = []
    for i in range(panel_count):
        fraction = i / max(panel_count - 1, 1) if panel_count > 1 else 0.0
        arc.append(f"panel {i}: {_arc_phase(fraction + 0.001)[0]}")
    system = ("You are the Story Architect in a deterministic comic pipeline. "
              "You only assign actions and dialogue to characters declared in "
              "the Asset Bible. Reply with JSON only, no markdown.")
    user = (
        f"Break this premise into exactly {panel_count} comic panel beats, "
        f"mapped in order across the classic narrative arc "
        f"(Setup -> Spark -> Discovery -> Escalation -> Turning point -> "
        f"Climax -> Resolution):\nPremise: {premise}\n"
        f"Arc mapping: {'; '.join(arc)}\n"
        f"Declared characters (id, name) — the only characters who may act "
        f"or speak: {json.dumps(chars)}\n"
        f"Other Bible assets: {json.dumps(others)}\n"
        "Return exactly this JSON schema and nothing else: "
        '{"beats": [{"index": int, "beat_title": str, "action": str, '
        '"dialogue": str, "narration": str}]} '
        f"with indexes 0..{panel_count - 1}. dialogue is spoken by a declared "
        "character (empty string when nobody speaks); narration is a short "
        "caption (may be empty)."
    )
    try:
        data = await call_meta_json(system, user)
        return _validate_beats(data, panel_count), _stage_meta("meta_model")
    except (LLMError, ValueError, TypeError) as e:
        return _build_beats_offline(premise, panel_count, seed, bible), \
            _stage_meta("offline_fallback", f"{type(e).__name__}: {e}")


async def build_beats(premise: str, panel_count: int, seed: int, bible: dict) -> list[dict]:
    """Meta-backed beats when live; the template arc above when offline or
    the API call/validation fails."""
    beats, _meta = await _build_beats_with_meta(premise, panel_count, seed, bible)
    return beats


# --------------------------------------------------------------------------
# Agent 3 — Layout Director
# --------------------------------------------------------------------------

SHOTS = ["wide", "medium", "medium", "close", "medium", "wide"]
ANGLES = ["eye-level", "low", "eye-level", "high", "eye-level", "low"]
POSES = ["stand", "walk", "reach", "stand", "leap", "stand"]
EXPRESSIONS = ["calm", "curious", "surprised", "determined", "determined", "relieved"]


def _direct_layout_offline(beats: list[dict], seed: int, bible: dict) -> list[dict]:
    chars = [eid for eid in bible["asset_order"]
             if bible["assets"][eid]["kind"] == "character"]
    envs = [eid for eid in bible["asset_order"]
            if bible["assets"][eid]["kind"] == "environment"]
    objs = [eid for eid in bible["asset_order"]
            if bible["assets"][eid]["kind"] == "object"]
    layouts = []
    for beat in beats:
        i = beat["index"]
        rng = seeded_rng("layout", seed, i)
        slot = min(int(beat["_phase"] * 6), 5)
        # Supporting characters join from the Discovery beat onward.
        present = chars[:1] if beat["_phase"] < 0.36 else chars[:2]
        layouts.append({
            "index": i,
            "camera": {"shot": SHOTS[slot], "angle": ANGLES[slot],
                       "lens": {"wide": "24mm", "medium": "35mm", "close": "85mm"}[SHOTS[slot]]},
            "pose": POSES[slot], "expression": EXPRESSIONS[slot],
            "character_ids": present,
            "environment_id": envs[0] if envs else None,
            "object_ids": objs if (objs and beat["_phase"] >= 0.36) else [],
            "focus_x": [140, 200, 260][i % 3],
            "jitter": round(rng.uniform(-6, 6), 2),
        })
    return layouts


ALLOWED_SHOTS = ["wide", "medium", "close"]
ALLOWED_ANGLES = ["eye-level", "low", "high"]
ALLOWED_POSES = ["stand", "walk", "reach", "leap"]
ALLOWED_EXPRESSIONS = ["calm", "curious", "surprised", "determined", "relieved"]
_LENS = {"wide": "24mm", "medium": "35mm", "close": "85mm"}


def _validate_layouts(data: dict, beats: list[dict], fallback: list[dict]) -> list[dict]:
    """Strict enum check on Meta layouts. Raises ValueError on any drift.

    Only camera/pose/expression come from the model; Bible blocking
    (characters, environment, objects, focus, jitter) stays deterministic.
    """
    raw = data.get("layouts") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != len(beats):
        raise ValueError(f"expected {len(beats)} layouts")
    layouts = []
    for pos, item in enumerate(sorted(raw, key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0)):
        if not isinstance(item, dict) or int(item.get("index", -1)) != pos:
            raise ValueError("layout indexes must match beat indexes")
        shot, angle = item.get("shot"), item.get("angle")
        pose, expression = item.get("pose"), item.get("expression")
        if shot not in ALLOWED_SHOTS or angle not in ALLOWED_ANGLES:
            raise ValueError(f"camera outside allowed enums: {shot}/{angle}")
        if pose not in ALLOWED_POSES or expression not in ALLOWED_EXPRESSIONS:
            raise ValueError(f"pose/expression outside allowed enums: {pose}/{expression}")
        merged = dict(fallback[pos])
        merged["camera"] = {"shot": shot, "angle": angle, "lens": _LENS[shot]}
        merged["pose"], merged["expression"] = pose, expression
        layouts.append(merged)
    return layouts


async def _direct_layout_with_meta(beats: list[dict], seed: int,
                                   bible: dict) -> tuple[list[dict], dict]:
    fallback = _direct_layout_offline(beats, seed, bible)
    if not settings.llm_live:
        return fallback, _stage_meta("offline_fallback", "llm_not_live")
    brief = [{"index": b["index"], "beat_title": b["beat_title"],
              "action": b["action"], "dialogue": b.get("dialogue", "")}
             for b in beats]
    system = ("You are the Layout Director in a deterministic comic pipeline. "
              "Choose camera and staging only from the allowed enums. "
              "Reply with JSON only, no markdown.")
    user = (
        "For each story beat, judge its narrative tension (quiet setup, "
        "rising discovery, decisive climax) and choose cinematic staging. "
        f"Allowed shot: {ALLOWED_SHOTS}; angle: {ALLOWED_ANGLES}; "
        f"pose: {ALLOWED_POSES}; expression: {ALLOWED_EXPRESSIONS}. "
        "Use wide/eye-level for calm setups, close/low for climaxes, and "
        "match pose and expression to the action.\n"
        f"Beats: {json.dumps(brief)}\n"
        "Return exactly this JSON and nothing else: "
        '{"layouts": [{"index": int, "shot": str, "angle": str, '
        '"pose": str, "expression": str}]} with one entry per beat, '
        "indexes matching the beats."
    )
    try:
        data = await call_meta_json(system, user, temperature=0.3)
        return _validate_layouts(data, beats, fallback), _stage_meta("meta_model")
    except (LLMError, ValueError, TypeError) as e:
        return fallback, _stage_meta("offline_fallback", f"{type(e).__name__}: {e}")


async def direct_layout(beats: list[dict], seed: int, bible: dict) -> list[dict]:
    """Meta-backed staging when live; modulo-based assignment when offline
    or the API call/validation fails. Blocking always comes from the Bible."""
    layouts, _meta = await _direct_layout_with_meta(beats, seed, bible)
    return layouts


# --------------------------------------------------------------------------
# Agent 4 — Panel Renderer (SVG, strictly from the Bible)
# --------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _wrap(text: str, width: int = 34, max_lines: int = 3) -> list[str]:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
            if len(lines) == max_lines:
                break
        else:
            cur = f"{cur} {w}".strip()
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if lines and len(" ".join(words)) > sum(len(x) for x in lines) + max_lines:
        lines[-1] = lines[-1].rstrip(".") + "…"
    return lines


def _character_svg(entry: dict, x: float, ground: float, scale: float,
                   pose: str, expression: str) -> str:
    d = entry["dna"]
    skin, hair, top = d.get("skin", "#e8b08a"), d.get("hair", "#333"), d.get("outfit_top", "#256d5a")
    bottom, accent = d.get("outfit_bottom", "#33404f"), d.get("accent", "#d9a03c")
    accessory = str(d.get("accessory", "none")).lower()
    marking = str(d.get("marking", "none")).lower()
    form = str(d.get("form", "human")).lower()
    tilt = {"stand": 0, "walk": 4, "reach": -3, "leap": -8}.get(pose, 0)
    arm = {"stand": 18, "walk": 38, "reach": -52, "leap": -70}.get(pose, 18)
    mouth = {"calm": "M -7 6 Q 0 9 7 6", "curious": "M -6 7 Q 0 5 6 7",
             "surprised": "M -4 8 Q 0 13 4 8", "determined": "M -7 8 L 7 8",
             "relieved": "M -7 5 Q 0 11 7 5"}.get(expression, "M -7 6 Q 0 9 7 6")
    brow = -3 if expression in ("determined", "surprised") else 0
    build = str(d.get("build", "average")).lower()
    torso_w = {"slim": 26, "average": 32, "broad": 40}.get(build, 32)
    half = torso_w // 2
    facial = str(d.get("facial_hair", "none")).lower()
    ear_shape = str(d.get("ear_shape", "auto")).lower()
    if ear_shape in ("", "auto"):  # legacy DNA: derive from form
        ear_shape = "animal" if form in ("fox", "cat") else "none"
    ears = ""
    if ear_shape == "animal":
        ears = (f'<g data-ears="animal"><path d="M -16 -52 L -10 -70 L -2 -54 Z" fill="{hair}"/>'
                f'<path d="M 16 -52 L 10 -70 L 2 -54 Z" fill="{hair}"/></g>')
    elif ear_shape == "pointed":
        ears = (f'<g data-ears="pointed"><path d="M -16 -46 L -22 -56 L -12 -52 Z" fill="{skin}"/>'
                f'<path d="M 16 -46 L 22 -56 L 12 -52 Z" fill="{skin}"/></g>')
    elif ear_shape == "round":
        ears = (f'<g data-ears="round"><circle cx="-17" cy="-42" r="5" fill="{skin}"/>'
                f'<circle cx="17" cy="-42" r="5" fill="{skin}"/></g>')
    acc = ""
    if "goggle" in accessory or "glasses" in accessory:
        acc = (f'<g data-accessory="{_esc(accessory)}"><circle cx="-9" cy="-38" r="7" fill="none" stroke="{accent}" stroke-width="3"/>'
               f'<circle cx="9" cy="-38" r="7" fill="none" stroke="{accent}" stroke-width="3"/>'
               f'<line x1="-2" y1="-38" x2="2" y2="-38" stroke="{accent}" stroke-width="3"/></g>')
    elif "hat" in accessory or "cap" in accessory:
        headwear = str(d.get("headwear_shape", "auto")).lower()
        pointed = headwear == "pointed" or (
            headwear in ("", "auto")
            and any(w in accessory for w in ("pointed", "tall", "cone")))
        if pointed:
            acc = (f'<g data-accessory="{_esc(accessory)}" data-hat="pointed">'
                   f'<path d="M -17 -46 L -4 -86 Q -1 -97 5 -89 L 17 -46 Q 0 -53 -17 -46 Z" fill="{accent}"/>'
                   f'<rect x="-20" y="-49" width="40" height="6" rx="3" fill="{accent}"/></g>')
        else:
            acc = f'<path data-accessory="{_esc(accessory)}" data-hat="brim" d="M -18 -50 Q 0 -66 18 -50 L 24 -46 L -24 -46 Z" fill="{accent}"/>'
    elif "scarf" in accessory:
        acc = f'<rect data-accessory="{_esc(accessory)}" x="-14" y="-22" width="28" height="8" rx="4" fill="{accent}"/>'
    elif accessory not in ("", "none"):
        acc = f'<circle data-accessory="{_esc(accessory)}" cx="16" cy="-18" r="5" fill="{accent}"/>'
    cheeks = ""
    stripe = ""
    if "cheek" in marking:
        cheeks = (f'<g data-cheeks="true"><circle cx="-9" cy="-29" r="3.4" fill="#e88a7a" opacity="0.7"/>'
                  f'<circle cx="9" cy="-29" r="3.4" fill="#e88a7a" opacity="0.7"/></g>')
    elif marking not in ("", "none"):
        stripe = f'<rect x="-15" y="-6" width="30" height="5" fill="{accent}"/>'
    beard = ""
    if facial == "beard":
        beard = (f'<path data-beard="true" d="M -13 -31 Q -15 -10 0 -5 Q 15 -10 13 -31 '
                 f'Q 7 -23 0 -23 Q -7 -23 -13 -31 Z" fill="{hair}"/>')
    elif facial == "moustache":
        beard = (f'<path data-beard="moustache" d="M -8 -27 Q -3 -23 0 -26 Q 3 -23 8 -27 '
                 f'Q 4 -20 0 -21 Q -4 -20 -8 -27 Z" fill="{hair}"/>')
    hair_top = (f'<path d="M -17 -44 Q 0 -62 17 -44 L 17 -50 Q 0 -64 -17 -50 Z" fill="{hair}"/>'
                if str(d.get("hair_style", "short")).lower() != "bald" else "")
    return (
        f'<g transform="translate({x:.1f} {ground:.1f}) scale({scale:.2f}) rotate({tilt})">'
        f'<rect x="{-half + 3}" y="-34" width="10" height="34" rx="4" fill="{bottom}"/>'
        f'<rect x="{half - 13}" y="-34" width="10" height="34" rx="4" fill="{bottom}"/>'
        f'<g transform="rotate({arm} {-half} -58)"><rect x="{-half - 4}" y="-58" width="8" height="30" rx="4" fill="{top}"/></g>'
        f'<g transform="rotate({-arm} {half} -58)"><rect x="{half - 4}" y="-58" width="8" height="30" rx="4" fill="{top}"/></g>'
        f'<rect x="{-half}" y="-62" width="{torso_w}" height="30" rx="7" fill="{top}"/>{stripe}'
        f'{ears}<circle cx="0" cy="-40" r="17" fill="{skin}"/>{beard}{hair_top}'
        f'<circle cx="-6" cy="-38" r="2" fill="{d.get("eyes", "#222")}"/>'
        f'<circle cx="6" cy="-38" r="2" fill="{d.get("eyes", "#222")}"/>'
        f'<ellipse cx="0" cy="-33" rx="3" ry="4.2" fill="{skin}" stroke="#c07a4e" stroke-width="0.8"/>'
        f'{cheeks}'
        f'<path d="M -10 {-46 + brow} L -3 {-44 + brow}" stroke="#222" stroke-width="1.6"/>'
        f'<path d="M 3 {-44 + brow} L 10 {-46 + brow}" stroke="#222" stroke-width="1.6"/>'
        f'<path d="{mouth}" transform="translate(0 -40)" stroke="#7a4632" fill="none" stroke-width="1.8"/>'
        f'{acc}</g>'
    )


def _object_svg(entry: dict, x: float, ground: float, scale: float) -> str:
    d = entry["dna"]
    primary, accent = d.get("primary", "#b08a3c"), d.get("accent", "#ffe9a8")
    shape = str(d.get("shape", "orb")).lower()
    glow = (f'<circle cx="0" cy="-16" r="24" fill="{accent}" opacity="0.25"/>'
            if str(d.get("glow", "none")).lower() not in ("", "none") else "")
    if "crystal" in shape or "gem" in shape:
        body = f'<path d="M 0 -34 L 13 -14 L 0 4 L -13 -14 Z" fill="{primary}" stroke="{accent}" stroke-width="2"/>'
    elif "box" in shape or "crate" in shape or "book" in shape:
        body = f'<rect x="-14" y="-28" width="28" height="24" rx="3" fill="{primary}" stroke="{accent}" stroke-width="2"/>'
    elif "sword" in shape or "blade" in shape or "key" in shape:
        body = f'<path d="M 0 -36 L 4 -8 L 12 -8 L 12 -2 L -12 -2 L -12 -8 L -4 -8 Z" fill="{primary}" stroke="{accent}" stroke-width="1.6"/>'
    else:
        body = f'<circle cx="0" cy="-16" r="13" fill="{primary}" stroke="{accent}" stroke-width="2.5"/>'
    return f'<g transform="translate({x:.1f} {ground:.1f}) scale({scale:.2f})">{glow}{body}</g>'


def render_panel(beat: dict, layout: dict, bible: dict, seed: int) -> tuple[str, str]:
    """Draw one panel. Returns (svg, locked_prompt)."""
    assets = bible["assets"]
    env = assets.get(layout.get("environment_id") or "", {})
    ed = env.get("dna", ENV_DEFAULTS)
    sky, ground_c = ed.get("sky", "#a8c8e0"), ed.get("ground", "#7a8a5a")
    rng = seeded_rng("render", seed, beat["index"])
    horizon = 205 + layout.get("jitter", 0)
    shot = layout["camera"]["shot"]
    scale = {"wide": 0.85, "medium": 1.25, "close": 2.1}[shot]
    char_y = 268 if shot != "close" else 300

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" '
             f'data-panel="{beat["index"]}" data-shot="{shot}" role="img">']
    parts.append(f'<rect width="400" height="300" fill="{sky}"/>')
    if str(ed.get("time", "day")).lower() in ("night", "dusk"):
        parts.append('<rect width="400" height="300" fill="#1c2a44" opacity="0.35"/>')
        parts.append('<circle cx="338" cy="42" r="17" fill="#f2ead0"/>')
    else:
        parts.append('<circle cx="338" cy="42" r="19" fill="#ffe9a8" opacity="0.9"/>')
    landmark = str(ed.get("landmark", "towers")).lower()
    if "mountain" in landmark or str(ed.get("biome", "")).lower() in ("mountains", "wilds"):
        parts.append(f'<path d="M0 {horizon} L70 118 L130 {horizon} L205 132 L275 {horizon} L330 150 L400 {horizon} Z" fill="{ed.get("secondary", "#3c4c5e")}" opacity="0.55"/>')
    elif "forest" in landmark or "tree" in landmark:
        for tx in (28, 92, 318, 362):
            parts.append(f'<path d="M{tx} {horizon} l12 -44 l12 44 Z" fill="{ed.get("secondary", "#3c4c5e")}" opacity="0.6"/>')
    else:
        for bx, bh in ((18, 64), (66, 92), (300, 78), (348, 56)):
            parts.append(f'<rect x="{bx}" y="{horizon - bh}" width="38" height="{bh}" fill="{ed.get("secondary", "#3c4c5e")}" opacity="0.5"/>')
    parts.append(f'<rect y="{horizon}" width="400" height="{300 - horizon}" fill="{ground_c}"/>')

    for oid in layout.get("object_ids", []):
        if oid in assets:
            parts.append(_object_svg(assets[oid], 316 + rng.uniform(-4, 4), 272, 1.0))
    chars = [assets[c] for c in layout.get("character_ids", []) if c in assets]
    for n, entry in enumerate(chars):
        cx = layout.get("focus_x", 200) + (n * 92 - (len(chars) - 1) * 46)
        if shot == "close":
            cx = 200 + (n * 120 - (len(chars) - 1) * 60)
        parts.append(_character_svg(entry, cx, char_y, scale if n == 0 else scale * 0.92,
                                     layout.get("pose", "stand"), layout.get("expression", "calm")))
    if layout.get("narration"):
        lines = _wrap(layout["narration"], 44, 2)
        tspans = "".join(f'<tspan x="12" dy="{0 if j == 0 else 13}">{_esc(t)}</tspan>' for j, t in enumerate(lines))
        parts.append(f'<g><rect x="6" y="6" width="388" height="{12 + 13 * len(lines)}" fill="#111827" opacity="0.82"/>'
                     f'<text x="12" y="20" font-family="monospace" font-size="10.5" fill="#f4f4f5">{tspans}</text></g>')
    if beat.get("dialogue") and chars:
        lines = _wrap(beat["dialogue"], 30, 3)
        tspans = "".join(f'<tspan x="{layout.get("focus_x", 200) - 62}" dy="{0 if j == 0 else 12}">{_esc(t)}</tspan>' for j, t in enumerate(lines))
        parts.append(f'<g><rect x="{layout.get("focus_x", 200) - 70}" y="52" width="150" height="{16 + 12 * len(lines)}" rx="7" fill="#ffffff" stroke="#111827"/>'
                     f'<text y="66" font-family="sans-serif" font-size="10" fill="#111827">{tspans}</text></g>')
    lines = _wrap(beat.get("action", ""), 46, 2)
    if lines:  # standalone graphics have no caption bar
        tspans = "".join(f'<tspan x="10" dy="{0 if j == 0 else 12}">{_esc(t)}</tspan>' for j, t in enumerate(lines))
        parts.append(f'<g><rect x="0" y="{300 - 14 - 12 * len(lines)}" width="400" height="{14 + 12 * len(lines)}" fill="#ffffff" opacity="0.92"/>'
                     f'<text y="{300 - 16 - 12 * (len(lines) - 1)}" font-family="sans-serif" font-size="10" fill="#27272a">{tspans}</text></g>')
    parts.append('<rect width="400" height="300" fill="none" stroke="#111827" stroke-width="3"/></svg>')
    svg = "".join(parts)

    char_desc = "; ".join(
        f"{e['name']} ({e['dna'].get('form', '')}, {e['dna'].get('outfit_top', '')} top, "
        f"accessory: {e['dna'].get('accessory', 'none')})" for e in chars) or "no character"
    prompt = (f"{shot} shot, {layout['camera']['angle']}, {layout['camera']['lens']}: {beat.get('action', '')} "
              f"| Characters: {char_desc} | Location: {env.get('name', 'unspecified')} "
              f"({ed.get('biome', '')}, {ed.get('time', '')}) | Pose: {layout.get('pose')}, "
              f"expression: {layout.get('expression')} | Locked to Asset Bible, seed {seed}")
    return svg, prompt


# --------------------------------------------------------------------------
# Agent 5 — Consistency Critic
# --------------------------------------------------------------------------

NEUTRALS = {"#ffffff", "#111827", "#27272a", "#f4f4f5", "#1c2a44", "#f2ead0", "#ffe9a8"}


def critique_panel(svg: str, layout: dict, bible: dict) -> tuple[float, list[str]]:
    assets = bible["assets"]
    checks: list[tuple[bool, str]] = []
    for cid in layout.get("character_ids", []):
        e = assets.get(cid)
        if not e:
            checks.append((False, f"Character {cid} missing from Bible")); continue
        d = e["dna"]
        for field in ("skin", "hair", "outfit_top", "outfit_bottom"):
            checks.append((str(d.get(field, "")).lower() in svg.lower(),
                           f"{e['name']}: {field} matches Bible"))
        acc = str(d.get("accessory", "none")).lower()
        if acc not in ("", "none"):
            checks.append((f'data-accessory="{acc}"' in svg,
                           f"{e['name']}: accessory '{acc}' present"))
    env = assets.get(layout.get("environment_id") or "")
    if env:
        for field in ("sky", "ground"):
            checks.append((str(env["dna"].get(field, "")).lower() in svg.lower(),
                           f"{env['name']}: {field} matches Bible"))
    for oid in layout.get("object_ids", []):
        e = assets.get(oid)
        if e:
            checks.append((str(e["dna"].get("primary", "")).lower() in svg.lower(),
                           f"{e['name']}: object color matches Bible"))
    # No off-Bible saturated colors: every fill must be Bible/neutral/derived.
    allowed = set(NEUTRALS)
    for e in assets.values():
        allowed |= {str(v).lower() for v in e["dna"].values() if str(v).startswith("#")}
    allowed |= {"#7a4632", "#222222", "#e88a7a", "#c07a4e"}  # derived face details
    allowed |= {str(v).lower() for v in ENV_DEFAULTS.values() if str(v).startswith("#")}
    used = {c.lower() for c in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)}
    off = used - allowed - {str(v).lower() for e in assets.values()
                             for v in [e["dna"].get("eyes", "")]}
    checks.append((not off, "No off-Bible colors" if not off else f"Off-Bible colors: {sorted(off)}"))
    if not checks:
        return 100.0, []
    score = round(100.0 * sum(1 for ok, _ in checks if ok) / len(checks), 1)
    issues = [label for ok, label in checks if not ok]
    return score, issues


async def render_single_graphic(asset_dicts: list[dict], seed: int = 42,
                                pose: str = "stand", expression: str = "calm",
                                shot: str = "medium") -> dict:
    """Render ONE standalone graphic from the frozen Bible.

    This is the default product output: a single image of the chosen
    asset(s), with no premise, story beats, or storyboard. Storyboard
    mode (run_pipeline) reuses the same renderer across many panels.
    Deterministic in (Bible, seed, pose, expression, shot).
    """
    bible, bible_hash = freeze_bible(asset_dicts)
    assets = bible["assets"]
    char_ids = [i for i, e in assets.items() if e["kind"] == "character"]
    env_ids = [i for i, e in assets.items() if e["kind"] == "environment"]
    obj_ids = [i for i, e in assets.items() if e["kind"] == "object"]
    layout = {"character_ids": char_ids, "object_ids": obj_ids,
              "environment_id": env_ids[0] if env_ids else "",
              "camera": {"shot": shot, "angle": "eye-level", "lens": "standard"},
              "pose": pose, "expression": expression, "focus_x": 200,
              "jitter": 0, "narration": ""}
    beat = {"index": 0, "beat_title": "", "action": "", "dialogue": "",
            "narration": ""}
    svg, prompt = render_panel(beat, layout, bible, seed)
    score, issues = critique_panel(svg, layout, bible)
    return {"svg": svg, "prompt": prompt, "consistency_score": score,
            "issues": issues, "bible_hash": bible_hash,
            "output_hash": canonical_hash([svg]), "seed": seed}


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

AGENT_ORDER = ["canon", "story", "layout", "render", "critic"]
AGENT_NAMES = {"canon": "Canon Keeper", "story": "Story Architect",
               "layout": "Layout Director", "render": "Panel Renderer",
               "critic": "Consistency Critic"}


async def run_pipeline(premise: str, panel_count: int, seed: int,
                       asset_dicts: list[dict]) -> dict:
    """Run all five agents. Returns bible, stages, panels (not persisted).

    Canon/Render/Critic are deterministic; Story/Layout come from the Meta
    model when live and from the seeded offline fallback otherwise (each
    stage's detail records which, with the model name).
    """
    bible, bible_hash = freeze_bible(asset_dicts)
    beats, story_meta = await _build_beats_with_meta(premise, panel_count, seed, bible)
    layouts, layout_meta = await _direct_layout_with_meta(beats, seed, bible)

    panels, render_notes = [], []
    for beat, layout in zip(beats, layouts):
        merged = {**beat, "narration": beat.get("narration", "")}
        lay = {**layout, "narration": beat.get("narration", "")}
        svg, prompt = render_panel(merged, lay, bible, seed)
        panels.append({"index": beat["index"], "beat_title": beat["beat_title"],
                       "action": beat["action"], "dialogue": beat["dialogue"],
                       "narration": beat["narration"], "camera": layout["camera"],
                       "layout": {k: v for k, v in layout.items() if k != "camera"},
                       "svg": svg, "prompt": prompt,
                       "consistency_score": 0.0, "issues": [], "repaired": False})
        render_notes.append(f"Panel {beat['index'] + 1}: {layout['camera']['shot']}/{layout['pose']}")

    # No repair re-render: render_panel is deterministic in (seed, beat,
    # layout, bible), so re-calling it with identical inputs returns the
    # identical SVG and score. A failed check therefore stays a reported
    # issue with repaired=False instead of a fake "repaired" pass.
    for panel, layout in zip(panels, layouts):
        score, issues = critique_panel(panel["svg"], layout, bible)
        panel["consistency_score"], panel["issues"] = score, issues

    avg = round(sum(p["consistency_score"] for p in panels) / max(len(panels), 1), 1)
    output_hash = canonical_hash([p["svg"] for p in panels])
    stages = [
        {"agent": "canon", "order": 0, "status": "done",
         "summary": f"Froze {len(bible['assets'])} asset(s) into Bible {short_hash(bible_hash)}.",
         "detail": {"bible_hash": bible_hash,
                    "assets": {e["name"]: e["dna_hash"] for e in bible["assets"].values()}}},
        {"agent": "story", "order": 1, "status": "done",
         "summary": f"Broke the premise into {len(beats)} beat(s) on a {ARC[0][1]}→{ARC[-1][1]} arc.",
         "detail": {**story_meta, "beats": [b["beat_title"] for b in beats],
                    "keywords": _keywords(premise)}},
        {"agent": "layout", "order": 2, "status": "done",
         "summary": "Assigned camera, pose, expression, and blocking for every panel.",
         "detail": {**layout_meta,
                    "shots": [f"{l['camera']['shot']}/{l['camera']['angle']}" for l in layouts]}},
        {"agent": "render", "order": 3, "status": "done",
         "summary": f"Rendered {len(panels)} panel(s) strictly from the Bible (seed {seed}).",
         "detail": {"panels": render_notes}},
        {"agent": "critic", "order": 4, "status": "done",
         "summary": f"Average consistency {avg}/100 across {len(panels)} panel(s); "
                    f"{sum(1 for p in panels if p['issues'])} panel(s) with reported issue(s).",
         "detail": {"scores": [p["consistency_score"] for p in panels],
                    "issues": {str(p['index']): p["issues"] for p in panels if p["issues"]}}},
    ]
    return {"bible": bible, "bible_hash": bible_hash, "output_hash": output_hash,
            "consistency_avg": avg, "stages": stages, "panels": panels}

