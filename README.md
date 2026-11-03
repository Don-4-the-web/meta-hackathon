# META Story Engine — no-code private LoRAs for brand mascots

Turn a handful of mascot photos into a reusable character model that draws
your figure in new scenes without drifting — no code, no GPU of your own.
Built for the META hackathon on **Muse Spark** (`muse-wonder-1.0`) +
**Modal** GPUs + open **SDXL**.

## The pitch

Reference-based generators (Meta, Google) nail likeness because they see
your photo every render — but every new scene needs the photo re-attached
and the composition fights you. A personal LoRA bakes the character into
weights you can reuse anywhere, yet small-data LoRAs notoriously collapse
(busts instead of bodies, scrambled outfit colours, duplicated figures)
and nothing tells you before you waste renders on them.

This app is the whole loop, plus the missing piece — **a likeness
acceptance gate**: every fresh adapter must pass three probe renders
(neutral full body, action, close-up: one figure, body visible, DNA outfit
colours present) before it is marked `ready`. A bad adapter parks at
`needs_review` with per-probe reasons instead of silently replacing a good
one. Reference model for faithful singles, LoRA for character series.

## 2-minute demo path (rehearse this, never train live on stage)

1. Open `http://localhost:3000`, sign up, open the **Asset Bible**.
2. Create a character, upload 8–12 varied full-body photos (no text in the
   images, no exact duplicates — the grader flags both for free).
3. Open **LoRA tools**: check the dataset grade, fix any caption, download
   the `.zip` to see exactly what trains.
4. Train (400 steps, ~$3–4 of Modal credit), **Check status** — the first
   check that sees a finished job auto-runs the 3-probe gate.
5. **Create flow**: pick the character, type an action ("waving hello"),
   generate through the trained LoRA. Flip the adapter off to compare with
   the Meta reference model on the same prompt.

## How it works

- **Asset Bible** (`backend/app/`): per-user characters/objects with a DNA
  profile (colours, build, garments) read from a reference photo by
  `muse-wonder-1.0`. Users own their mascots; legacy rows stay visible.
- **Free pre-flight**: `GET /grade` (duplicates, wardrobe spread, minimum
  count), per-image caption overrides by fingerprint, downloadable dataset.
- **Training** (`backend/modal_lora_app.py` on Modal A10): SDXL LoRA, rank
  16, UNet + text encoder, trigger token per user+name, capped steps.
- **Acceptance gate** (`backend/app/validation.py`): probes + single-body,
  full-body-height, outfit-colour-share and headwear checks. Verdict stored
  on the asset; `needs_review` adapters are never served to explicit
  requests.
- **Render** (`backend/app/imagegen.py`): compact under-77-token LoRA
  prompts (SDXL truncates past that), negative prompt against bust/crop/
  sticker collapse, best-of-3 seed retry through the gate checks, winning
  seed reported. Segmentation-first cutout (rembg, cleaned + feathered),
  containment checks, optional phrase band. Profiles: `tshirt`
  (isolated + margins), `storyboard`, `video` (wide crop).
- **Frontend** (`frontend/`): Create flow, Asset Bible + LoRA tools
  (grade, captions editor, steps, strength slider, status, re-validate),
  login, per-user training quota display.

## Quickstart

Prereqs: Python 3.12 with the repo venv, Node 18+, a Meta API key, a Modal
account (`python -m modal setup`) with ~$30 free credit.

```bash
# backend (http://localhost:8000)
cd backend
cp .env.example .env   # set META_API_KEY
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# deploy training app once (and after editing modal_lora_app.py)
.\.venv\Scripts\modal.exe deploy modal_lora_app.py

# frontend (http://localhost:3000)
cd frontend
npm install; npm run build; npm run start -- -p 3000
```

Key settings (`backend/app/config.py`, overridable in `.env`):
`META_API_KEY`, `META_API_BASE_URL`, `max_trainings_per_user=5`,
`modal_max_train_steps=400`, `DATABASE_URL=sqlite:///./chat.db`,
`NEXT_PUBLIC_API_URL=http://localhost:8000` (frontend `.env.local`).

## Costs (approx, paid to Modal/Meta, not us)

- Train: 400 steps ≈ **$3–4**; quota 5 trainings/user.
- Gate + re-validate: 3 probe inferences, **cents**.
- Render: 1–3 LoRA inferences (**cents**); Meta reference renders bill to
  your Meta key. Dataset grade, captions, status checks: **free**.

## API map (all under `/api`, bearer auth)

`POST /auth/register|login`, `GET /auth/me` · assets CRUD + `/encode`,
`/dataset`, `/dataset.zip`, `/grade`, `/captions`, `/train`,
`/train/status` (auto-validates on completion), `/validate`, `/render`
(`profile`, `action`, `adapter`, `lora_scale`, `steps`, `pose`,
`expression`, `shot`, `phrase`) · projects CRUD + `/run`, `/export`.

## Tests

```bash
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests   # 86 tests
```

Covers prompts, isolation masks, feathering, seed retry, validation
probes, auth/ownership/quotas, Modal app config, and the dataset grader —
all with faked GPU/vision seams; no spend.

## Repo layout

```
backend/
  app/            FastAPI: auth, assets, training, imagegen, validation, ...
  modal_lora_app.py   Modal training + inference (deploy separately)
  tests/          86-test suite
frontend/
  app/            Create flow, Asset Bible, login, chat
  components/     TrainPanel (grade/captions/train/validate), UserBadge
  lib/api.ts      typed client with login-token handling
```
