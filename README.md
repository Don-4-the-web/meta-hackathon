# WREN Story Engine — no-code private LoRAs for brand mascots

My name is Don Martirez, and for Meta's Global AI Developer Hackathon, I decided to build for my world.

As a visual designer and storyteller, my pain point is simple: I can't produce content fast without losing consistency. Norman the Good Gnome is IP I'm developing with AI. The first graphics took forever — and every new storyline since has meant hours of fighting to match a look and feel I already had. Every iteration drifts a little further from the original.

So I built the app I wished existed: WREN turns anyone's brand mascot into a private, reusable AI model — no code, no GPU of your own. Upload a few reference photos and generate that same character in new scenes, on demand, without drift. This isn't an image analyzer. Analysis is step zero. What WREN manufactures is new model weights: curated datasets, paid GPU fine-tuning, versioned adapters, and finished artwork in three profiles — t-shirt graphics, storyboard panels, and video-ready scenes.

## How it works — three steps

**1 · Add your character to the Asset Bible.** Meta's vision models (`muse-wonder-1.0`) read the reference photo and encode what makes your character yours — colours, build, garments, signature details — into a frozen DNA profile. Before you spend a cent, WREN grades the dataset for free: duplicates, wardrobe clashes, too few poses. Rewrite any caption in plain words, download the `.zip` to see exactly what trains.

**2 · Train.** Not everyone owns expensive GPUs or can code a LoRA pipeline — so the backend does it all: it packages the dataset, spins up a GPU on Modal, and fine-tunes an open image model (SDXL) into your character's personal adapter, locked to a trigger word only you own. Rank 16, UNet + text encoder, capped steps.

**3 · Prove it before serving.** This is the part no one else demos: **WREN refuses to ship a bad adapter.** The moment training finishes, the app auto-renders three probe images (neutral full body, action, close-up) through the fresh model and checks them against the DNA — one figure, full body, correct outfit colours. Pass and the adapter goes live as `ready`. Fail and it parks at `needs_review` with the reasons spelled out; a bad adapter never silently replaces a good one, and `needs_review` adapters are never served to explicit requests.

Only then do you create: pick the character, type an action like "waving hello" or "shoveling dirt", and generate — through your LoRA for series consistency, or flip the adapter off and use Meta's reference model on the same prompt for faithful one-offs.

## Under the hood

A deterministic, seed-locked pipeline: Meta's models for understanding, Modal GPUs for training, open SDXL for generation. Every user is isolated with their own trigger words, adapters, and quotas. The differentiator isn't the stack, it's the loop: grade before you pay, train without code, and prove likeness before serving. No consumer tool validates its own fine-tunes. This one rejects them.

- **Backend** (`backend/app/`): FastAPI + SQLAlchemy + SQLite. Token auth, per-user ownership (legacy rows stay visible), training quotas. Assets CRUD + `/encode`, `/dataset`, `/dataset.zip`, `/grade`, `/captions`, `/train`, `/train/status` (auto-validates on completion), `/validate`, `/render`, projects CRUD + `/run`, `/export` — all under `/api`.
- **Training + inference** (`backend/modal_lora_app.py`, deployed to Modal separately): A10 GPUs, SDXL base, rank-16 LoRA over UNet + text encoder, 400-step cap. Retraining never silently replaces a good adapter — the gate verdict decides.
- **Acceptance gate** (`backend/app/validation.py`): single-subject, full-body-height, outfit-colour-share, and headwear checks over best-of-3 seed retries. Verdict stored on the asset.
- **Render** (`backend/app/imagegen.py`): compact under-77-token LoRA prompts (SDXL truncates past that), negative prompt against bust/crop/sticker collapse, winning seed reported. Segmentation-first cutout (rembg, cleaned + feathered), containment checks, optional phrase band. Profiles: `tshirt` (isolated + margins), `storyboard`, `video` (wide crop).
- **Frontend** (`frontend/`): Next.js 14 + React 18 + Tailwind. Create flow, Asset Bible + LoRA tools (grade, captions editor, steps, strength slider, status, re-validate), login, per-user training quota display.

Request path: browser → FastAPI (`:8000`) → Modal or Meta depending on the step. Grades, captions, and status checks run free locally; GPU time bills to your Modal/Meta accounts.

## 2-minute demo path (rehearse this, never train live on stage)

1. Open `http://localhost:3000`, sign up, open the **Asset Bible**.
2. Create a character, upload 8–12 varied full-body photos (no text in the images, no exact duplicates — the grader flags both for free).
3. Open **LoRA tools**: check the dataset grade, fix any caption, download the `.zip` to see exactly what trains.
4. Train (400 steps, ~$3–4 of Modal credit), **Check status** — the first check that sees a finished job auto-runs the 3-probe gate.
5. **Create**: pick the character, type an action ("waving hello"), generate through the trained LoRA. Flip the adapter off to compare with the Meta reference model on the same prompt.

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

## API map (all under `/api`, Bearer [REDACTED]

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

## Vision

Today you click through it. The vision is fully conversational creation — "put Norman in a pirate comic" and get back a finished, on-model story. Consistent merch, comics, and shorts without an art department.
