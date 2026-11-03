import json

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from . import auth, dataset, models, schemas, training, validation
from .config import settings
from .db import Base, engine, get_db
from .llm import LLMError, stream_chat
from .encode import encode_reference
from .imagegen import generate_graphic, generate_panel_graphic, score_likeness
from .pipeline import canonical_hash, run_pipeline

Base.metadata.create_all(bind=engine)

# SQLite create_all does not add columns to existing tables; the Panel
# image column arrived after the first release, so add it in place.
with engine.begin() as _conn:
    _cols = [r[1] for r in _conn.exec_driver_sql("PRAGMA table_info(panels)")]
    if _cols and "image" not in _cols:
        _conn.exec_driver_sql("ALTER TABLE panels ADD COLUMN image TEXT DEFAULT ''")
    _pcols = [r[1] for r in _conn.exec_driver_sql("PRAGMA table_info(projects)")]
    if _pcols and "profile" not in _pcols:
        _conn.exec_driver_sql(
            "ALTER TABLE projects ADD COLUMN profile TEXT DEFAULT 'storyboard'")
    _acols = [r[1] for r in _conn.exec_driver_sql("PRAGMA table_info(assets)")]
    for _col, _ddl in (("reference_images",
                        "ALTER TABLE assets ADD COLUMN reference_images JSON DEFAULT '[]'"),
                       ("train_status",
                        "ALTER TABLE assets ADD COLUMN train_status TEXT DEFAULT 'none'"),
                       ("adapter_ref",
                        "ALTER TABLE assets ADD COLUMN adapter_ref TEXT DEFAULT ''"),
                       ("train_call_id",
                        "ALTER TABLE assets ADD COLUMN train_call_id TEXT DEFAULT ''"),
                       ("user_id",
                        "ALTER TABLE assets ADD COLUMN user_id TEXT DEFAULT NULL"),
                       ("caption_overrides",
                        "ALTER TABLE assets ADD COLUMN caption_overrides JSON DEFAULT '{}'"),
                       ("validation",
                        "ALTER TABLE assets ADD COLUMN validation JSON DEFAULT '{}'")):
        if _acols and _col not in _acols:
            _conn.exec_driver_sql(_ddl)
    _ccols = [r[1] for r in _conn.exec_driver_sql("PRAGMA table_info(conversations)")]
    if _ccols and "user_id" not in _ccols:
        _conn.exec_driver_sql(
            "ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT NULL")
    _prcols = [r[1] for r in _conn.exec_driver_sql("PRAGMA table_info(projects)")]
    if _prcols and "user_id" not in _prcols:
        _conn.exec_driver_sql(
            "ALTER TABLE projects ADD COLUMN user_id TEXT DEFAULT NULL")

app = FastAPI(title="META AI Chat API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _owned(db: Session, model, oid: str, user: models.User, label: str):
    """Fetch one row visible to this user: theirs, or legacy shared data."""
    obj = db.get(model, oid)
    if not obj or (obj.user_id is not None and obj.user_id != user.id):
        raise HTTPException(404, f"{label} not found")
    return obj


def _visible(db: Session, model, user: models.User):
    return db.query(model).filter(
        (model.user_id == user.id) | (model.user_id.is_(None)))


def _get_conv(db: Session, cid: str, user: models.User) -> models.Conversation:
    return _owned(db, models.Conversation, cid, user, "Conversation")


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.meta_model, "llm_live": settings.llm_live,
            "provider": "meta"}


@app.get("/api/models")
def list_models():
    return {"models": [settings.meta_model], "default": settings.meta_model, "llm_live": settings.llm_live}


# --- Auth ---------------------------------------------------------------------

@app.post("/api/auth/register", response_model=schemas.AuthTokenOut, status_code=201)
def register(body: schemas.RegisterRequest, db: Session = Depends(get_db)):
    username = body.username.strip().lower()
    if not username:
        raise HTTPException(422, "Username is required")
    if db.query(models.User).filter_by(username=username).first():
        raise HTTPException(409, "That username is taken")
    user = models.User(username=username,
                       password_hash=auth.hash_password(body.password),
                       max_trainings=settings.max_trainings_per_user)
    db.add(user); db.flush()
    raw, digest = auth.issue_token()
    user.token_hash = digest
    db.commit()
    return {"token": raw, "user": _user_out(user)}


@app.post("/api/auth/login", response_model=schemas.AuthTokenOut)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(
        username=body.username.strip().lower()).first()
    if not user or not auth.check_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong username or password")
    raw, digest = auth.issue_token()
    user.token_hash = digest
    db.commit()
    return {"token": raw, "user": _user_out(user)}


@app.post("/api/auth/logout", status_code=204)
def logout(db: Session = Depends(get_db),
           user: models.User = Depends(auth.get_current_user)):
    user.token_hash = ""
    db.commit()
    return None


@app.get("/api/auth/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(auth.get_current_user)):
    return _user_out(user)


@app.get("/api/conversations", response_model=list[schemas.ConversationOut])
def list_conversations(db: Session = Depends(get_db),
                       user: models.User = Depends(auth.get_current_user)):
    return _visible(db, models.Conversation, user).order_by(
        models.Conversation.updated_at.desc()).all()


@app.post("/api/conversations", response_model=schemas.ConversationOut, status_code=201)
def create_conversation(body: schemas.ConversationCreate, db: Session = Depends(get_db),
                        user: models.User = Depends(auth.get_current_user)):
    conv = models.Conversation(title=body.title or "New chat",
                               model=body.model or settings.meta_model,
                               user_id=user.id)
    db.add(conv); db.commit(); db.refresh(conv)
    return conv


@app.get("/api/conversations/{cid}", response_model=schemas.ConversationDetail)
def get_conversation(cid: str, db: Session = Depends(get_db),
                     user: models.User = Depends(auth.get_current_user)):
    return _get_conv(db, cid, user)


@app.patch("/api/conversations/{cid}", response_model=schemas.ConversationOut)
def patch_conversation(cid: str, body: schemas.ConversationPatch, db: Session = Depends(get_db),
                       user: models.User = Depends(auth.get_current_user)):
    conv = _get_conv(db, cid, user)
    if body.title is not None:
        conv.title = body.title.strip() or conv.title
    db.commit(); db.refresh(conv)
    return conv


@app.delete("/api/conversations/{cid}", status_code=204)
def delete_conversation(cid: str, db: Session = Depends(get_db),
                        user: models.User = Depends(auth.get_current_user)):
    conv = _get_conv(db, cid, user)
    db.delete(conv); db.commit()
    return None


@app.post("/api/conversations/{cid}/messages")
async def send_message(cid: str, body: schemas.ChatRequest, db: Session = Depends(get_db),
                       user: models.User = Depends(auth.get_current_user)):
    """Persist user message, stream assistant reply as SSE, persist reply."""
    conv = _get_conv(db, cid, user)
    user_msg = models.Message(conversation_id=cid, role="user", content=body.content)
    db.add(user_msg)
    if conv.title == "New chat":
        conv.title = body.content.strip().splitlines()[0][:60]
    db.commit()

    history = [{"role": m.role, "content": m.content} for m in conv.messages if m.role in ("user", "assistant")]
    assistant = models.Message(conversation_id=cid, role="assistant", content="")
    db.add(assistant); db.commit(); db.refresh(assistant)

    # Capture plain values now: the request session closes before the SSE
    # generator runs, so touching ORM attributes inside it raises
    # DetachedInstanceError.
    user_msg_id = user_msg.id
    assistant_msg_id = assistant.id
    chosen_model = body.model or conv.model or settings.meta_model

    async def events():
        from .db import SessionLocal
        full = ""
        yield f"data: {json.dumps({'type': 'meta', 'user_message_id': user_msg_id, 'assistant_message_id': assistant_msg_id, 'model': chosen_model})}\n\n"
        try:
            async for delta in stream_chat(history, chosen_model):
                full += delta
                yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
        except LLMError as e:
            # Never fake a reply: drop the empty placeholder and tell the client.
            with SessionLocal() as s:
                msg = s.get(models.Message, assistant_msg_id)
                if msg and not full:
                    s.delete(msg); s.commit()
                elif msg:
                    msg.content = full; s.commit()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        # persist in a fresh session (request session may close)
        with SessionLocal() as s:
            msg = s.get(models.Message, assistant_msg_id)
            if msg:
                msg.content = full
                s.commit()
        yield f"data: {json.dumps({'type': 'done', 'content': full})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- Asset & Story engine ---------------------------------------------------

def _get_asset(db: Session, aid: str, user: models.User) -> models.Asset:
    return _owned(db, models.Asset, aid, user, "Asset")


def _get_project(db: Session, pid: str, user: models.User) -> models.Project:
    return _owned(db, models.Project, pid, user, "Project")


def _asset_dict(a: models.Asset) -> dict:
    return {"id": a.id, "kind": a.kind, "name": a.name, "description": a.description,
            "dna": a.dna or {}, "version": a.version, "user_id": a.user_id}


def _asset_dict_full(a: models.Asset) -> dict:
    return {**_asset_dict(a), "reference_image": a.reference_image,
            "reference_images": a.reference_images or [],
            "caption_overrides": a.caption_overrides or {}}


def _user_out(user: models.User) -> dict:
    return {"id": user.id, "username": user.username,
            "trainings_used": user.trainings_used,
            "max_trainings": user.max_trainings}


@app.get("/api/assets", response_model=list[schemas.AssetOut])
def list_assets(kind: str | None = None, db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    q = _visible(db, models.Asset, user)
    if kind:
        q = q.filter(models.Asset.kind == kind)
    return q.order_by(models.Asset.created_at).all()


@app.post("/api/assets", response_model=schemas.AssetOut, status_code=201)
def create_asset(body: schemas.AssetCreate, db: Session = Depends(get_db),
                 user: models.User = Depends(auth.get_current_user)):
    images = body.reference_images or ([body.reference_image] if body.reference_image else [])
    asset = models.Asset(kind=body.kind, name=body.name.strip(), description=body.description,
                         dna=body.dna, user_id=user.id,
                         reference_image=images[0] if images else "",
                         reference_images=images)
    db.add(asset); db.commit(); db.refresh(asset)
    return asset


@app.post("/api/assets/encode", response_model=schemas.EncodeOut)
async def encode_asset_reference(body: schemas.EncodeRequest,
                                 user: models.User = Depends(auth.get_current_user)):
    """Read a reference image into suggested DNA. Persists nothing: the
    client pre-fills the form and the user confirms before saving."""
    try:
        return await encode_reference(body.image, body.kind)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/assets/{aid}/dataset")
def get_dataset(aid: str, db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    """Preview the LoRA training set for this character (no image data)."""
    asset = _get_asset(db, aid, user)
    try:
        ds = dataset.build_dataset(_asset_dict_full(asset))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"trigger": ds["trigger"], "count": ds["count"],
            "recommended_minimum": ds["recommended_minimum"],
            "ready": ds["ready"], "train_status": asset.train_status,
            "adapter_ref": asset.adapter_ref,
            "train_call_id": asset.train_call_id,
            "captions": [i["caption"] for i in ds["items"]],
            "fingerprints": [i["fingerprint"] for i in ds["items"]],
            "custom": [i["custom"] for i in ds["items"]]}


@app.get("/api/assets/{aid}/grade")
def grade_asset(aid: str, db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    """Free local pre-flight check: duplicates, size, blank, wardrobe spread."""
    asset = _get_asset(db, aid, user)
    try:
        return dataset.grade_dataset(_asset_dict_full(asset))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.patch("/api/assets/{aid}/captions")
def patch_captions(aid: str, body: schemas.CaptionsPatch, db: Session = Depends(get_db),
                   user: models.User = Depends(auth.get_current_user)):
    """Store per-image caption overrides keyed by image fingerprint.

    Only known fingerprints are kept, so stale edits from removed images
    cannot leak into a future dataset.
    """
    asset = _get_asset(db, aid, user)
    try:
        ds = dataset.build_dataset(_asset_dict_full(asset))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    known = {i["fingerprint"] for i in ds["items"]}
    kept = {fp: str(text)[:500] for fp, text in (body.overrides or {}).items()
            if fp in known and str(text or "").strip()}
    asset.caption_overrides = kept
    db.commit()
    return {"updated": len(kept)}


@app.get("/api/assets/{aid}/dataset.zip")
def download_dataset(aid: str, db: Session = Depends(get_db),
                     user: models.User = Depends(auth.get_current_user)):
    """Download the LoRA training set as a zip (000.png/000.txt pairs).

    This is the local-training path: export from the Bible, train on
    your own GPU, bring the adapter file back here.
    """
    asset = _get_asset(db, aid, user)
    full = _asset_dict_full(asset)
    try:
        blob = dataset.training_zip(full)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    slug = dataset.namespaced_trigger(full).removesuffix("_tok")
    return Response(content=blob, media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{slug}-dataset.zip"'})


@app.post("/api/assets/{aid}/train")
def train_asset(aid: str, body: schemas.TrainRequest | None = None,
                db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    """Submit this character's LoRA train to the configured provider."""
    if user.trainings_used >= user.max_trainings:
        raise HTTPException(402, "Training quota used up "
                                 f"({user.trainings_used}/{user.max_trainings}). "
                                 "Ask the operator to raise your limit.")
    asset = _get_asset(db, aid, user)
    full = _asset_dict_full(asset)
    options = {"steps": body.steps} if body and body.steps else None
    try:
        blob = dataset.training_zip(full)
        result = training.submit_training(full, blob, options)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except training.TrainingNotConfigured as e:
        asset.train_status = "awaiting_modal"
        db.commit()
        return {"status": "awaiting_modal", "message": str(e),
                "dataset_count": len(dataset.asset_images(full))}
    except training.ModalTrainingError as e:
        asset.train_status = "failed"
        db.commit()
        return {"status": "failed", "message": str(e),
                "dataset_count": len(dataset.asset_images(full))}
    asset.train_status = result["status"]
    asset.adapter_ref = result["adapter_ref"]
    asset.train_call_id = result.get("call_id", "")
    user.trainings_used += 1
    db.commit()
    return result


@app.get("/api/assets/{aid}/train/status")
async def train_status(aid: str, db: Session = Depends(get_db),
                       user: models.User = Depends(auth.get_current_user)):
    """Poll an in-flight Modal training job; safe to call repeatedly.

    The first check that sees a finished job runs the likeness acceptance
    gate (three probe renders through the fresh adapter, a few cents of
    Modal GPU) before the adapter can be used: a pass marks it ``ready``,
    anything else parks it at ``needs_review`` with per-probe reasons.
    """
    asset = _get_asset(db, aid, user)
    if asset.train_status != "training":
        return {"status": asset.train_status or "none",
                "adapter_ref": asset.adapter_ref,
                "call_id": asset.train_call_id,
                "validation": asset.validation or {}}
    try:
        result = training.training_status(
            _asset_dict_full(asset), asset.train_call_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except training.TrainingNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except training.ModalTrainingError as e:
        asset.train_status = "failed"
        db.commit()
        return {"status": "failed", "message": str(e),
                "call_id": asset.train_call_id,
                "validation": asset.validation or {}}
    if result["status"] != "ready":
        asset.train_status = result["status"]
        db.commit()
        return {**result, "validation": asset.validation or {}}
    adapter_ref = result.get("adapter_ref", asset.adapter_ref)
    try:
        verdict = await validation.validate_adapter(
            _asset_dict_full(asset), adapter_ref)
    except (training.ModalTrainingError, training.TrainingNotConfigured) as e:
        asset.train_status = "failed"
        db.commit()
        return {"status": "failed",
                "message": f"Validation renders failed: {e}",
                "call_id": asset.train_call_id,
                "validation": asset.validation or {}}
    asset.adapter_ref = adapter_ref
    asset.validation = verdict
    asset.train_status = "ready" if verdict["passed"] else "needs_review"
    db.commit()
    return {"status": asset.train_status, "adapter_ref": asset.adapter_ref,
            "call_id": asset.train_call_id, "validation": verdict}


@app.post("/api/assets/{aid}/validate")
async def validate_asset_adapter(aid: str, db: Session = Depends(get_db),
                                 user: models.User = Depends(auth.get_current_user)):
    """Re-run the likeness acceptance gate on the stored adapter.

    Each run renders three probes through Modal (a few cents of GPU).
    A pass marks the adapter ``ready``; a fail parks it at
    ``needs_review`` with per-probe reasons.
    """
    asset = _get_asset(db, aid, user)
    if not (asset.adapter_ref or "").startswith("modal://"):
        raise HTTPException(409, "No stored Modal adapter to validate yet.")
    try:
        verdict = await validation.validate_adapter(
            _asset_dict_full(asset), asset.adapter_ref)
    except (training.ModalTrainingError, training.TrainingNotConfigured) as e:
        raise HTTPException(502, f"Validation renders failed: {e}") from e
    asset.validation = verdict
    asset.train_status = "ready" if verdict["passed"] else "needs_review"
    db.commit()
    return {"status": asset.train_status, "adapter_ref": asset.adapter_ref,
            "call_id": asset.train_call_id, "validation": verdict}


def _adapter_for_render(asset: models.Asset, requested: bool | None) -> str | None:
    ready = (asset.train_status == "ready" and
             asset.adapter_ref.startswith("modal://"))
    if requested is False:
        return None
    if requested is True:
        if not ready:
            raise HTTPException(
                409, "This character has no ready Modal LoRA yet. Train it, "
                "wait for status 'ready', then retry adapter generation.")
        return asset.adapter_ref
    return asset.adapter_ref if ready else None


@app.post("/api/assets/{aid}/render", response_model=schemas.GraphicOut)
async def render_asset_graphic(aid: str, body: schemas.GraphicRenderRequest,
                               db: Session = Depends(get_db),
                               user: models.User = Depends(auth.get_current_user)):
    """Generate ONE true-likeness graphic of this asset.

    Without a ready LoRA, this calls the Meta image model, conditioned on
    the asset's uploaded reference image when one exists. When the caller
    requests ``adapter=true`` (or leaves it unset after a LoRA is ready),
    generation goes through the stored Modal adapter instead. Image
    generation is non-deterministic; upstream failures surface as 502.
    """
    asset = _get_asset(db, aid, user)
    primary = {**_asset_dict(asset), "reference_image": asset.reference_image}
    extras = [_asset_dict(_get_asset(db, extra_id, user))
              for extra_id in body.include_ids]
    adapter_ref = _adapter_for_render(asset, body.adapter)
    try:
        return await generate_graphic(primary, extras, seed=body.seed,
                                      adapter_ref=adapter_ref,
                                      lora_scale=max(0.1, min(1.0, body.lora_scale or 1.0)),
                                      pose=body.pose,
                                      expression=body.expression,
                                      shot=body.shot, profile=body.profile,
                                      action=body.action,
                                      composition=body.composition,
                                      elements=body.elements,
                                      phrase=body.phrase,
                                      phrase_placement=body.phrase_placement)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except training.ModalTrainingError as e:
        raise HTTPException(502, str(e)) from e
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/assets/{aid}", response_model=schemas.AssetOut)
def get_asset(aid: str, db: Session = Depends(get_db),
              user: models.User = Depends(auth.get_current_user)):
    return _get_asset(db, aid, user)


@app.patch("/api/assets/{aid}", response_model=schemas.AssetOut)
def patch_asset(aid: str, body: schemas.AssetPatch, db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    asset = _get_asset(db, aid, user)
    changed = False
    if body.name is not None and body.name.strip():
        asset.name = body.name.strip(); changed = True
    if body.description is not None:
        asset.description = body.description; changed = True
    if body.dna is not None:
        asset.dna = body.dna; changed = True
    if body.reference_image is not None:
        asset.reference_image = body.reference_image; changed = True
    if body.reference_images is not None:
        asset.reference_images = body.reference_images
        asset.reference_image = body.reference_images[0] if body.reference_images else asset.reference_image
        changed = True
    if changed:
        asset.version += 1  # Bible hashes change; finished boards keep their frozen Bible
    db.commit(); db.refresh(asset)
    return asset


@app.delete("/api/assets/{aid}", status_code=204)
def delete_asset(aid: str, db: Session = Depends(get_db),
                 user: models.User = Depends(auth.get_current_user)):
    db.delete(_get_asset(db, aid, user)); db.commit()
    return None


@app.get("/api/projects", response_model=list[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db),
                  user: models.User = Depends(auth.get_current_user)):
    return _visible(db, models.Project, user).order_by(
        models.Project.updated_at.desc()).all()


@app.post("/api/projects", response_model=schemas.ProjectOut, status_code=201)
def create_project(body: schemas.ProjectCreate, db: Session = Depends(get_db),
                   user: models.User = Depends(auth.get_current_user)):
    for aid in body.asset_ids:
        _get_asset(db, aid, user)
    project = models.Project(title=body.title.strip() or "Untitled story", premise=body.premise,
                             seed=body.seed, panel_count=body.panel_count,
                             asset_ids=body.asset_ids, profile=body.profile,
                             user_id=user.id)
    db.add(project); db.commit(); db.refresh(project)
    return project


@app.get("/api/projects/{pid}", response_model=schemas.ProjectDetail)
def get_project(pid: str, db: Session = Depends(get_db),
                user: models.User = Depends(auth.get_current_user)):
    return _get_project(db, pid, user)


@app.patch("/api/projects/{pid}", response_model=schemas.ProjectOut)
def patch_project(pid: str, body: schemas.ProjectPatch, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.get_current_user)):
    project = _get_project(db, pid, user)
    if body.title is not None and body.title.strip():
        project.title = body.title.strip()
    if body.premise is not None:
        project.premise = body.premise
    if body.seed is not None:
        project.seed = body.seed
    if body.panel_count is not None:
        project.panel_count = body.panel_count
    if body.asset_ids is not None:
        for aid in body.asset_ids:
            _get_asset(db, aid, user)
        project.asset_ids = body.asset_ids
    db.commit(); db.refresh(project)
    return project


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: str, db: Session = Depends(get_db),
                   user: models.User = Depends(auth.get_current_user)):
    db.delete(_get_project(db, pid, user)); db.commit()
    return None


@app.post("/api/projects/{pid}/run", response_model=schemas.RunOut)
async def run_project(pid: str, body: schemas.RunRequest | None = None, db: Session = Depends(get_db),
                       user: models.User = Depends(auth.get_current_user)):
    """Run the 5-agent pipeline (Meta story/layout, deterministic rest)
    and persist panels + stages."""
    body = body or schemas.RunRequest()
    project = _get_project(db, pid, user)
    seed = body.seed if body.seed is not None else project.seed
    panel_count = body.panel_count if body.panel_count is not None else project.panel_count
    run_profile = body.profile or getattr(project, "profile", None) or "storyboard"
    if run_profile not in ("storyboard", "video"):
        run_profile = "storyboard"
    assets = [db.get(models.Asset, aid) for aid in (project.asset_ids or [])]
    assets = [a for a in assets
              if a is not None and (a.user_id is None or a.user_id == user.id)]
    if not any(a.kind == "character" for a in assets):
        raise HTTPException(422, "Add at least one character asset before running the pipeline")
    result = await run_pipeline(project.premise or project.title, panel_count, seed,
                                [_asset_dict(a) for a in assets])

    # Panels are true-likeness images, not shapes: generate each panel
    # with the image model from the lead character's reference, then
    # vision-check it against the frozen Bible. A failed generation keeps
    # the panel with the failure recorded as an issue.
    lead = next(a for a in assets if a.kind == "character")
    primary = {**_asset_dict(lead), "reference_image": lead.reference_image}
    extras = [_asset_dict(a) for a in assets if a.id != lead.id]
    bible_dna = result["bible"]["assets"].get(lead.id, {}).get("dna", {})
    generated = 0
    fingerprints = []
    for p in result["panels"]:
        p["image"] = ""
        try:
            g = await generate_panel_graphic(primary, extras, p,
                                             profile=run_profile)
            p["image"], p["prompt"] = g["image"], g["prompt"]
            p["consistency_score"], p["issues"] = await score_likeness(
                g["image"], bible_dna)
            generated += 1
        except LLMError as e:
            p["issues"] = [*p["issues"], f"Image generation failed: {e}"]
        fingerprints.append(p["image"] or p["svg"])
    if result["panels"]:
        result["consistency_avg"] = round(
            sum(p["consistency_score"] for p in result["panels"])
            / len(result["panels"]), 1)
    result["output_hash"] = canonical_hash(fingerprints)
    for s in result["stages"]:
        if s["agent"] == "render":
            s["summary"] = (f"Generated {generated}/{len(result['panels'])} panel "
                            f"image(s) with {settings.meta_image_model} from the "
                            f"reference image ({run_profile} profile).")
            s["detail"] = {**s["detail"], "model": settings.meta_image_model}
        if s["agent"] == "critic":
            s["summary"] = (f"Average likeness {result['consistency_avg']}/100 "
                            f"across {len(result['panels'])} panel(s), "
                            f"vision-checked against the frozen Bible.")

    run = models.PipelineRun(project_id=pid, seed=seed, status="done",
                             bible=result["bible"], bible_hash=result["bible_hash"],
                             output_hash=result["output_hash"],
                             consistency_avg=result["consistency_avg"])
    db.add(run); db.flush()
    for s in result["stages"]:
        db.add(models.StageRun(run_id=run.id, agent=s["agent"], order=s["order"],
                               status=s["status"], summary=s["summary"], detail=s["detail"]))
    for old in list(project.panels):
        db.delete(old)
    db.flush()
    for p in result["panels"]:
        db.add(models.Panel(project_id=pid, index=p["index"], beat_title=p["beat_title"],
                            action=p["action"], dialogue=p["dialogue"], narration=p["narration"],
                            camera=p["camera"], layout=p["layout"], svg=p["svg"],
                            image=p.get("image", ""),
                            prompt=p["prompt"], consistency_score=p["consistency_score"],
                            issues=p["issues"], repaired=p["repaired"]))
    project.status, project.seed, project.panel_count = "ready", seed, panel_count
    db.commit(); db.refresh(run)
    return run


@app.get("/api/projects/{pid}/export")
def export_project(pid: str, db: Session = Depends(get_db),
                   user: models.User = Depends(auth.get_current_user)):
    """Portable board export: frozen Bible, locked prompts, and panel SVGs."""
    project = _get_project(db, pid, user)
    run = project.runs[-1] if project.runs else None
    return {
        "project": {"id": project.id, "title": project.title, "premise": project.premise,
                    "seed": project.seed, "panel_count": project.panel_count},
        "bible": run.bible if run else {}, "bible_hash": run.bible_hash if run else "",
        "output_hash": run.output_hash if run else "",
        "consistency_avg": run.consistency_avg if run else 0,
        "panels": [{"index": p.index, "beat": p.beat_title, "action": p.action,
                    "dialogue": p.dialogue, "camera": p.camera, "prompt": p.prompt,
                    "consistency_score": p.consistency_score, "svg": p.svg,
                    "image": p.image}
                   for p in project.panels],
    }


DEMO_ASSETS = [
    ("character", "Mira Vale", "Fox-eared sky mechanic who repairs airships.",
     {"form": "fox", "build": "slim", "skin": "#e8b08a", "hair": "#b3502e", "hair_style": "short",
      "eyes": "#20302a", "outfit_top": "#256d5a", "outfit_bottom": "#33404f",
      "primary": "#256d5a", "secondary": "#33404f", "accent": "#d9a03c",
      "accessory": "brass goggles", "marking": "accent stripe"}),
    ("character", "Captain Rho", "Retired airship captain, Mira's mentor.",
     {"form": "human", "build": "broad", "skin": "#8a5a3c", "hair": "#d8d8d8", "hair_style": "short",
      "eyes": "#1c1c1c", "outfit_top": "#5a2e35", "outfit_bottom": "#2c3440",
      "primary": "#5a2e35", "secondary": "#2c3440", "accent": "#c8b06a",
      "accessory": "captain hat", "marking": "none"}),
    ("object", "Storm Compass", "Brass compass that points at coming storms.",
     {"shape": "orb", "material": "brass", "primary": "#b08a3c", "secondary": "#6e5426",
      "accent": "#ffe9a8", "marking": "etched ring", "glow": "soft"}),
    ("environment", "Cloudhaven Docks", "Floating dockyards above a sea of cloud.",
     {"biome": "sky docks", "time": "day", "sky": "#a8c8e0", "ground": "#8a7a5a",
      "primary": "#5a7a9a", "secondary": "#3c4c5e", "accent": "#d9a03c",
      "landmark": "towers", "weather": "clear"}),
]


@app.post("/api/demo", status_code=201)
def seed_demo(db: Session = Depends(get_db),
              user: models.User = Depends(auth.get_current_user)):
    """Create the demo cast + a runnable project (idempotent per user)."""
    ids = []
    for kind, name, desc, dna in DEMO_ASSETS:
        existing = db.query(models.Asset).filter_by(name=name, user_id=user.id).first()
        if existing:
            ids.append(existing.id); continue
        asset = models.Asset(kind=kind, name=name, description=desc, dna=dna,
                             user_id=user.id)
        db.add(asset); db.flush(); ids.append(asset.id)
    title = "Mira and the Storm Compass"
    project = db.query(models.Project).filter_by(title=title, user_id=user.id).first()
    if not project:
        project = models.Project(
            title=title, seed=7, panel_count=6, asset_ids=ids,
            premise=("Mira the sky mechanic finds a storm compass that points at a "
                     "gathering tempest. Captain Rho warns the docks will scatter "
                     "unless the compass reaches the beacon. Mira races the storm "
                     "and lights the beacon, saving Cloudhaven."),
            user_id=user.id)
        db.add(project)
    db.commit(); db.refresh(project)
    return {"project_id": project.id, "asset_ids": ids}
