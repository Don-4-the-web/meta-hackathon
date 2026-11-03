from datetime import datetime

from pydantic import BaseModel, Field


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    model: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


class ConversationCreate(BaseModel):
    title: str = "New chat"
    model: str | None = None


class ConversationPatch(BaseModel):
    title: str | None = None


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=32000)
    model: str | None = None


# --- Asset & Story engine -------------------------------------------------

ASSET_KINDS = ("character", "object", "environment")


class AssetOut(BaseModel):
    id: str
    kind: str
    name: str
    description: str = ""
    dna: dict = {}
    reference_image: str = ""
    reference_images: list = []
    train_status: str = "none"
    adapter_ref: str = ""
    train_call_id: str = ""
    validation: dict = {}
    version: int = 1
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AssetCreate(BaseModel):
    kind: str = Field(pattern="^(character|object|environment)$")
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    dna: dict = {}
    reference_image: str = Field(default="", max_length=2_800_000)
    reference_images: list[str] = Field(default=[], max_length=24)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=4, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str
    trainings_used: int = 0
    max_trainings: int = 5


class AuthTokenOut(BaseModel):
    token: str
    user: UserOut


class TrainRequest(BaseModel):
    steps: int | None = None


class CaptionsPatch(BaseModel):
    overrides: dict[str, str] = {}


class AssetPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    dna: dict | None = None
    reference_image: str | None = Field(default=None, max_length=2_800_000)
    reference_images: list[str] | None = Field(default=None, max_length=24)


class GraphicRenderRequest(BaseModel):
    adapter: bool | None = None  # None = use a ready Modal LoRA when available
    lora_scale: float = 1.0
    seed: int = 42
    pose: str = Field(default="stand", pattern="^(stand|walk|reach|leap)$")
    expression: str = Field(default="calm",
                            pattern="^(calm|curious|surprised|determined|relieved)$")
    shot: str = Field(default="medium", pattern="^(wide|medium|close)$")
    include_ids: list[str] = []  # extra Bible assets (environment/objects)
    profile: str = Field(default="tshirt",
                         pattern="^(tshirt|storyboard|video)$")
    action: str = Field(default="", max_length=500)  # what they are doing
    composition: str = Field(default="vignette",
                             pattern="^(figure|vignette|badge)$")
    elements: str = Field(default="", max_length=200)
    phrase: str = Field(default="", max_length=80)
    phrase_placement: str = Field(default="below",
                                  pattern="^(above|below|banner)$")


class GraphicOut(BaseModel):
    image: str = ""  # data URL from the image model
    prompt: str = ""
    model: str = ""
    source: str = ""  # meta-image-edit (reference) | meta-image-generation
    profile: str = ""
    checks: list = []  # failed format checks (e.g. t-shirt containment)
    bible_hash: str = ""
    output_hash: str = ""
    seed: int = 42


class EncodeRequest(BaseModel):
    kind: str = Field(pattern="^(character|object|environment)$")
    image: str = Field(min_length=1, max_length=2_800_000)


class EncodeOut(BaseModel):
    dna: dict = {}
    description: str = ""
    source: str = ""
    model: str = ""
    fallback: bool = False
    fallback_reason: str = ""
    palette: list = []


class PanelOut(BaseModel):
    id: str
    project_id: str
    index: int
    beat_title: str = ""
    action: str = ""
    dialogue: str = ""
    narration: str = ""
    camera: dict = {}
    layout: dict = {}
    svg: str = ""
    image: str = ""
    prompt: str = ""
    consistency_score: float = 0.0
    issues: list = []
    repaired: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class StageOut(BaseModel):
    id: str
    run_id: str
    agent: str
    order: int
    status: str
    summary: str = ""
    detail: dict = {}
    created_at: datetime

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id: str
    project_id: str
    seed: int
    status: str
    bible: dict = {}
    bible_hash: str = ""
    output_hash: str = ""
    consistency_avg: float = 0.0
    created_at: datetime
    stages: list[StageOut] = []

    model_config = {"from_attributes": True}


class ProjectOut(BaseModel):
    id: str
    title: str
    premise: str = ""
    seed: int = 42
    panel_count: int = 1
    asset_ids: list = []
    profile: str = "storyboard"
    status: str = "draft"
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectDetail(ProjectOut):
    panels: list[PanelOut] = []
    runs: list[RunOut] = []


class ProjectCreate(BaseModel):
    title: str = Field(default="Untitled story", max_length=160)
    premise: str = Field(default="", max_length=4000)
    seed: int = 42
    panel_count: int = Field(default=1, ge=1, le=24)
    asset_ids: list[str] = []
    profile: str = Field(default="storyboard",
                         pattern="^(tshirt|storyboard|video)$")


class ProjectPatch(BaseModel):
    title: str | None = None
    premise: str | None = None
    seed: int | None = None
    panel_count: int | None = Field(default=None, ge=1, le=24)
    asset_ids: list[str] | None = None


class RunRequest(BaseModel):
    seed: int | None = None
    panel_count: int | None = Field(default=None, ge=1, le=24)
    profile: str | None = Field(default=None,
                                pattern="^(storyboard|video)$")
