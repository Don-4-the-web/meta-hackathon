import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String, default="")
    token_hash: Mapped[str] = mapped_column(String, default="", index=True)
    trainings_used: Mapped[int] = mapped_column(Integer, default=0)
    max_trainings: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id"), default=None, index=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    model: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Asset(Base):
    """A reusable bible asset: character, object, or environment.

    `dna` is the structured source of truth. Once an asset is used in a
    pipeline run, the Canon Keeper freezes a hash of it into the run's
    Asset Bible, so later edits never silently change finished boards.
    """

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String, index=True)  # character | object | environment
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, default="")
    dna: Mapped[dict] = mapped_column(JSON, default=dict)
    user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id"), default=None, index=True)
    reference_image: Mapped[str] = mapped_column(Text, default="")  # data URL, optional
    reference_images: Mapped[list] = mapped_column(JSON, default=list)
    caption_overrides: Mapped[dict] = mapped_column(JSON, default=dict)
    train_status: Mapped[str] = mapped_column(String, default="none")
    adapter_ref: Mapped[str] = mapped_column(String, default="")
    train_call_id: Mapped[str] = mapped_column(String, default="")
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id"), default=None, index=True)
    title: Mapped[str] = mapped_column(String, default="Untitled story")
    premise: Mapped[str] = mapped_column(Text, default="")
    seed: Mapped[int] = mapped_column(Integer, default=42)
    panel_count: Mapped[int] = mapped_column(Integer, default=1)
    asset_ids: Mapped[list] = mapped_column(JSON, default=list)
    profile: Mapped[str] = mapped_column(String, default="storyboard")
    status: Mapped[str] = mapped_column(String, default="draft")  # draft | ready | failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    panels: Mapped[list["Panel"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Panel.index"
    )
    runs: Mapped[list["PipelineRun"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="PipelineRun.created_at"
    )


class Panel(Base):
    __tablename__ = "panels"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    beat_title: Mapped[str] = mapped_column(String, default="")
    action: Mapped[str] = mapped_column(Text, default="")
    dialogue: Mapped[str] = mapped_column(Text, default="")
    narration: Mapped[str] = mapped_column(Text, default="")
    camera: Mapped[dict] = mapped_column(JSON, default=dict)
    layout: Mapped[dict] = mapped_column(JSON, default=dict)
    svg: Mapped[str] = mapped_column(Text, default="")
    image: Mapped[str] = mapped_column(Text, default="")  # image-model data URL
    prompt: Mapped[str] = mapped_column(Text, default="")
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    issues: Mapped[list] = mapped_column(JSON, default=list)
    repaired: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="panels")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String, default="running")  # running | done | failed
    bible: Mapped[dict] = mapped_column(JSON, default=dict)
    bible_hash: Mapped[str] = mapped_column(String, default="")
    output_hash: Mapped[str] = mapped_column(String, default="")
    consistency_avg: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="runs")
    stages: Mapped[list["StageRun"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="StageRun.order"
    )


class StageRun(Base):
    __tablename__ = "stage_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True)
    agent: Mapped[str] = mapped_column(String)  # canon | story | layout | render | critic
    order: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="done")
    summary: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[PipelineRun] = relationship(back_populates="stages")
