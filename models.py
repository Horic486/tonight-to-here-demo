from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


DEFAULT_AUDIO_VOLUME = 0.18
LEGACY_DEFAULT_AUDIO_VOLUME = 0.35


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TodoItem(BaseModel):
    content: str
    category: Literal["task", "worry", "entertainment", "other"] = "other"
    suggested_slot: Literal["tonight", "tomorrow", "later"] = "tomorrow"
    minimum_action: str = ""


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str = Field(min_length=1)
    kind: Literal[
        "preference",
        "routine",
        "pattern",
        "context",
        "trigger",
        "helpful_action",
        "constraint",
    ]
    content: str
    source_type: Literal["session", "user_statement", "system_inference"] = "system_inference"
    source_session_id: Optional[str] = None
    source_ref: Optional[str] = None
    confidence: float = Field(default=0.7, ge=0, le=1)
    importance: float = Field(default=0.6, ge=0, le=1)
    effective_score: float = Field(default=0.0, ge=0)
    consent: bool = True
    status: Literal["active", "superseded", "expired", "revoked"] = "active"
    valid_from: str = Field(default_factory=utc_now)
    valid_until: Optional[str] = None
    half_life_days: float = Field(default=90.0, gt=0)
    evidence_count: int = Field(default=1, ge=1)
    memory_key: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    last_used_at: Optional[str] = None


class UserProfileFact(BaseModel):
    profile_id: str
    user_id: str = Field(min_length=1)
    profile_key: str = Field(min_length=1)
    profile_value: str = Field(min_length=1)
    source_type: Literal["user_statement", "observed_pattern", "system_inference"] = "observed_pattern"
    confidence: float = Field(default=0.7, ge=0, le=1)
    source_memory_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=1, ge=1)
    status: Literal["active", "superseded", "expired", "revoked"] = "active"
    valid_until: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class GuidanceChunk(BaseModel):
    chunk_id: str
    title: str
    content: str
    source: str


class AudioAsset(BaseModel):
    audio_id: str
    title: str
    category: str
    file_name: str
    owner_type: Literal["developer", "user"] = "developer"
    owner_id: Optional[str] = None
    duration_seconds: int = 30
    loopable: bool = True
    source: str = "built-in"


class AudioPreference(BaseModel):
    user_id: str
    default_audio_id: str
    volume: float = Field(default=DEFAULT_AUDIO_VOLUME, ge=0, le=1)
    autoplay_enabled: bool = True
    fade_out_minutes: int = Field(default=20, ge=0, le=120)


class ContextBundle(BaseModel):
    current_stage: str
    today_input: str
    short_term_summary: str = ""
    recent_turns: list[str] = Field(default_factory=list)
    long_term_memories: list[str] = Field(default_factory=list)
    user_profile: dict[str, list[str]] = Field(default_factory=dict)
    retrieved_guidance: list[str] = Field(default_factory=list)
    user_preferences: dict = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    state: str
    items: list[TodoItem] = Field(default_factory=list)
    message: str = ""
    action_title: str = ""
    action_steps: list[str] = Field(default_factory=list)
    tomorrow_card: str = ""
    audio_id: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    round_index: int = 0


class ConversationRound(BaseModel):
    round_id: str
    session_id: str
    user_id: str = Field(min_length=1)
    local_date: str
    round_index: int = Field(ge=1)
    started_at: str
    completed_at: Optional[str] = None
    status: Literal["active", "completed", "abandoned"] = "active"
    initial_feeling: str = ""
    concern_input: str = ""
    items: list[TodoItem] = Field(default_factory=list)
    arrangements: list[dict[str, str]] = Field(default_factory=list)
    tomorrow_card: str = ""
    wind_down_advice: dict[str, Any] = Field(default_factory=dict)
    followup_feedback: list[dict[str, Any]] = Field(default_factory=list)
    followup_advice: list[dict[str, Any]] = Field(default_factory=list)
    tonight_action: dict[str, Any] = Field(default_factory=dict)
    closure_message: str = ""
    created_at: str
    updated_at: str
