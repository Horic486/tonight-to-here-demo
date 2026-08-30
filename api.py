from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from audio import AudioService
from config import AUDIO_DIR, DB_PATH, USER_AUDIO_DIR, ensure_directories
from database import Database
from models import AudioPreference


ensure_directories()
database = Database(DB_PATH)
audio = AudioService(database, AUDIO_DIR, USER_AUDIO_DIR)
app = FastAPI(title="今晚到此 API", version="0.1.0")


class PreferencePayload(BaseModel):
    default_audio_id: str
    volume: float = Field(default=0.18, ge=0, le=1)
    autoplay_enabled: bool = True
    fade_out_minutes: int = Field(default=20, ge=0, le=120)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/audio")
def list_audio(user_id: str) -> list[dict]:
    database.ensure_user(user_id)
    return audio.catalog(user_id)


@app.get("/users/{user_id}/audio-preference")
def get_preference(user_id: str) -> AudioPreference:
    database.ensure_user(user_id)
    return audio.preference(user_id)


@app.put("/users/{user_id}/audio-preference")
def update_preference(user_id: str, payload: PreferencePayload) -> AudioPreference:
    database.ensure_user(user_id)
    ids = {asset["audio_id"] for asset in audio.catalog(user_id)}
    if payload.default_audio_id not in ids:
        raise HTTPException(status_code=400, detail="音频不存在或不属于当前用户")
    preference = AudioPreference(user_id=user_id, **payload.model_dump())
    audio.save_preference(preference)
    return preference


@app.post("/audio/uploads")
async def upload_audio(user_id: str, file: UploadFile = File(...)) -> dict:
    try:
        database.ensure_user(user_id)
        return audio.upload(user_id, file.filename or "my-audio.wav", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
