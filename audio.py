from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from database import Database
from models import AudioPreference, utc_now


BUILT_INS = [
    ("rain_01", "雨天小镇与鸟鸣", "雨声", "rainy-day-in-town-with-birds-singing.mp3", 581),
    ("fan_01", "强降雨声", "雨声", "strong-rain.mp3", 40),
    ("ocean_01", "轻触水面", "水声", "touching-the-water.mp3", 18),
]


class AudioService:
    def __init__(self, database: Database, audio_dir: str | Path, user_audio_dir: str | Path):
        self.database = database
        self.audio_dir = Path(audio_dir)
        self.user_audio_dir = Path(user_audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.user_audio_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_built_ins()

    def ensure_built_ins(self) -> None:
        for audio_id, title, category, file_name, duration_seconds in BUILT_INS:
            file_path = self.audio_dir / file_name
            if not file_path.exists():
                continue
            self.database.save_audio_asset({
                "audio_id": audio_id, "title": title, "category": category, "file_name": file_name,
                "owner_type": "developer", "duration_seconds": duration_seconds, "loopable": True,
                "source": "built-in", "created_at": utc_now(),
            })

    def catalog(self, user_id: str) -> list[dict[str, Any]]:
        return self.database.get_audio_assets(user_id)

    def preference(self, user_id: str) -> AudioPreference:
        assets = self.database.get_audio_assets(user_id)
        available_ids = {asset["audio_id"] for asset in assets}
        default_audio_id = BUILT_INS[0][0]
        fallback = default_audio_id if default_audio_id in available_ids else (
            assets[0]["audio_id"] if assets else default_audio_id
        )
        return self.database.get_audio_preference(user_id, fallback)

    def save_preference(self, preference: AudioPreference) -> None:
        self.database.save_audio_preference(preference)

    def path_for(self, asset: dict[str, Any]) -> Path:
        base_dir = self.user_audio_dir if asset["owner_type"] == "user" else self.audio_dir
        base_dir = base_dir.resolve()
        candidate = (base_dir / asset["file_name"]).resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError as exc:
            raise ValueError("音频文件路径不在允许的音频目录内") from exc
        return candidate

    def upload(self, user_id: str, file_name: str, data: bytes) -> dict[str, Any]:
        safe_name = Path(str(file_name).replace("\\", "/")).name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or safe_name in {".", ".."} or extension not in {".wav", ".mp3", ".ogg", ".m4a"}:
            raise ValueError("只支持 WAV、MP3、OGG 或 M4A 音频")
        if any(ord(character) < 32 for character in safe_name):
            raise ValueError("音频文件名包含不支持的字符")
        if len(data) > 20 * 1024 * 1024:
            raise ValueError("单个音频文件不能超过 20MB")
        digest = hashlib.sha256(f"{user_id}\0{safe_name}".encode("utf-8")).hexdigest()[:24]
        audio_id = f"user_{digest}"
        stored_name = f"{audio_id}{extension}"
        target = self.path_for({"owner_type": "user", "file_name": stored_name})
        target.write_bytes(data)
        asset = {
            "audio_id": audio_id, "title": Path(safe_name).stem[:40], "category": "我的音频",
            "file_name": stored_name, "owner_type": "user", "owner_id": user_id,
            "duration_seconds": 30, "loopable": True, "source": "user-upload", "created_at": utc_now(),
        }
        self.database.save_audio_asset(asset)
        return asset
