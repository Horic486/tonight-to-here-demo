from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _load_env_file() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
_configured_runtime = os.getenv("TONIGHT_RUNTIME_DIR")
if _configured_runtime:
    RUNTIME_DIR = Path(_configured_runtime)
else:
    _runtime_root = os.getenv("LOCALAPPDATA") or str(DATA_DIR / ".runtime")
    RUNTIME_DIR = Path(_runtime_root) / "TonightToHere"
USER_AUDIO_DIR = RUNTIME_DIR / "audio" / "user"
DB_PATH = RUNTIME_DIR / "tonight_to_here.sqlite3"
VECTOR_PATH = RUNTIME_DIR / "vectors.json"


def ensure_directories() -> None:
    for path in (DATA_DIR, AUDIO_DIR, KNOWLEDGE_DIR, RUNTIME_DIR, USER_AUDIO_DIR):
        path.mkdir(parents=True, exist_ok=True)


def model_mode() -> str:
    return os.getenv("MODEL_MODE", "mock").strip().lower()
