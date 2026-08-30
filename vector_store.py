from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_SAVE_LOCK = threading.RLock()


def tokens(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text.lower())
    joined = "".join(raw)
    bigrams = [joined[index:index + 2] for index in range(max(0, len(joined) - 1))]
    return raw + bigrams


def embed(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LocalVectorStore:
    """Small persistent vector-store adapter for the runnable local demo."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.records = []

    def _save(self) -> None:
        payload = json.dumps(self.records, ensure_ascii=False, indent=2)
        with _SAVE_LOCK:
            try:
                temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(self.path)
            except (OSError, PermissionError):
                # The demo can still retrieve from memory on a read-only or locked filesystem.
                # A production deployment should emit this event to its observability system.
                return

    def upsert(self, record_id: str, namespace: str, text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self.records = [record for record in self.records if record["record_id"] != record_id]
        self.records.append({
            "record_id": record_id,
            "namespace": namespace,
            "text": text,
            "metadata": metadata or {},
            "vector": embed(text),
        })
        self._save()

    def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None,
        record_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        query_vector = embed(query)
        candidates = []
        for record in self.records:
            if record["namespace"] != namespace:
                continue
            if record_ids is not None and record["record_id"] not in record_ids:
                continue
            if metadata_filter and any(record["metadata"].get(k) != v for k, v in metadata_filter.items()):
                continue
            score = cosine(query_vector, record["vector"])
            candidates.append({**record, "score": score})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]
