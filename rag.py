from __future__ import annotations

import re
from pathlib import Path

from models import GuidanceChunk
from vector_store import LocalVectorStore


class GuidanceRAG:
    """Controlled, one-shot retrieval over a small reviewed guidance corpus."""

    def __init__(self, knowledge_dir: str | Path, vectors: LocalVectorStore):
        self.knowledge_dir = Path(knowledge_dir)
        self.vectors = vectors
        self.chunks: list[GuidanceChunk] = []
        self._load()

    def _load(self) -> None:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        files = list(self.knowledge_dir.glob("*.md"))
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
            for index, section in enumerate(sections):
                lines = [line.strip() for line in section.splitlines() if line.strip()]
                if len(lines) < 2:
                    continue
                title = lines[0]
                content = " ".join(lines[1:])
                chunk = GuidanceChunk(
                    chunk_id=f"{file_path.stem}-{index}", title=title, content=content, source=file_path.name
                )
                self.chunks.append(chunk)
                self.vectors.upsert(
                    chunk.chunk_id,
                    "guidance",
                    f"{chunk.title} {chunk.content}",
                    {"source": chunk.source},
                )

    def retrieve(self, query: str, top_k: int = 2) -> list[GuidanceChunk]:
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        records = self.vectors.search(query, "guidance", top_k=top_k)
        return [by_id[record["record_id"]] for record in records if record["record_id"] in by_id]

