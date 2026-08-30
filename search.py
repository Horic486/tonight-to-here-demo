from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from models import GuidanceChunk


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._field: str | None = None

    def _flush(self) -> None:
        if self._current and self._current.get("title"):
            self.results.append(self._current)
        self._current = None
        self._field = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current = {"title": "", "url": "", "snippet": ""}
            self._field = "title"
            href = dict(attrs).get("href") or ""
            self._current["url"] = href
        elif self._current and ("result__snippet" in classes or "result-snippet" in classes):
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current and self._field == "title":
            self._field = None
        elif self._current and self._field == "snippet":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._current and self._field:
            self._current[self._field] += data

    def close(self) -> None:
        super().close()
        self._flush()


def clean_text(value: str, limit: int = 600) -> str:
    value = html.unescape(re.sub(r"\s+", " ", value)).strip()
    return value[:limit]


def split_chunks(text: str, max_chars: int = 320) -> list[str]:
    text = clean_text(text, limit=1600)
    if len(text) <= max_chars:
        return [text] if text else []
    return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]


def sanitize_query(text: str) -> str:
    """Keep external search opt-in and avoid sending the raw personal narrative."""
    text = re.sub(r"(我|现在|今天|刚刚|有点|感觉|觉得|特别)", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", text)
    return clean_text(text, 80) or "睡前放松 安静入睡 建议"


class WebSearchClient:
    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, top_k: int = 4) -> list[GuidanceChunk]:
        safe_query = sanitize_query(query)
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": safe_query})
        request = urllib.request.Request(url, headers={"User-Agent": "TonightToHere/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parser = _DuckDuckGoParser()
        parser.feed(body)
        parser.close()
        chunks: list[GuidanceChunk] = []
        for index, item in enumerate(parser.results[:top_k]):
            title = clean_text(item.get("title", ""), 120)
            snippet = clean_text(item.get("snippet", ""), 600)
            if not title or not snippet:
                continue
            url = clean_text(item.get("url", ""), 400)
            for chunk_index, content in enumerate(split_chunks(snippet)):
                chunks.append(GuidanceChunk(
                    chunk_id=f"web-{index}-{chunk_index}",
                    title=title,
                    content=content,
                    source=url or "DuckDuckGo 摘要",
                ))
        return chunks


def reciprocal_rank_fusion(
    local_chunks: list[GuidanceChunk],
    web_chunks: list[GuidanceChunk],
    top_k: int = 5,
    constant: int = 60,
) -> list[GuidanceChunk]:
    scores: dict[str, float] = {}
    records: dict[str, GuidanceChunk] = {}
    for rank, chunk in enumerate(local_chunks, start=1):
        key = clean_text(f"{chunk.title} {chunk.content}", 180).lower()
        records.setdefault(key, chunk)
        scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    for rank, chunk in enumerate(web_chunks, start=1):
        key = clean_text(f"{chunk.title} {chunk.content}", 180).lower()
        records.setdefault(key, chunk)
        scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    ordered = sorted(records, key=lambda key: scores[key], reverse=True)
    return [records[key] for key in ordered[:top_k]]
