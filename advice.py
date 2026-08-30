from __future__ import annotations

from context import ContextAssembler
from llm import LLMClient
from models import AudioPreference, GuidanceChunk, WorkflowResult
from rag import GuidanceRAG
from search import WebSearchClient, reciprocal_rank_fusion


class AdviceService:
    """Local-first advice pipeline with optional web retrieval and deterministic fallback."""

    def __init__(self, context: ContextAssembler, rag: GuidanceRAG, llm: LLMClient, web_search: WebSearchClient):
        self.context = context
        self.rag = rag
        self.llm = llm
        self.web_search = web_search

    def generate(
        self,
        user_id: str,
        session_id: str,
        preference: AudioPreference,
        feedback: str,
        allow_web: bool = False,
    ) -> WorkflowResult:
        local_chunks = self.rag.retrieve(feedback, top_k=4)
        web_chunks: list[GuidanceChunk] = []
        if allow_web:
            try:
                web_chunks = self.web_search.search(feedback, top_k=4)
            except Exception:
                web_chunks = []
        merged = reciprocal_rank_fusion(local_chunks, web_chunks, top_k=5)
        bundle = self.context.build(user_id, session_id, "SLEEP_FOLLOWUP", feedback, preference)
        bundle.retrieved_guidance = [
            f"{chunk.title}：{chunk.content}（来源：{chunk.source}）" for chunk in merged
        ]
        result = self.llm.generate_followup(bundle, feedback)
        result.sources = [chunk.source for chunk in merged]
        result.fallback_used = result.fallback_used or not bool(merged)
        return result

