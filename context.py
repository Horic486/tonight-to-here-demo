from __future__ import annotations

from database import Database
from memory import MemoryService
from models import AudioPreference, ContextBundle
from rag import GuidanceRAG


class ContextAssembler:
    def __init__(self, database: Database, memories: MemoryService, rag: GuidanceRAG):
        self.database = database
        self.memories = memories
        self.rag = rag

    def build(self, user_id: str, session_id: str, stage: str, today_input: str, preference: AudioPreference) -> ContextBundle:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")
        summary, turns = self.memories.short_term_context(user_id, session_id)
        long_term = self.memories.retrieve_long_term(user_id, today_input, top_k=3)
        user_profile = self.memories.profile_context(user_id)
        guidance = self.rag.retrieve(today_input, top_k=2)
        return ContextBundle(
            current_stage=stage,
            today_input=today_input,
            short_term_summary=summary,
            recent_turns=turns,
            long_term_memories=long_term,
            user_profile=user_profile,
            retrieved_guidance=[f"{item.title}：{item.content}（来源：{item.source}）" for item in guidance],
            user_preferences={
                "default_audio_id": preference.default_audio_id,
                "volume": preference.volume,
                "fade_out_minutes": preference.fade_out_minutes,
            },
        )
