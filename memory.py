from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from database import Database
from models import MemoryItem, UserProfileFact
from vector_store import LocalVectorStore


MEMORY_POLICIES = {
    "preference": {"valid_days": None, "half_life_days": 365.0, "importance": 0.80, "confidence": 0.72},
    "routine": {"valid_days": 180, "half_life_days": 120.0, "importance": 0.68, "confidence": 0.68},
    "pattern": {"valid_days": 120, "half_life_days": 60.0, "importance": 0.70, "confidence": 0.72},
    "context": {"valid_days": 21, "half_life_days": 10.0, "importance": 0.58, "confidence": 0.78},
    "trigger": {"valid_days": 120, "half_life_days": 60.0, "importance": 0.76, "confidence": 0.76},
    "helpful_action": {"valid_days": 180, "half_life_days": 120.0, "importance": 0.86, "confidence": 0.78},
    "constraint": {"valid_days": None, "half_life_days": 365.0, "importance": 0.92, "confidence": 0.82},
}

PROFILE_MIN_CONFIDENCE = 0.65
PROFILE_MAX_FACTS = 8
PROFILE_MAX_CHARS = 500
SINGULAR_PROFILE_KEYS = {"sound_preference", "advice_style"}
SECRET_TERMS = ("api key", "apikey", "密码", "密钥", "access token", "secret")
SENSITIVE_PROFILE_TERMS = ("政治立场", "性取向", "宗教信仰", "疾病诊断")


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class MemoryService:
    def __init__(
        self,
        database: Database,
        vectors: LocalVectorStore,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.database = database
        self.vectors = vectors
        self.clock = clock or _utc_clock

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")
        return user_id.strip()

    def short_term_context(self, user_id: str, session_id: str) -> tuple[str, list[str]]:
        user_id = self._require_user_id(user_id)
        summaries = self.database.recent_summaries(user_id, limit=3)
        turns = self.database.recent_turns_for_user(user_id, session_id, limit=8)
        summary = "\n".join(f"最近会话摘要：{item}" for item in summaries)
        return summary, turns

    def effective_score(
        self,
        memory: MemoryItem,
        relevance_score: float,
        now: Optional[datetime] = None,
    ) -> float:
        current = now or self._now()
        confirmed_at = _parse_time(memory.updated_at or memory.created_at)
        age_days = max(0.0, (current - confirmed_at).total_seconds() / 86400)
        time_decay = 0.5 ** (age_days / memory.half_life_days)
        relevance_boost = 0.75 + 0.5 * max(0.0, min(1.0, relevance_score))
        return memory.importance * memory.confidence * time_decay * relevance_boost

    def rank_long_term(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        now = self._now()
        now_text = now.isoformat()
        self.database.expire_memories(user_id, now_text)
        active = self.database.list_memories(user_id, now=now_text)
        if not active:
            return []
        by_id = {memory.memory_id: memory for memory in active}
        records = self.vectors.search(
            query,
            "memory",
            top_k=len(active),
            metadata_filter={"user_id": user_id},
            record_ids=set(by_id),
        )
        ranked: list[MemoryItem] = []
        for record in records:
            relevance = float(record["score"])
            if relevance <= 0.05:
                continue
            memory = by_id[record["record_id"]]
            score = self.effective_score(memory, relevance, now)
            if score < 0.05:
                continue
            ranked.append(memory.model_copy(update={"effective_score": score}))
        ranked.sort(key=lambda item: item.effective_score, reverse=True)
        selected = ranked[:top_k]
        for memory in selected:
            self.database.touch_memory(user_id, memory.memory_id, now_text)
        return selected

    def retrieve_long_term(self, user_id: str, query: str, top_k: int = 3) -> list[str]:
        return [memory.content for memory in self.rank_long_term(user_id, query, top_k)]

    def save_memory(self, memory: MemoryItem) -> None:
        self._require_user_id(memory.user_id)
        self.database.save_memory(memory)
        self.vectors.upsert(
            memory.memory_id,
            "memory",
            memory.content,
            {"user_id": memory.user_id, "kind": memory.kind, "status": memory.status},
        )

    def record_evidence(
        self,
        user_id: str,
        kind: str,
        content: str,
        *,
        source_type: str = "system_inference",
        source_session_id: Optional[str] = None,
        source_ref: Optional[str] = None,
        memory_key: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        profile_key: Optional[str] = None,
        profile_value: Optional[str] = None,
    ) -> MemoryItem:
        user_id = self._require_user_id(user_id)
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")
        if kind not in MEMORY_POLICIES:
            raise ValueError(f"不支持的记忆类型：{kind}")
        policy = MEMORY_POLICIES[kind]
        now = self._now()
        now_text = now.isoformat()
        self.database.expire_memories(user_id, now_text)
        memory_key = memory_key or f"{kind}:{self._normalized_key(content)}"
        existing = self.database.find_active_memory(user_id, content, memory_key, now_text)
        valid_until = self._valid_until(now, policy["valid_days"])

        if existing:
            memory = existing.model_copy(
                update={
                    "confidence": min(0.98, existing.confidence + 0.08),
                    "importance": min(0.98, existing.importance + 0.05),
                    "evidence_count": existing.evidence_count + 1,
                    "source_type": "user_statement" if source_type == "user_statement" else existing.source_type,
                    "source_session_id": source_session_id or existing.source_session_id,
                    "source_ref": source_ref or existing.source_ref,
                    "valid_until": valid_until,
                    "updated_at": now_text,
                }
            )
        else:
            can_supersede = kind in {"preference", "constraint"} and (
                source_type == "user_statement"
                or (confidence is not None and confidence >= 0.85)
            )
            if can_supersede:
                for old in self.database.active_memories_for_key(user_id, memory_key, now_text):
                    if old.content != content:
                        self.database.update_memory_status(
                            user_id, old.memory_id, "superseded", now_text
                        )
            memory = MemoryItem(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                kind=kind,
                content=content,
                source_type=source_type,
                source_session_id=source_session_id,
                source_ref=source_ref,
                confidence=policy["confidence"] if confidence is None else confidence,
                importance=policy["importance"] if importance is None else importance,
                status="active",
                valid_from=now_text,
                valid_until=valid_until,
                half_life_days=policy["half_life_days"],
                evidence_count=1,
                memory_key=memory_key,
                created_at=now_text,
                updated_at=now_text,
            )
        self.save_memory(memory)
        if profile_key and profile_value:
            self._update_profile_from_memory(memory, profile_key, profile_value)
        return memory

    def revoke_memory(self, user_id: str, memory_id: str) -> None:
        user_id = self._require_user_id(user_id)
        now_text = self._now().isoformat()
        memory = self.database.get_memory(user_id, memory_id)
        if not memory:
            raise ValueError("找不到该用户的记忆")
        self.database.update_memory_status(user_id, memory_id, "revoked", now_text)
        for fact in self.database.list_profile_facts(user_id, now_text, include_inactive=False):
            if memory_id in fact.source_memory_ids:
                self.database.save_profile_fact(
                    fact.model_copy(update={"status": "revoked", "updated_at": now_text})
                )

    def get_profile(
        self,
        user_id: str,
        *,
        include_temporary: bool = False,
        limit: int = PROFILE_MAX_FACTS,
    ) -> list[UserProfileFact]:
        user_id = self._require_user_id(user_id)
        now_text = self._now().isoformat()
        self.database.expire_memories(user_id, now_text)
        facts = self.database.list_profile_facts(user_id, now_text)
        facts = [fact for fact in facts if fact.confidence >= PROFILE_MIN_CONFIDENCE]
        if not include_temporary:
            facts = [fact for fact in facts if fact.profile_key != "recent_context"]
        return facts[: max(0, limit)]

    def profile_context(self, user_id: str) -> dict[str, list[str]]:
        facts = self.get_profile(user_id, include_temporary=True, limit=PROFILE_MAX_FACTS)
        result: dict[str, list[str]] = {}
        used_chars = 0
        for fact in facts:
            value = fact.profile_value.strip()[:160]
            if not value or used_chars + len(value) > PROFILE_MAX_CHARS:
                continue
            values = result.setdefault(fact.profile_key, [])
            if value not in values:
                values.append(value)
                used_chars += len(value)
        return result

    def observe_statement(self, user_id: str, session_id: str, text: str) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        text = text.strip()
        if "不要记住" in text or "别记住" in text:
            now_text = self._now().isoformat()
            if "雨声" in text:
                targets = self.database.active_memories_for_key(
                    user_id, "sound:preference", now_text
                )
            else:
                targets = self.database.active_memories_for_source_session(
                    user_id, session_id, now_text
                )
            for memory in targets:
                self.revoke_memory(user_id, memory.memory_id)
            return []
        if not text or any(term in text.lower() for term in SECRET_TERMS):
            return []
        if any(term in text for term in SENSITIVE_PROFILE_TERMS):
            return []
        memories: list[MemoryItem] = []
        actions = ("写下明天第一步", "写下第一步", "慢呼吸", "听低音量雨声", "把手机放远")
        helpful_markers = ("很有用", "很有效", "有帮助", "对我有效", "能让我放松", "帮助我入睡")
        rejected_markers = ("不喜欢", "不想", "不要", "更清醒", "没用", "不适合我")
        for action in actions:
            if action in text and any(marker in text for marker in helpful_markers):
                memories.append(
                    self.record_evidence(
                        user_id,
                        "helpful_action",
                        action,
                        source_type="user_statement",
                        source_session_id=session_id,
                        memory_key=f"helpful:{action}",
                        confidence=0.88,
                        profile_key="helpful_action",
                        profile_value=action,
                    )
                )

        rejected_actions = ("刷视频", "玩游戏", "继续聊天", "数呼吸", "听雨声")
        for action in rejected_actions:
            if action in text and any(marker in text for marker in rejected_markers):
                memories.append(
                    self.record_evidence(
                        user_id,
                        "constraint",
                        action,
                        source_type="user_statement",
                        source_session_id=session_id,
                        memory_key="sound:preference" if action == "听雨声" else f"constraint:{action}",
                        confidence=0.90,
                        profile_key="rejected_action",
                        profile_value=action,
                    )
                )
                if action == "听雨声":
                    self.database.supersede_profile_key(
                        user_id, "sound_preference", "__none__", self._now().isoformat()
                    )

        sound_values = ("雨声", "流水声", "鸟鸣声", "白噪音")
        if any(marker in text for marker in ("喜欢", "习惯听", "想听")) and not any(
            marker in text for marker in ("不喜欢", "不想听", "不要听")
        ):
            for sound in sound_values:
                if sound in text:
                    memories.append(
                        self.record_evidence(
                            user_id,
                            "preference",
                            f"偏好{sound}",
                            source_type="user_statement",
                            source_session_id=session_id,
                            memory_key="sound:preference",
                            confidence=0.90,
                            profile_key="sound_preference",
                            profile_value=sound,
                        )
                    )
                    break

        if "下周" in text or "近期" in text or "这几天" in text:
            for topic in ("考试", "汇报", "截止", "答辩", "出差"):
                if topic in text:
                    memories.append(
                        self.record_evidence(
                            user_id,
                            "context",
                            f"近期有{topic}安排",
                            source_type="user_statement",
                            source_session_id=session_id,
                            memory_key=f"context:{topic}",
                            confidence=0.86,
                            profile_key="recent_context",
                            profile_value=f"近期有{topic}安排",
                        )
                    )
                    break

        if "咖啡" in text and any(term in text for term in ("睡不着", "清醒", "影响睡眠")):
            memories.append(
                self.record_evidence(
                    user_id,
                    "trigger",
                    "睡前摄入咖啡会影响入睡",
                    source_type="user_statement",
                    source_session_id=session_id,
                    memory_key="trigger:caffeine",
                    confidence=0.88,
                    profile_key="sleep_trigger",
                    profile_value="睡前摄入咖啡",
                )
            )

        if "建议" in text and "简短" in text and any(term in text for term in ("喜欢", "希望", "要")):
            memories.append(
                self.record_evidence(
                    user_id,
                    "preference",
                    "偏好简短、低刺激的建议",
                    source_type="user_statement",
                    source_session_id=session_id,
                    memory_key="advice:style",
                    confidence=0.90,
                    profile_key="advice_style",
                    profile_value="简短、低刺激",
                )
            )
        return memories

    def consolidate_session(
        self,
        user_id: str,
        session_id: str,
        today_input: str,
        audio_id: str | None = None,
    ) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        candidates = self.observe_statement(user_id, session_id, today_input)
        if "不要记住" in today_input or "别记住" in today_input:
            return candidates
        lowered = today_input.lower()
        if any(word in lowered for word in ("工作", "邮件", "开会", "任务", "deadline", "项目")):
            candidates.append(
                self.record_evidence(
                    user_id,
                    "pattern",
                    "工作或任务未收尾时，睡前容易继续思考这些事情",
                    source_session_id=session_id,
                    memory_key="pattern:unfinished_work",
                    confidence=0.72,
                    profile_key="recurring_concern",
                    profile_value="工作或任务未收尾",
                )
            )
        if audio_id:
            candidates.append(
                self.record_evidence(
                    user_id,
                    "preference",
                    f"睡前选择音频 {audio_id}",
                    source_type="session",
                    source_session_id=session_id,
                    source_ref=audio_id,
                    memory_key="audio:default",
                    confidence=0.65,
                    profile_key="sound_preference",
                    profile_value=audio_id,
                )
            )
        return candidates

    def _update_profile_from_memory(
        self, memory: MemoryItem, profile_key: str, profile_value: str
    ) -> None:
        if any(term in profile_value for term in SENSITIVE_PROFILE_TERMS):
            return
        is_temporary = profile_key == "recent_context" and memory.valid_until is not None
        is_explicit = memory.source_type == "user_statement" and memory.confidence >= 0.85
        if not (is_temporary or is_explicit or memory.evidence_count >= 2):
            return
        now_text = self._now().isoformat()
        existing = self.database.find_profile_fact(memory.user_id, profile_key, profile_value, now_text)
        if profile_key in SINGULAR_PROFILE_KEYS:
            self.database.supersede_profile_key(memory.user_id, profile_key, profile_value, now_text)
        if existing:
            sources = list(dict.fromkeys([*existing.source_memory_ids, memory.memory_id]))
            fact = existing.model_copy(
                update={
                    "confidence": max(existing.confidence, memory.confidence),
                    "source_memory_ids": sources,
                    "evidence_count": max(existing.evidence_count, memory.evidence_count),
                    "valid_until": memory.valid_until,
                    "updated_at": now_text,
                }
            )
        else:
            fact = UserProfileFact(
                profile_id=str(uuid.uuid4()),
                user_id=memory.user_id,
                profile_key=profile_key,
                profile_value=profile_value[:160],
                source_type="user_statement" if memory.source_type == "user_statement" else "observed_pattern",
                confidence=memory.confidence,
                source_memory_ids=[memory.memory_id],
                evidence_count=memory.evidence_count,
                valid_until=memory.valid_until,
                created_at=now_text,
                updated_at=now_text,
            )
        self.database.save_profile_fact(fact)

    @staticmethod
    def _valid_until(now: datetime, valid_days: Optional[float]) -> Optional[str]:
        if valid_days is None:
            return None
        return (now + timedelta(days=valid_days)).isoformat()

    @staticmethod
    def _normalized_key(content: str) -> str:
        return re.sub(r"\W+", "", content.lower())[:80] or str(uuid.uuid4())
