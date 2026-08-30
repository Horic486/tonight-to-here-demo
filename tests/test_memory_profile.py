from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from advice import AdviceService
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from models import AudioPreference, ContextBundle, MemoryItem
from rag import GuidanceRAG
from search import WebSearchClient
from vector_store import LocalVectorStore


class FixedClock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class MemoryProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = FixedClock(datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.database = Database(self.root / "memory.sqlite3")
        self.vectors = LocalVectorStore(self.root / "vectors.json")
        self.memories = MemoryService(self.database, self.vectors, clock=self.clock)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def _memory(self, *, status: str = "active", valid_until: str | None = None) -> MemoryItem:
        now = self.clock().isoformat()
        return MemoryItem(
            memory_id=f"memory-{status}-{valid_until or 'open'}",
            user_id="user-a",
            kind="helpful_action",
            content="写下明天第一步有助于停止反复思考",
            source_type="user_statement",
            confidence=0.9,
            importance=0.8,
            status=status,
            valid_from=now,
            valid_until=valid_until,
            half_life_days=30,
            created_at=now,
            updated_at=now,
        )

    def test_only_active_unexpired_memories_are_recalled(self) -> None:
        active = self._memory(valid_until=(self.clock() + timedelta(days=2)).isoformat())
        expired = self._memory(valid_until=(self.clock() - timedelta(seconds=1)).isoformat())
        expired.memory_id = "memory-expired-by-time"
        revoked = self._memory(status="revoked")
        superseded = self._memory(status="superseded")
        for memory in (active, expired, revoked, superseded):
            self.memories.save_memory(memory)

        recalled = self.memories.retrieve_long_term("user-a", "写下明天第一步", top_k=10)

        self.assertEqual(recalled, [active.content])
        self.assertEqual(self.database.get_memory("user-a", expired.memory_id).status, "expired")

    def test_effective_score_halves_after_one_half_life(self) -> None:
        memory = self._memory()
        initial = self.memories.effective_score(memory, relevance_score=1.0)
        self.clock.value += timedelta(days=30)

        decayed = self.memories.effective_score(memory, relevance_score=1.0)

        self.assertAlmostEqual(decayed, initial / 2, places=6)

    def test_repeated_evidence_reinforces_and_refreshes_memory(self) -> None:
        first = self.memories.record_evidence(
            "user-a",
            "pattern",
            "工作未收尾时睡前容易反复思考",
            memory_key="pattern:unfinished_work",
            source_session_id="session-1",
        )
        original_valid_until = first.valid_until
        self.clock.value += timedelta(days=7)

        reinforced = self.memories.record_evidence(
            "user-a",
            "pattern",
            "工作未收尾时睡前容易反复思考",
            memory_key="pattern:unfinished_work",
            source_session_id="session-2",
        )

        self.assertEqual(reinforced.memory_id, first.memory_id)
        self.assertEqual(reinforced.evidence_count, 2)
        self.assertGreater(reinforced.confidence, first.confidence)
        self.assertGreater(reinforced.importance, first.importance)
        self.assertGreater(reinforced.valid_until, original_valid_until)

    def test_new_preference_supersedes_old_preference_without_deleting_it(self) -> None:
        old = self.memories.record_evidence(
            "user-a",
            "preference",
            "偏好雨声音频",
            source_type="user_statement",
            memory_key="audio:default",
            confidence=0.9,
        )
        self.clock.value += timedelta(days=1)
        new = self.memories.record_evidence(
            "user-a",
            "preference",
            "偏好流水声音频",
            source_type="user_statement",
            memory_key="audio:default",
            confidence=0.9,
        )

        self.assertEqual(self.database.get_memory("user-a", old.memory_id).status, "superseded")
        self.assertEqual(self.database.get_memory("user-a", new.memory_id).status, "active")
        self.assertNotIn(old.content, self.memories.retrieve_long_term("user-a", "雨声音频", top_k=10))

    def test_do_not_remember_revokes_same_session_memory_and_profile(self) -> None:
        memory = self.memories.record_evidence(
            "user-a",
            "helpful_action",
            "写下明天第一步",
            source_type="user_statement",
            source_session_id="session-1",
            memory_key="helpful:first_step",
            confidence=0.9,
            profile_key="helpful_action",
            profile_value="写下明天第一步",
        )

        self.memories.observe_statement("user-a", "session-1", "不要记住这件事")

        self.assertEqual(self.database.get_memory("user-a", memory.memory_id).status, "revoked")
        self.assertEqual(self.memories.get_profile("user-a"), [])

    def test_sqlite_and_vector_memory_are_strictly_user_isolated(self) -> None:
        first = self.memories.record_evidence(
            "user-a",
            "helpful_action",
            "写下明天第一步",
            source_type="user_statement",
            memory_key="helpful:first_step",
            confidence=0.9,
            profile_key="helpful_action",
            profile_value="写下明天第一步",
        )
        second = self.memories.record_evidence(
            "user-b", "helpful_action", "写下明天第一步", memory_key="helpful:first_step"
        )
        self.memories.record_evidence(
            "user-b",
            "helpful_action",
            "慢呼吸",
            source_type="user_statement",
            memory_key="helpful:breathing",
            confidence=0.9,
            profile_key="helpful_action",
            profile_value="慢呼吸",
        )

        self.assertNotEqual(first.memory_id, second.memory_id)
        now = self.clock().isoformat()
        self.assertEqual(len(self.database.list_memories("user-a", now=now)), 1)
        self.assertEqual(len(self.database.list_memories("user-b", now=now)), 2)
        self.assertEqual(
            {record["metadata"]["user_id"] for record in self.vectors.records if record["record_id"] == first.memory_id},
            {"user-a"},
        )
        self.assertEqual(self.memories.retrieve_long_term("user-a", "明天第一步"), [first.content])
        self.assertEqual(
            self.memories.profile_context("user-a").get("helpful_action"),
            ["写下明天第一步"],
        )
        self.assertEqual(
            self.memories.profile_context("user-b").get("helpful_action"),
            ["慢呼吸"],
        )

    def test_missing_user_id_is_rejected(self) -> None:
        for operation in (
            lambda: self.memories.retrieve_long_term("", "query"),
            lambda: self.memories.record_evidence("  ", "pattern", "content"),
            lambda: self.database.list_memories(""),
            lambda: self.memories.get_profile(""),
            lambda: self.database.ensure_user(""),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_vague_single_statement_does_not_become_stable_profile(self) -> None:
        self.memories.observe_statement("user-a", "session-1", "最近好像总有点事情放不下")

        self.assertEqual(self.memories.get_profile("user-a"), [])

    def test_repeated_consistent_evidence_forms_profile(self) -> None:
        self.memories.consolidate_session("user-a", "session-1", "今晚工作任务还没收尾")
        self.clock.value += timedelta(days=1)
        self.memories.consolidate_session("user-a", "session-2", "项目邮件没处理完，一直在想")

        profile = self.memories.get_profile("user-a")

        self.assertEqual(len(profile), 1)
        self.assertEqual(profile[0].profile_key, "recurring_concern")
        self.assertGreaterEqual(profile[0].evidence_count, 2)

    def test_temporary_context_has_a_finite_profile_lifetime(self) -> None:
        self.memories.observe_statement("user-a", "session-1", "下周要考试")

        facts = self.memories.get_profile("user-a", include_temporary=True)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].profile_key, "recent_context")
        self.assertIsNotNone(facts[0].valid_until)

    def test_context_includes_bounded_profile_separately_from_memories(self) -> None:
        for session_index in range(2):
            self.memories.record_evidence(
                "user-a",
                "helpful_action",
                "写下明天第一步",
                memory_key="helpful:first_step",
                source_session_id=f"session-{session_index}",
                profile_key="helpful_action",
                profile_value="写下明天第一步",
            )
        rag = GuidanceRAG(Path(__file__).parents[1] / "data" / "knowledge", self.vectors)
        assembler = ContextAssembler(self.database, self.memories, rag)

        bundle = assembler.build(
            "user-a",
            "missing-session-is-safe",
            "WIND_DOWN",
            "脑子还在想明天的汇报",
            AudioPreference(user_id="user-a", default_audio_id="rain_01"),
        )

        self.assertIn("写下明天第一步", bundle.user_profile["helpful_action"])
        self.assertLessEqual(sum(len(value) for values in bundle.user_profile.values() for value in values), 500)

    def test_mock_and_api_failure_prioritize_helpful_action_and_avoid_rejected_action(self) -> None:
        context = ContextBundle(
            current_stage="SLEEP_FOLLOWUP",
            today_input="脑子还在想明天的汇报",
            user_profile={
                "helpful_action": ["写下明天第一步"],
                "rejected_action": ["刷视频"],
            },
        )
        mock_result = LLMClient("mock").generate_followup(context, "还在反复想汇报")
        api_client = LLMClient("api")
        api_client.api_key = "test-key"
        api_client._call_json = lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable"))
        api_result = api_client.generate_followup(context, "还在反复想汇报")

        for result in (mock_result, api_result):
            rendered = " ".join([result.action_title, *result.action_steps])
            self.assertIn("写下明天第一步", rendered)
            self.assertNotIn("刷视频", rendered)
        self.assertFalse(mock_result.fallback_used)
        self.assertTrue(api_result.fallback_used)

    def test_api_result_is_filtered_when_model_recommends_rejected_action(self) -> None:
        context = ContextBundle(
            current_stage="SLEEP_FOLLOWUP",
            today_input="还在想工作",
            user_profile={
                "helpful_action": ["写下明天第一步"],
                "rejected_action": ["刷视频"],
            },
        )
        client = LLMClient("api")
        client.api_key = "test-key"
        client._call_json = lambda _: {
            "message": "先放松一下",
            "action_title": "刷视频转移注意力",
            "action_steps": ["刷视频十分钟", "然后回到床上"],
        }

        result = client.generate_followup(context, "脑子停不下来")
        rendered = " ".join([result.action_title, *result.action_steps])

        self.assertNotIn("刷视频", rendered)
        self.assertIn("写下明天第一步", rendered)

    def test_profile_participates_in_wind_down_and_close(self) -> None:
        context = ContextBundle(
            current_stage="WIND_DOWN",
            today_input="脑子还在想明天的工作汇报",
            user_profile={
                "helpful_action": ["写下明天第一步"],
                "sound_preference": ["雨声"],
            },
        )
        client = LLMClient("mock")

        transition = client.generate_transition(context)
        closure = client.generate_closure(context, "明天第一步：列出汇报提纲")

        self.assertIn("写下明天第一步", " ".join([transition.action_title, *transition.action_steps]))
        self.assertIn("低音量雨声", closure)

    def test_old_memory_schema_migrates_idempotently_without_data_loss(self) -> None:
        path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute(
            """CREATE TABLE memory_items (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                consent INTEGER NOT NULL,
                source_session_id TEXT,
                created_at TEXT NOT NULL,
                last_used_at TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy-memory", "legacy-user", "pattern", "旧记忆", 0.7, 1, "old-session", "2025-01-01T00:00:00+00:00", None),
        )
        connection.commit()
        connection.close()

        migrated = Database(path)
        migrated.initialize()
        columns = {row["name"] for row in migrated.connection.execute("PRAGMA table_info(memory_items)")}
        indexes = {row["name"] for row in migrated.connection.execute("PRAGMA index_list(memory_items)")}
        memory = migrated.get_memory("legacy-user", "legacy-memory")

        self.assertTrue(
            {"source_type", "source_ref", "importance", "status", "valid_from", "valid_until", "half_life_days", "evidence_count", "updated_at"}
            <= columns
        )
        self.assertIn("idx_memory_user_status_valid_until", indexes)
        self.assertEqual(memory.content, "旧记忆")
        self.assertEqual(memory.status, "active")
        self.assertEqual(memory.evidence_count, 1)
        self.assertIsNotNone(
            migrated.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'user_profile_facts'"
            ).fetchone()
        )
        migrated.close()


if __name__ == "__main__":
    unittest.main()
