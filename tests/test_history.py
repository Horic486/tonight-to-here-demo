from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from advice import AdviceService
from audio import AudioService
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from rag import GuidanceRAG
from search import WebSearchClient
from vector_store import LocalVectorStore
from workflow import WorkflowEngine


class FixedClock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class ConversationHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = FixedClock(datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc))
        self.database = Database(self.root / "history.sqlite3", clock=self.clock)
        self.vectors = LocalVectorStore(self.root / "vectors.json")
        self.memories = MemoryService(self.database, self.vectors, clock=self.clock)
        self.audio = AudioService(
            self.database,
            Path(__file__).parents[1] / "data" / "audio",
            self.root / "user-audio",
        )
        self.rag = GuidanceRAG(Path(__file__).parents[1] / "data" / "knowledge", self.vectors)
        self.llm = LLMClient("mock")
        self.context = ContextAssembler(self.database, self.memories, self.rag)
        self.advice = AdviceService(self.context, self.rag, self.llm, WebSearchClient())
        self.workflow = WorkflowEngine(
            self.database,
            self.memories,
            self.context,
            self.llm,
            self.audio,
            self.advice,
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_same_hong_kong_date_reuses_session_and_creates_new_round(self) -> None:
        first_session, first_round = self.workflow.start_round("user-a")
        self.clock.value += timedelta(minutes=20)

        second_session, second_round = self.workflow.start_round("user-a")

        self.assertEqual(second_session, first_session)
        self.assertNotEqual(second_round, first_round)
        self.assertEqual(self.database.get_history_round("user-a", first_round).status, "abandoned")
        self.assertEqual(self.database.get_history_round("user-a", second_round).round_index, 2)

    def test_hong_kong_date_change_only_creates_session_when_new_round_starts(self) -> None:
        session_id, round_id = self.workflow.start_round("user-a")
        original_counts = self._id_counts()
        self.clock.value += timedelta(hours=2)

        self.database.list_history_dates("user-a")

        self.assertEqual(self._id_counts(), original_counts)
        self.assertEqual(self.database.get_history_round("user-a", round_id).session_id, session_id)

        new_session, new_round = self.workflow.start_round("user-a")
        self.assertNotEqual(new_session, session_id)
        self.assertNotEqual(new_round, round_id)
        self.assertEqual(self.database.get_session(new_session)["local_date"], "2026-08-31")

    def test_utc_date_change_does_not_split_same_hong_kong_date(self) -> None:
        self.clock.value = datetime(2026, 8, 30, 23, 30, tzinfo=timezone.utc)
        first_session, first_round = self.workflow.start_round("user-a")
        self.clock.value = datetime(2026, 8, 31, 0, 30, tzinfo=timezone.utc)

        second_session, second_round = self.workflow.start_round("user-a")

        self.assertEqual(second_session, first_session)
        self.assertNotEqual(second_round, first_round)
        self.assertEqual(self.database.get_session(first_session)["local_date"], "2026-08-31")

    def test_full_round_history_restores_feeling_items_arrangements_and_advice(self) -> None:
        user_id = "user-a"
        session_id, round_id = self.workflow.start_round(user_id)
        preference = self.audio.preference(user_id)
        concern = "明天要准备汇报，还要回复客户邮件。"

        items = self.workflow.capture(
            session_id,
            concern,
            round_id=round_id,
            initial_feeling="有点焦虑，脑子停不下来",
        )
        self.workflow.triage(
            session_id,
            {index: "tomorrow" for index in range(len(items))},
            round_id=round_id,
        )
        tomorrow_card = self.workflow.tomorrow_plan(session_id, round_id=round_id)
        transition = self.workflow.wind_down(
            session_id, user_id, preference, round_id=round_id
        )
        first_followup = self.workflow.follow_up(
            session_id,
            user_id,
            preference,
            "脑子还在反复想汇报",
            round_index=1,
            allow_web=False,
            round_id=round_id,
        )
        second_followup = self.workflow.follow_up(
            session_id,
            user_id,
            preference,
            "身体不困，但不想继续刷手机",
            round_index=2,
            allow_web=False,
            round_id=round_id,
        )
        closure = self.workflow.close(session_id, user_id, preference, round_id=round_id)

        history = self.database.get_history_round(user_id, round_id)

        self.assertEqual(history.initial_feeling, "有点焦虑，脑子停不下来")
        self.assertEqual(history.concern_input, concern)
        self.assertEqual(len(history.items), len(items))
        self.assertEqual(
            [item["slot"] for item in history.arrangements],
            ["tomorrow"] * len(items),
        )
        self.assertEqual(history.tomorrow_card, tomorrow_card)
        self.assertEqual(history.wind_down_advice["action_title"], transition.action_title)
        self.assertEqual(
            [entry["text"] for entry in history.followup_feedback],
            ["脑子还在反复想汇报", "身体不困，但不想继续刷手机"],
        )
        self.assertEqual(
            [entry["action_title"] for entry in history.followup_advice],
            [first_followup.action_title, second_followup.action_title],
        )
        self.assertTrue(all(entry["generation_mode"] == "mock" for entry in history.followup_advice))
        self.assertEqual(history.closure_message, closure)
        self.assertEqual(history.status, "completed")
        self.assertIsNotNone(history.completed_at)

    def test_tonight_minimum_action_is_saved_and_completes_round(self) -> None:
        user_id = "user-a"
        session_id, round_id = self.workflow.start_round(user_id)
        preference = self.audio.preference(user_id)
        items = self.workflow.capture(
            session_id,
            "今晚把桌上的文件收好。",
            round_id=round_id,
            initial_feeling="有点放不下",
        )
        self.workflow.triage(session_id, {0: "tonight"}, round_id=round_id)
        self.workflow.tomorrow_plan(session_id, round_id=round_id)

        result = self.workflow.finish_with_tonight_actions(
            session_id, user_id, preference, round_id=round_id
        )
        history = self.database.get_history_round(user_id, round_id)

        self.assertEqual(history.status, "completed")
        self.assertEqual(history.tonight_action["action_title"], result.action_title)
        self.assertIn(items[0].minimum_action, history.tonight_action["action_steps"])

    def test_repeated_capture_does_not_create_duplicate_history_round(self) -> None:
        session_id, round_id = self.workflow.start_round("user-a")

        self.workflow.capture(session_id, "明天要交报告", round_id=round_id)
        self.workflow.capture(session_id, "明天要交报告", round_id=round_id)

        rounds = self.database.list_history_rounds("user-a", "2026-08-30")
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0].round_id, round_id)

    def test_empty_active_round_is_not_shown_until_input_is_submitted(self) -> None:
        session_id, round_id = self.workflow.start_round("user-a")

        self.assertEqual(self.database.list_history_dates("user-a"), [])

        self.workflow.capture(session_id, "明天要开会", round_id=round_id)

        self.assertEqual(self.database.list_history_dates("user-a")[0]["round_count"], 1)

    def test_history_queries_are_user_isolated_and_read_only(self) -> None:
        session_id, round_id = self.workflow.start_round("user-a")
        self.workflow.capture(session_id, "明天要开会", round_id=round_id)
        before = self._id_counts()
        state_before = self.database.get_session_for_user("user-a", session_id)["state"]

        dates = self.database.list_history_dates("user-a")
        rounds = self.database.list_history_rounds("user-a", dates[0]["local_date"])
        own = self.database.get_history_round("user-a", round_id)

        self.assertEqual(self._id_counts(), before)
        self.assertEqual(own.round_id, round_id)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(self.database.list_history_dates("user-b"), [])
        self.assertIsNone(self.database.get_history_round("user-b", round_id))
        self.assertIsNone(self.database.get_session_for_user("user-b", session_id))
        self.assertEqual(
            self.database.get_session_for_user("user-a", session_id)["state"], state_before
        )
        with self.assertRaises(ValueError):
            self.database.list_history_dates("")

    def test_api_failure_followup_is_persisted_as_local_fallback(self) -> None:
        user_id = "user-a"
        api_llm = LLMClient("api")
        api_llm.api_key = "test-key"
        api_llm._call_json = lambda _: (_ for _ in ()).throw(RuntimeError("offline"))
        advice = AdviceService(self.context, self.rag, api_llm, WebSearchClient())
        workflow = WorkflowEngine(
            self.database,
            self.memories,
            self.context,
            api_llm,
            self.audio,
            advice,
        )
        session_id, round_id = workflow.start_round(user_id)
        preference = self.audio.preference(user_id)
        items = workflow.capture(session_id, "明天要开会", round_id=round_id)
        workflow.triage(
            session_id, {index: "tomorrow" for index in range(len(items))}, round_id=round_id
        )
        workflow.tomorrow_plan(session_id, round_id=round_id)
        workflow.wind_down(session_id, user_id, preference, round_id=round_id)

        result = workflow.follow_up(
            session_id,
            user_id,
            preference,
            "还是睡不着",
            round_index=1,
            round_id=round_id,
        )
        history = self.database.get_history_round(user_id, round_id)

        self.assertTrue(result.fallback_used)
        self.assertTrue(history.wind_down_advice["fallback_used"])
        self.assertEqual(history.wind_down_advice["generation_mode"], "local_fallback")
        self.assertTrue(history.followup_advice[0]["fallback_used"])
        self.assertEqual(history.followup_advice[0]["generation_mode"], "local_fallback")
        self.assertFalse(history.followup_advice[0]["web_used"])

    def test_legacy_sessions_migrate_once_to_history_rounds(self) -> None:
        path = self.root / "legacy-history.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute(
            """CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                state TEXT NOT NULL,
                today_input TEXT DEFAULT '',
                items_json TEXT DEFAULT '[]',
                tomorrow_card TEXT DEFAULT '',
                transition_json TEXT DEFAULT '{}',
                closure_message TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-session",
                "legacy-user",
                "CLOSE",
                "明天要交报告",
                '[{"content":"明天要交报告","suggested_slot":"tomorrow"}]',
                "明天第一步：整理提纲",
                '{"state":"WIND_DOWN","action_title":"慢呼吸","action_steps":["呼吸"]}',
                "今晚到此",
                "2026-08-29T18:00:00+00:00",
                "2026-08-29T18:10:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        migrated = Database(path, clock=self.clock)
        migrated.initialize()
        history = migrated.list_history_rounds("legacy-user", "2026-08-30")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].concern_input, "明天要交报告")
        self.assertEqual(history[0].status, "completed")
        self.assertEqual(history[0].round_index, 1)
        self.assertEqual(
            migrated.connection.execute("SELECT COUNT(*) FROM conversation_rounds").fetchone()[0],
            1,
        )
        migrated.close()

    def _id_counts(self) -> tuple[int, int]:
        sessions = self.database.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        rounds = self.database.connection.execute(
            "SELECT COUNT(*) FROM conversation_rounds"
        ).fetchone()[0]
        return sessions, rounds


if __name__ == "__main__":
    unittest.main()
