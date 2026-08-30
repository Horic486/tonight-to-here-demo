from __future__ import annotations

import tempfile
import threading
import unittest
import sqlite3
import os
from pathlib import Path
from unittest.mock import patch

from audio import AudioService
from advice import AdviceService
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from models import AudioPreference, ContextBundle, DEFAULT_AUDIO_VOLUME
from rag import GuidanceRAG
from search import WebSearchClient, _DuckDuckGoParser, reciprocal_rank_fusion
from vector_store import LocalVectorStore
from workflow import WorkflowEngine


class CoreFlowTest(unittest.TestCase):
    def test_llm_reads_api_key_from_named_environment_variable(self) -> None:
        with patch.dict(
            os.environ,
            {"MODEL_API_KEY": "", "MODEL_API_KEY_ENV": "HUOSHAN_FREE_API_KEY", "HUOSHAN_FREE_API_KEY": "test-key"},
        ):
            client = LLMClient("api")

        self.assertEqual(client.api_key, "test-key")

    def test_llm_applies_configured_output_and_thinking_limits(self) -> None:
        with patch.dict(
            os.environ,
            {"MODEL_MAX_TOKENS": "512", "MODEL_THINKING": "disabled"},
        ):
            client = LLMClient("api")
        payloads = []
        client._post_chat_completion = lambda payload: (
            payloads.append(payload)
            or {"choices": [{"message": {"content": '{"ok":true}'}}]}
        )

        result = client._call_json("test")

        self.assertTrue(result["ok"])
        self.assertEqual(payloads[0]["max_tokens"], 512)
        self.assertEqual(payloads[0]["thinking"], {"type": "disabled"})

    def test_semantic_event_segmentation_cases(self) -> None:
        cases = [
            (
                "single event",
                "明天要整理课程笔记。",
                1,
                ("课程笔记",),
            ),
            (
                "causal explanation across sentences",
                "我明天想做行测练习，我现在睡不着。明天想做练习是睡不着的原因。",
                1,
                ("行测练习", "睡不着"),
            ),
            (
                "explicit cause and result",
                "我明天要做行测练习，因为担心来不及，所以现在一直睡不着。",
                1,
                ("行测练习", "睡不着"),
            ),
            (
                "anaphoric emotional supplement",
                "明天要开会，这件事让我有点焦虑。",
                1,
                ("开会", "焦虑"),
            ),
            (
                "implicit action and emotion",
                "明天要提交论文，我一直担心会出错。",
                1,
                ("论文", "担心"),
            ),
            (
                "implicit desire and sleep-state chain",
                "我现在饥渴难耐，想要做爱，翻来覆去睡不着。",
                1,
                ("饥渴难耐", "做爱", "睡不着"),
            ),
            (
                "implicit task and modified sleep state",
                "明天要参加重要考试，脑子停不下来，翻来覆去睡不着。",
                1,
                ("考试", "停不下来", "睡不着"),
            ),
            (
                "explicit topic shift after internal state",
                "我现在饥渴难耐。另外，明天要提交课程报告。",
                2,
                ("饥渴难耐", "课程报告"),
            ),
            (
                "two independent tasks",
                "我需要准备明天的汇报，还要给客户回复邮件。",
                2,
                ("汇报", "邮件"),
            ),
            (
                "topic shift with punctuation",
                "我担心明天的考试。另外，家里的水费还没有交。",
                2,
                ("考试", "水费"),
            ),
            (
                "three newline separated tasks",
                "明天要交报告\n还要预约体检\n并给妈妈回电话",
                3,
                ("报告", "体检", "妈妈"),
            ),
            (
                "multiple tasks without punctuation",
                "明天要交报告还要预约体检并给妈妈回电话",
                3,
                ("报告", "体检", "妈妈"),
            ),
            (
                "mixed punctuation in one causal event",
                "明天要准备汇报；因为还没写完\n所以一直睡不着",
                1,
                ("汇报", "睡不着"),
            ),
            (
                "different dates remain independent",
                "明天要做行测练习。后天要交课程报告。",
                2,
                ("行测练习", "课程报告"),
            ),
            (
                "ambiguous state supplements previous task",
                "明天还有事情没处理，我有点放不下。",
                1,
                ("事情没处理", "放不下"),
            ),
            (
                "goal and ordered steps",
                "明天要完成课程报告，先整理实验数据，再补写结论。",
                1,
                ("课程报告", "实验数据", "结论"),
            ),
            (
                "same topic supplement",
                "项目方案还没写完，其中预算部分还需要调整。",
                1,
                ("项目方案", "预算部分"),
            ),
            (
                "additional task without punctuation",
                "明天复习考试还有一封邮件要回复",
                2,
                ("考试", "邮件"),
            ),
        ]
        client = LLMClient("mock")

        for name, text, expected_count, expected_fragments in cases:
            with self.subTest(name=name):
                items = client.extract_items(text)
                combined_content = " | ".join(item.content for item in items)
                self.assertEqual(len(items), expected_count, combined_content)
                for fragment in expected_fragments:
                    self.assertIn(fragment, combined_content)

    def test_api_items_use_the_same_semantic_post_processing(self) -> None:
        client = LLMClient("api")
        client.api_key = "test-key"
        responses = iter([
            {
                "items": [
                    {
                        "content": "我明天想做行测练习",
                        "category": "task",
                        "suggested_slot": "tomorrow",
                        "minimum_action": "准备练习",
                    },
                    {
                        "content": "我现在睡不着",
                        "category": "worry",
                        "suggested_slot": "tonight",
                        "minimum_action": "先休息",
                    },
                    {
                        "content": "明天想做练习是睡不着的原因",
                        "category": "worry",
                        "suggested_slot": "tomorrow",
                        "minimum_action": "记录原因",
                    },
                ]
            },
            {
                "items": [
                    {
                        "content": "我需要准备明天的汇报，还要给客户回复邮件",
                        "category": "task",
                        "suggested_slot": "tomorrow",
                        "minimum_action": "整理任务",
                    }
                ]
            },
        ])
        client._call_json = lambda _: next(responses)

        related = client.extract_items("相关事项")
        independent = client.extract_items("独立事项")

        self.assertEqual(len(related), 1)
        self.assertIn("行测练习", related[0].content)
        self.assertIn("睡不着", related[0].content)
        self.assertEqual(len(independent), 2)
        self.assertIn("汇报", independent[0].content)
        self.assertIn("邮件", independent[1].content)

    def test_api_post_processing_merges_implicit_relation_chain(self) -> None:
        client = LLMClient("api")
        client.api_key = "test-key"
        client._call_json = lambda _: {
            "items": [
                {
                    "content": "我现在饥渴难耐",
                    "category": "other",
                    "suggested_slot": "tonight",
                    "minimum_action": "记录状态",
                },
                {
                    "content": "想要做爱",
                    "category": "other",
                    "suggested_slot": "tonight",
                    "minimum_action": "记录想法",
                },
                {
                    "content": "翻来覆去睡不着",
                    "category": "worry",
                    "suggested_slot": "tonight",
                    "minimum_action": "先休息",
                },
            ]
        }

        items = client.extract_items("隐含关系事项")

        self.assertEqual(len(items), 1)
        self.assertIn("饥渴难耐", items[0].content)
        self.assertIn("做爱", items[0].content)
        self.assertIn("睡不着", items[0].content)

    def test_api_preserves_model_boundary_for_unseen_implicit_relation(self) -> None:
        client = LLMClient("api")
        client.api_key = "test-key"
        client._call_json = lambda _: {
            "items": [
                {
                    "content": "和朋友吵了一架，胸口发闷，整晚没睡好",
                    "category": "worry",
                    "suggested_slot": "tonight",
                    "minimum_action": "先记下这件事",
                }
            ]
        }

        items = client.extract_items("本地规则未覆盖的隐含关系")

        self.assertEqual(len(items), 1)
        self.assertIn("朋友吵了一架", items[0].content)
        self.assertIn("胸口发闷", items[0].content)
        self.assertIn("没睡好", items[0].content)

    def test_builtin_audio_catalog_uses_supplied_mp3_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "audio.sqlite3")
            audio = AudioService(
                database,
                Path(__file__).parents[1] / "data" / "audio",
                root / "user",
            )

            assets = {asset["audio_id"]: asset for asset in audio.catalog("test-user")}

            self.assertEqual(
                {asset["file_name"] for asset in assets.values()},
                {
                    "rainy-day-in-town-with-birds-singing.mp3",
                    "strong-rain.mp3",
                    "touching-the-water.mp3",
                },
            )
            self.assertEqual(assets["rain_01"]["duration_seconds"], 581)
            self.assertTrue(all(audio.path_for(asset).is_file() for asset in assets.values()))
            database.close()

    def test_fade_out_minutes_are_bounded_and_stored_as_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "preference.sqlite3")
            preference = AudioPreference(
                user_id="timer-user",
                default_audio_id="rain_01",
                fade_out_minutes=120,
            )

            database.save_audio_preference(preference)

            saved = database.get_audio_preference("timer-user", "rain_01")
            stored_seconds = database.connection.execute(
                "SELECT fade_out_seconds FROM audio_preferences WHERE user_id = ?",
                ("timer-user",),
            ).fetchone()[0]
            self.assertEqual(saved.fade_out_minutes, 120)
            self.assertEqual(stored_seconds, 7200)
            with self.assertRaises(ValueError):
                AudioPreference(
                    user_id="timer-user",
                    default_audio_id="rain_01",
                    fade_out_minutes=121,
                )
            database.close()

    def test_legacy_fade_out_default_migrates_to_twenty_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE audio_preferences (
                    user_id TEXT PRIMARY KEY,
                    default_audio_id TEXT NOT NULL,
                    volume REAL NOT NULL,
                    autoplay_enabled INTEGER NOT NULL,
                    fade_out_seconds INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO audio_preferences VALUES (?, ?, ?, ?, ?, ?)",
                ("legacy-user", "rain_01", 0.18, 1, 20, "now"),
            )
            connection.commit()
            connection.close()

            database = Database(path)

            preference = database.get_audio_preference("legacy-user", "rain_01")
            self.assertEqual(preference.fade_out_minutes, 20)
            self.assertEqual(
                database.connection.execute(
                    "SELECT fade_out_seconds FROM audio_preferences WHERE user_id = ?",
                    ("legacy-user",),
                ).fetchone()[0],
                1200,
            )
            database.close()

    def test_shared_database_serializes_concurrent_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary) / "concurrent.sqlite3")
            errors: list[Exception] = []
            barrier = threading.Barrier(8)

            def worker(index: int) -> None:
                try:
                    barrier.wait()
                    for turn_index in range(20):
                        user_id = f"thread-{index}-{turn_index}"
                        database.ensure_user(user_id)
                        session_id = database.create_session(user_id)
                        database.add_turn(session_id, "user", "测试输入")
                        database.update_session(session_id, state="CAPTURE")
                except Exception as exc:  # pragma: no cover - assertion reports the injected failure
                    errors.append(exc)

            workers = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for worker_thread in workers:
                worker_thread.start()
            for worker_thread in workers:
                worker_thread.join()

            database.close()
            self.assertEqual(errors, [])

    def test_audio_upload_keeps_user_path_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "audio.sqlite3")
            audio = AudioService(database, root / "builtins", root / "user")

            asset = audio.upload("x/../../escape", "night.wav", b"not-real-audio")
            stored_path = audio.path_for(asset)

            self.assertEqual(stored_path.parent, (root / "user").resolve())
            self.assertTrue(stored_path.exists())
            with self.assertRaises(ValueError):
                audio.path_for({"owner_type": "user", "file_name": "../outside.wav"})
            with self.assertRaises(ValueError):
                audio.upload("safe-user", "bad\nname.wav", b"invalid")
            database.close()

    def test_legacy_default_volume_migrates_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preference.sqlite3"
            database = Database(path)
            database.save_audio_preference(
                AudioPreference(user_id="legacy-user", default_audio_id="rain_01", volume=0.35)
            )
            database.connection.execute("DELETE FROM app_metadata WHERE key = ?", ("audio_default_volume_v2",))
            database.connection.commit()
            database.close()

            migrated = Database(path)
            preference = migrated.get_audio_preference("legacy-user", "rain_01")
            self.assertAlmostEqual(preference.volume, DEFAULT_AUDIO_VOLUME)

            migrated.save_audio_preference(
                AudioPreference(user_id="legacy-user", default_audio_id="rain_01", volume=0.35)
            )
            migrated.close()

            reopened = Database(path)
            self.assertAlmostEqual(reopened.get_audio_preference("legacy-user", "rain_01").volume, 0.35)
            reopened.close()

    def test_related_food_phrases_are_one_item(self) -> None:
        items = LLMClient("mock").extract_items("现在嘴有点馋，想吃点东西")
        self.assertEqual(len(items), 1)
        self.assertIn("嘴有点馋", items[0].content)
        self.assertIn("想吃点东西", items[0].content)

    def test_full_workflow_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "demo.sqlite3")
            vectors = LocalVectorStore(root / "vectors.json")
            audio = AudioService(database, root / "audio", root / "audio" / "user")
            memories = MemoryService(database, vectors)
            rag = GuidanceRAG(Path(__file__).parents[1] / "data" / "knowledge", vectors)
            context = ContextAssembler(database, memories, rag)
            advice = AdviceService(context, rag, LLMClient("mock"), WebSearchClient())
            workflow = WorkflowEngine(database, memories, context, LLMClient("mock"), audio, advice)
            user_id = "test-user"
            preference = audio.preference(user_id)

            session_id = workflow.start(user_id)
            items = workflow.capture(session_id, "明天要开会，今晚还有一封邮件没回。")
            self.assertGreaterEqual(len(items), 1)
            workflow.triage(session_id, {index: "tomorrow" for index in range(len(items))})
            card = workflow.tomorrow_plan(session_id)
            transition = workflow.wind_down(session_id, user_id, preference)
            closure = workflow.close(session_id, user_id, preference)

            self.assertIn("明天", card)
            self.assertTrue(transition.message)
            self.assertTrue(closure)
            self.assertTrue(memories.retrieve_long_term(user_id, "工作没收尾", top_k=3))
            database.close()

    def test_tonight_action_finishes_without_wind_down(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "demo.sqlite3")
            vectors = LocalVectorStore(root / "vectors.json")
            audio = AudioService(database, root / "audio", root / "audio" / "user")
            memories = MemoryService(database, vectors)
            rag = GuidanceRAG(Path(__file__).parents[1] / "data" / "knowledge", vectors)
            context = ContextAssembler(database, memories, rag)
            llm = LLMClient("mock")
            advice = AdviceService(context, rag, llm, WebSearchClient())
            workflow = WorkflowEngine(database, memories, context, llm, audio, advice)
            user_id = "tonight-user"
            preference = audio.preference(user_id)

            session_id = workflow.start(user_id)
            items = workflow.capture(session_id, "今晚把桌上的文件收好，还要准备明天的汇报。")
            self.assertEqual(len(items), 2)
            workflow.triage(session_id, {0: "tonight", 1: "tomorrow"})
            workflow.tomorrow_plan(session_id)
            result = workflow.finish_with_tonight_actions(session_id, user_id, preference)

            self.assertEqual(result.state, "TONIGHT_ACTION")
            self.assertIn("现在就去做", result.message)
            self.assertIn("可以放下", result.action_title)
            self.assertIn("不继续扩展任务", result.action_steps[-1])
            self.assertEqual(len(result.action_steps), 2)
            self.assertNotIn("汇报", "；".join(result.action_steps))
            self.assertEqual(database.get_session(session_id)["state"], "TONIGHT_ACTION")
            database.close()

    def test_followup_has_fallback_advice_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "demo.sqlite3")
            vectors = LocalVectorStore(root / "vectors.json")
            audio = AudioService(database, root / "audio", root / "audio" / "user")
            memories = MemoryService(database, vectors)
            rag = GuidanceRAG(Path(__file__).parents[1] / "data" / "knowledge", vectors)
            context = ContextAssembler(database, memories, rag)
            advice = AdviceService(context, rag, LLMClient("mock"), WebSearchClient())
            workflow = WorkflowEngine(database, memories, context, LLMClient("mock"), audio, advice)
            user_id = "followup-user"
            preference = audio.preference(user_id)
            session_id = workflow.start(user_id)
            workflow.capture(session_id, "今晚总在想明天的安排。")
            workflow.triage(session_id, {0: "tomorrow"})
            workflow.tomorrow_plan(session_id)
            workflow.wind_down(session_id, user_id, preference)
            result = workflow.follow_up(
                session_id, user_id, preference, "脑子还在反复想明天的事情", round_index=1, allow_web=False
            )
            self.assertEqual(result.state, "SLEEP_FOLLOWUP")
            self.assertTrue(result.action_steps)
            self.assertFalse(result.fallback_used)
            self.assertIn("user: 脑子还在反复想明天的事情", database.recent_turns(session_id))
            database.close()

    def test_llm_failure_sets_fallback_flag(self) -> None:
        llm = LLMClient("api")
        llm.api_key = "test-key"

        def fail(_: str):
            raise RuntimeError("model unavailable")

        llm._call_json = fail
        result = llm.generate_followup(ContextBundle(current_stage="SLEEP_FOLLOWUP", today_input="睡不着"), "睡不着")
        self.assertTrue(result.fallback_used)

    def test_rrf_keeps_local_and_web_candidates(self) -> None:
        from models import GuidanceChunk

        local = [GuidanceChunk(chunk_id="l", title="本地", content="减少屏幕刺激", source="local.md")]
        web = [GuidanceChunk(chunk_id="w", title="网页", content="安静活动建议", source="https://example.com")]
        merged = reciprocal_rank_fusion(local, web, top_k=2)
        self.assertEqual(len(merged), 2)

    def test_web_result_parser_cleans_title_and_snippet(self) -> None:
        parser = _DuckDuckGoParser()
        parser.feed('<a class="result__a" href="https://example.com">标题</a><a class="result__snippet">摘要内容</a>')
        parser.close()
        self.assertEqual(parser.results[0]["title"], "标题")
        self.assertEqual(parser.results[0]["snippet"], "摘要内容")


if __name__ == "__main__":
    unittest.main()
