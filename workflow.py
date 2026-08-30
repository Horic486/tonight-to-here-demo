from __future__ import annotations

import json

from advice import AdviceService
from audio import AudioService
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from models import AudioPreference, TodoItem, WorkflowResult


class WorkflowEngine:
    def __init__(self, database: Database, memories: MemoryService, context: ContextAssembler, llm: LLMClient, audio: AudioService, advice: AdviceService):
        self.database = database
        self.memories = memories
        self.context = context
        self.llm = llm
        self.audio = audio
        self.advice = advice

    def start(self, user_id: str) -> str:
        session_id, _ = self.start_round(user_id)
        return session_id

    def start_round(self, user_id: str) -> tuple[str, str]:
        session_id, round_id = self.database.start_conversation_round(user_id)
        self.database.log_event(session_id, None, "CHECK_IN", {}, round_id=round_id)
        return session_id, round_id

    def capture(
        self,
        session_id: str,
        text: str,
        *,
        round_id: str | None = None,
        initial_feeling: str = "",
    ) -> list[TodoItem]:
        session, history = self._require_round(session_id, round_id=round_id)
        items = self.llm.extract_items(text)
        self.database.add_turn(session_id, "user", text, round_id=history.round_id)
        self.database.update_session(session_id, state="CAPTURE", today_input=text, items_json=json.dumps([item.model_dump() for item in items], ensure_ascii=False))
        self.database.update_history_round(
            session["user_id"],
            session_id,
            history.round_id,
            initial_feeling=initial_feeling.strip() or history.initial_feeling,
            concern_input=text,
            items_json=[item.model_dump() for item in items],
        )
        self.database.log_event(
            session_id,
            session["state"],
            "CAPTURE",
            {"item_count": len(items)},
            round_id=history.round_id,
        )
        return items

    def triage(
        self, session_id: str, slots: dict[int, str], *, round_id: str | None = None
    ) -> list[TodoItem]:
        session, history = self._require_round(session_id, round_id=round_id)
        items = [item.model_copy() for item in history.items]
        for index, slot in slots.items():
            if 0 <= index < len(items) and slot in {"tonight", "tomorrow", "later"}:
                items[index].suggested_slot = slot
        arrangements = [
            {
                "content": item.content,
                "slot": item.suggested_slot,
                "minimum_action": item.minimum_action,
            }
            for item in items
        ]
        self.database.update_session(session_id, state="TRIAGE", items_json=json.dumps([item.model_dump() for item in items], ensure_ascii=False))
        self.database.update_history_round(
            session["user_id"],
            session_id,
            history.round_id,
            items_json=[item.model_dump() for item in items],
            arrangements_json=arrangements,
        )
        self.database.log_event(
            session_id,
            session["state"],
            "TRIAGE",
            {"slots": slots},
            round_id=history.round_id,
        )
        return items

    def tomorrow_plan(self, session_id: str, *, round_id: str | None = None) -> str:
        session, history = self._require_round(session_id, round_id=round_id)
        items = history.items
        tomorrow = [item for item in items if item.suggested_slot == "tomorrow"]
        if tomorrow:
            card = "明天第一步：" + tomorrow[0].minimum_action
        else:
            card = "明天只需要从最重要的一步开始。"
        self.database.update_session(session_id, state="TOMORROW_PLAN", tomorrow_card=card)
        self.database.update_history_round(
            session["user_id"], session_id, history.round_id, tomorrow_card=card
        )
        self.database.log_event(
            session_id,
            session["state"],
            "TOMORROW_PLAN",
            {"card": card},
            round_id=history.round_id,
        )
        return card

    def finish_with_tonight_actions(
        self,
        session_id: str,
        user_id: str,
        preference: AudioPreference,
        *,
        round_id: str | None = None,
    ) -> WorkflowResult:
        session, history = self._require_round(session_id, user_id, round_id)
        items = history.items
        tonight = [item for item in items if item.suggested_slot == "tonight"]
        if not tonight:
            raise ValueError("没有选择今晚要做的最小动作")

        action_steps = [
            item.minimum_action or f"只完成“{item.content[:18]}”的最小动作"
            for item in tonight
        ]
        action_steps.append("做到心里能够放下就停，不继续扩展任务")
        result = WorkflowResult(
            state="TONIGHT_ACTION",
            message="你已经决定今晚只做最小动作。现在就去做吧，不需要把整件事全部完成。",
            action_title="做到你觉得可以放下为止",
            action_steps=action_steps,
        )
        summary = "；".join(item.content for item in tonight)
        self.database.update_session(
            session_id,
            state="TONIGHT_ACTION",
            transition_json=json.dumps(result.model_dump(), ensure_ascii=False),
            closure_message=result.message,
        )
        self.database.save_summary(session_id, user_id, f"今晚最小动作：{summary}")
        self.database.update_history_round(
            user_id,
            session_id,
            history.round_id,
            tonight_action_json={
                **result.model_dump(),
                "generation_mode": "local",
                "allow_web": False,
                "web_used": False,
            },
            closure_message=result.message,
            status="completed",
        )
        self.memories.consolidate_session(
            user_id, session_id, session["today_input"], preference.default_audio_id
        )
        self.database.log_event(
            session_id,
            session["state"],
            "TONIGHT_ACTION",
            {"item_count": len(tonight), **result.model_dump()},
            round_id=history.round_id,
        )
        return result

    def wind_down(
        self,
        session_id: str,
        user_id: str,
        preference: AudioPreference,
        *,
        round_id: str | None = None,
    ) -> WorkflowResult:
        session, history = self._require_round(session_id, user_id, round_id)
        bundle = self.context.build(user_id, session_id, "WIND_DOWN", session["today_input"], preference)
        result = self.llm.generate_transition(bundle)
        self.database.update_session(
            session_id, state="WIND_DOWN", transition_json=json.dumps(result.model_dump(), ensure_ascii=False)
        )
        advice_record = result.model_dump()
        advice_record["generation_mode"] = (
            "local_fallback" if result.fallback_used else self.llm.mode
        )
        advice_record["allow_web"] = False
        advice_record["web_used"] = False
        self.database.update_history_round(
            user_id,
            session_id,
            history.round_id,
            wind_down_advice_json=advice_record,
        )
        self.database.log_event(
            session_id,
            session["state"],
            "WIND_DOWN",
            result.model_dump(),
            round_id=history.round_id,
        )
        return result

    def follow_up(
        self,
        session_id: str,
        user_id: str,
        preference: AudioPreference,
        feedback: str,
        round_index: int,
        allow_web: bool = False,
        round_id: str | None = None,
    ) -> WorkflowResult:
        session, history = self._require_round(session_id, user_id, round_id)
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("请先写下现在的感受或状态")
        self.database.add_turn(session_id, "user", feedback, round_id=history.round_id)
        result = self.advice.generate(user_id, session_id, preference, feedback, allow_web=allow_web)
        self.memories.observe_statement(user_id, session_id, feedback)
        result.round_index = round_index
        self.database.append_history_entry(
            user_id,
            session_id,
            history.round_id,
            "followup_feedback_json",
            {
                "text": feedback,
                "allow_web": allow_web,
                "created_at": self.database.now_text(),
            },
        )
        advice_record = result.model_dump()
        advice_record["generation_mode"] = (
            "local_fallback" if result.fallback_used else self.llm.mode
        )
        advice_record["allow_web"] = allow_web
        advice_record["web_used"] = any(
            str(source).startswith(("http://", "https://")) for source in result.sources
        )
        self.database.append_history_entry(
            user_id,
            session_id,
            history.round_id,
            "followup_advice_json",
            advice_record,
        )
        self.database.update_session(
            session_id,
            state="SLEEP_FOLLOWUP",
            transition_json=json.dumps(result.model_dump(), ensure_ascii=False),
        )
        self.database.log_event(
            session_id,
            session["state"],
            "SLEEP_FOLLOWUP",
            {"round_index": round_index, "allow_web": allow_web, **result.model_dump()},
            round_id=history.round_id,
        )
        return result

    def close(
        self,
        session_id: str,
        user_id: str,
        preference: AudioPreference,
        *,
        round_id: str | None = None,
    ) -> str:
        session, history = self._require_round(session_id, user_id, round_id)
        self.memories.consolidate_session(
            user_id, session_id, session["today_input"], preference.default_audio_id
        )
        bundle = self.context.build(user_id, session_id, "CLOSE", session["today_input"], preference)
        message = self.llm.generate_closure(bundle, session["tomorrow_card"])
        self.database.update_session(session_id, state="CLOSE", closure_message=message)
        self.database.save_summary(session_id, user_id, f"{session['today_input']}；{session['tomorrow_card']}")
        self.database.update_history_round(
            user_id,
            session_id,
            history.round_id,
            closure_message=message,
            status="completed",
        )
        self.database.log_event(
            session_id,
            session["state"],
            "CLOSE",
            {"message": message},
            round_id=history.round_id,
        )
        return message

    def _require_session(self, session_id: str, user_id: str | None = None):
        session = self.database.get_session(session_id)
        if not session:
            raise ValueError("找不到该睡前会话")
        if user_id is not None and session["user_id"] != user_id:
            raise ValueError("该睡前会话不属于当前用户")
        return session

    def _require_round(
        self,
        session_id: str,
        user_id: str | None = None,
        round_id: str | None = None,
    ):
        session = self._require_session(session_id, user_id)
        owner_id = session["user_id"]
        history = (
            self.database.get_history_round(owner_id, round_id)
            if round_id
            else self.database.latest_round_for_session(session_id, owner_id)
        )
        if not history or history.session_id != session_id:
            raise ValueError("找不到当前用户的历史轮次")
        return session, history
