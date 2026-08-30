from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from event_segmentation import compose_event, group_related_texts, segment_events, split_explicit_events
from models import ContextBundle, TodoItem, WorkflowResult


class LLMClient:
    """LLM gateway with a deterministic mock mode and an optional OpenAI-compatible mode."""

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.api_key_env = os.getenv("MODEL_API_KEY_ENV", "").strip()
        self.api_key = os.getenv("MODEL_API_KEY", "").strip()
        if not self.api_key and self.api_key_env:
            self.api_key = os.getenv(self.api_key_env, "").strip()
        self.base_url = os.getenv("MODEL_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.model_name = os.getenv("MODEL_NAME", "openrouter/free")
        self.http_referer = os.getenv("MODEL_HTTP_REFERER", "")
        self.app_name = os.getenv("MODEL_APP_NAME", "今晚到此")
        try:
            self.timeout_seconds = max(5.0, float(os.getenv("MODEL_TIMEOUT_SECONDS", "45")))
        except ValueError:
            self.timeout_seconds = 45.0
        try:
            self.max_tokens = max(128, min(4096, int(os.getenv("MODEL_MAX_TOKENS", "800"))))
        except ValueError:
            self.max_tokens = 800
        self.thinking_mode = os.getenv("MODEL_THINKING", "").strip().lower()

    def extract_items(self, text: str) -> list[TodoItem]:
        if self.mode == "api" and self.api_key:
            try:
                value = self._call_json(
                    "你负责识别用户实际上挂念的独立事件，而不是计算句子或分句数量。"
                    "日常表达经常省略‘因为、所以、这件事’等关系词：即使没有连接词，只要连续片段"
                    "构成同一条自然的状态、欲望或行动、情绪、结果链，也必须合并。"
                    "原因与结果、事项与其引起的担忧或失眠、指代补充、目标与步骤、同一对象的补充说明应合并。"
                    "只有能够分别处理、分别完成，或通过‘另外、还要、此外’明确转向的事项才分开。"
                    "例如‘我现在饥渴难耐，想要做爱，翻来覆去睡不着’是一个事件；"
                    "‘我需要准备明天的汇报，还要给客户回复邮件’是两个事件。"
                    "先在内部完成关系判断，不要输出分析过程，不要做医疗判断。只返回 JSON 对象："
                    '{"items":[{"content":"...","category":"task|worry|entertainment|other",'
                    '"suggested_slot":"tonight|tomorrow|later","minimum_action":"..."}]}。\n'
                    "用户输入：" + text
                )
                raw_items = value.get("items", []) if isinstance(value, dict) else value
                if not isinstance(raw_items, list) or not raw_items:
                    raise ValueError("模型未返回可用事项")
                return self._merge_related_items(
                    [TodoItem(**item) for item in raw_items], preserve_model_boundaries=True
                )
            except Exception:
                pass
        return self._merge_related_items(self._mock_extract_items(text))

    def generate_transition(self, context: ContextBundle) -> WorkflowResult:
        if self.mode == "api" and self.api_key:
            try:
                value = self._call_json(self._prompt(context, "低刺激过渡"))
                return self._apply_profile_rules(
                    context, WorkflowResult(state="WIND_DOWN", **value), prefer_helpful=True
                )
            except Exception:
                pass
        fallback_used = self.mode == "api"
        helpful = self._first_profile_value(context, "helpful_action")
        if helpful and any(word in context.today_input for word in ("想", "担心", "工作", "任务", "汇报", "邮件")):
            return WorkflowResult(
                state="WIND_DOWN",
                message="今天的事情已经有了去处，现在只做一个能帮助你停下来的小动作。",
                action_title=helpful,
                action_steps=self._personalized_steps(
                    context,
                    [f"只做一次：{helpful}", "完成后停止补充细节，把注意力转回休息"],
                ),
                fallback_used=fallback_used,
            )
        words = context.today_input
        if any(word in words for word in ("工作", "邮件", "任务", "开会")):
            message = "今天的事情已经有了去处，今晚不必继续替明天工作。"
            action_title = "把手机放远，听一会儿固定的声音"
            steps = ["打开默认白噪音", "把屏幕亮度调低", "只听 10 分钟，不再处理新任务"]
        else:
            message = "你已经为今天按下了暂停键，不需要现在把所有事情想完。"
            action_title = "选择一个安静的小动作"
            steps = ["打开默认白噪音", "做几次慢呼吸", "让注意力回到声音和身体"]
        return WorkflowResult(
            state="WIND_DOWN",
            message=message,
            action_title=action_title,
            action_steps=self._personalized_steps(context, steps),
            fallback_used=fallback_used,
        )

    def generate_followup(self, context: ContextBundle, feedback: str) -> WorkflowResult:
        if self.mode == "api" and self.api_key:
            try:
                value = self._call_json(self._prompt(context, "还是睡不着后的下一步") + f"\n用户最新感受：{feedback}")
                return self._apply_profile_rules(
                    context,
                    WorkflowResult(state="SLEEP_FOLLOWUP", **value),
                    prefer_helpful=True,
                )
            except Exception:
                return self._fallback_followup(context, feedback, fallback_used=True)
        return self._fallback_followup(context, feedback, fallback_used=False)

    def generate_closure(self, context: ContextBundle, tomorrow_card: str) -> str:
        if self.mode == "api" and self.api_key:
            try:
                value = self._call_json(self._prompt(context, "结束今天") + f"\n明日卡片：{tomorrow_card}")
                message = str(value.get("message", "今晚到此，剩下的事情明天再处理。"))
                if any(
                    action in message
                    for action in context.user_profile.get("rejected_action", [])
                ):
                    raise ValueError("模型建议包含用户已拒绝的动作")
                return message
            except Exception:
                pass
        sound = self._first_profile_value(context, "sound_preference")
        sound_phrase = f" 保持低音量{sound}，" if sound else " "
        return (
            f"{tomorrow_card or '明天再从最重要的一步开始。'}"
            f"{sound_phrase}今晚到此，剩下的事情明天再处理。"
        )

    def _fallback_followup(
        self, context: ContextBundle, feedback: str, fallback_used: bool
    ) -> WorkflowResult:
        helpful = self._first_profile_value(context, "helpful_action")
        if helpful and any(word in feedback for word in ("想", "焦虑", "担心", "汇报", "任务", "停不下来", "反复")):
            return WorkflowResult(
                state="SLEEP_FOLLOWUP",
                message="先不继续分析，把注意力放到一个已经对你有帮助的小动作上。",
                action_title=helpful,
                action_steps=self._personalized_steps(
                    context,
                    [f"现在只做一次：{helpful}", "做完就停，不再扩展今晚要处理的内容"],
                ),
                fallback_used=fallback_used,
            )
        if any(word in feedback for word in ("焦虑", "心慌", "担心", "停不下来", "反复想")):
            result = WorkflowResult(
                state="SLEEP_FOLLOWUP",
                message="先不用解决所有担心，把注意力从想法拉回一个可以完成的小动作。",
                action_title="写下明天的第一步，然后停止继续分析",
                action_steps=[
                    "只写一句：明天开始时要做什么",
                    "把这句话放到明日卡片里，不再补充更多细节",
                    "回到床上听 5 分钟白噪音，允许自己只是休息",
                ],
                fallback_used=fallback_used,
            )
            result.action_steps = self._personalized_steps(context, result.action_steps)
            return result
        if any(word in feedback for word in ("嘴馋", "想吃", "饿", "零食", "宵夜", "吃东西")):
            result = WorkflowResult(
                state="SLEEP_FOLLOWUP",
                message="如果只是嘴馋，不必继续搜索更多食物；先让身体和注意力慢下来。",
                action_title="先喝水，再决定是否需要少量食物",
                action_steps=[
                    "先喝几口水，离开屏幕 3 分钟",
                    "如果仍然饿，选择少量、简单的食物并尽快结束进食",
                    "回到低音量白噪音，不再打开新的内容",
                ],
                fallback_used=fallback_used,
            )
            result.action_steps = self._personalized_steps(context, result.action_steps)
            return result
        if any(word in feedback for word in ("手机", "视频", "刷", "聊天", "游戏")):
            result = WorkflowResult(
                state="SLEEP_FOLLOWUP",
                message="现在不需要强迫自己立刻睡着，先把外界输入降到最低。",
                action_title="做一次无屏幕的 5 分钟过渡",
                action_steps=[
                    "把手机屏幕朝下并放远",
                    "保持一个舒服的姿势，慢慢呼吸几轮",
                    "只听白噪音 5 分钟，不再切换内容",
                ],
                fallback_used=fallback_used,
            )
            result.action_steps = self._personalized_steps(context, result.action_steps)
            return result
        result = WorkflowResult(
            state="SLEEP_FOLLOWUP",
            message="还睡不着也不代表今晚失败，可以先把目标改成安静地休息一会儿。",
            action_title="给自己一个 5 分钟的低刺激暂停",
            action_steps=[
                "调低屏幕亮度并把手机放远",
                "选择一个舒服的姿势，不再处理新任务",
                "听 5 分钟低音量白噪音，再判断下一步",
            ],
            fallback_used=fallback_used,
        )
        result.action_steps = self._personalized_steps(context, result.action_steps)
        return result

    @staticmethod
    def _first_profile_value(context: ContextBundle, key: str) -> str:
        values = context.user_profile.get(key, [])
        return values[0] if values else ""

    def _personalized_steps(self, context: ContextBundle, steps: list[str]) -> list[str]:
        rejected = context.user_profile.get("rejected_action", [])
        filtered = [step for step in steps if not any(action in step for action in rejected)]
        sound = self._first_profile_value(context, "sound_preference")
        if sound:
            filtered = [
                step.replace("默认白噪音", sound).replace("白噪音", sound)
                for step in filtered
            ]
        return filtered or ["把外界刺激降到最低，安静休息几分钟"]

    def _apply_profile_rules(
        self,
        context: ContextBundle,
        result: WorkflowResult,
        *,
        prefer_helpful: bool,
    ) -> WorkflowResult:
        rejected = context.user_profile.get("rejected_action", [])
        helpful = self._first_profile_value(context, "helpful_action")
        steps = self._personalized_steps(context, result.action_steps)
        if any(action in result.action_title for action in rejected):
            result.action_title = helpful or "做一个低刺激的小动作"
        rendered = " ".join([result.action_title, *steps])
        if prefer_helpful and helpful and helpful not in rendered:
            steps.insert(0, f"只做一次：{helpful}")
        result.action_steps = steps[:4]
        return result

    def _mock_extract_items(self, text: str) -> list[TodoItem]:
        segments = segment_events(text)[:5]
        return [self._todo_item_for(segment) for segment in segments] or [
            TodoItem(content=text, category="other", suggested_slot="tomorrow", minimum_action="明天再处理这件事")
        ]

    @staticmethod
    def _is_food_related(text: str) -> bool:
        colloquial_craving = "嘴" in text and "馋" in text
        return colloquial_craving or any(term in text for term in ("嘴馋", "想吃", "吃点", "吃东西", "饿", "零食", "宵夜", "外卖"))

    def _merge_related_items(
        self, items: list[TodoItem], preserve_model_boundaries: bool = False
    ) -> list[TodoItem]:
        expanded: list[TodoItem] = []
        splitter = split_explicit_events if preserve_model_boundaries else segment_events
        for item in items:
            for content in splitter(item.content):
                expanded.append(item.model_copy(update={"content": content}))

        groups = group_related_texts([item.content for item in expanded])
        merged: list[TodoItem] = []
        for group in groups:
            group_items = [expanded[index] for index in group]
            content = compose_event([item.content for item in group_items])
            if merged and self._is_food_related(merged[-1].content) and self._is_food_related(content):
                previous = merged[-1]
                previous.content = f"{previous.content}，{content}"
                previous.minimum_action = "先喝一点水；如果仍然饿，再选择少量、简单的食物"
                continue
            merged.append(self._todo_item_for(content, group_items))
        return merged[:5]

    def _todo_item_for(self, content: str, source_items: list[TodoItem] | None = None) -> TodoItem:
        source_items = source_items or []
        category = "worry" if any(
            word in content for word in ("担心", "焦虑", "睡不着", "失眠", "放不下", "紧张", "不安")
        ) else "task"
        if any(word in content for word in ("刷", "游戏", "视频", "聊天")):
            category = "entertainment"
        if self._is_food_related(content):
            category = "other"
        elif category == "task" and source_items:
            categories = {item.category for item in source_items}
            if len(categories) == 1:
                category = source_items[0].category

        slot = "tomorrow" if any(word in content for word in ("明天", "之后", "下周", "后天")) else "tonight"
        if source_items and slot == "tonight" and any(item.suggested_slot == "tomorrow" for item in source_items):
            slot = "tomorrow"
        if self._is_food_related(content):
            action = "先喝一点水；如果仍然饿，再选择少量、简单的食物"
        else:
            action = f"记下“{content[:18]}”的下一步" if slot == "tomorrow" else f"只完成“{content[:18]}”的最小动作"
        return TodoItem(content=content, category=category, suggested_slot=slot, minimum_action=action)

    def _prompt(self, context: ContextBundle, stage: str) -> str:
        payload = context.model_dump() if hasattr(context, "model_dump") else context.dict()
        return (
            "你是睡前收尾助手，只输出简短、温和、非医疗性的内容。不要诊断，不要鼓励用户继续聊天。"
            "只使用上下文中当前有效且与本轮相关的信息。优先采用 user_profile.helpful_action，"
            "不得推荐 user_profile.rejected_action；不要向用户提及系统记忆、画像、置信度或内部评分。"
            f"当前阶段：{stage}\n上下文 JSON：{json.dumps(payload, ensure_ascii=False)}\n"
            "返回 JSON：{message, action_title, action_steps}。"
        )

    def _call_json(self, prompt: str) -> Any:
        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if self.thinking_mode in {"enabled", "disabled", "auto"}:
            payload["thinking"] = {"type": self.thinking_mode}
        try:
            response_payload = self._post_chat_completion(payload)
        except urllib.error.HTTPError as error:
            # Some free routed models do not advertise structured-output support.
            if error.code not in {400, 404, 422}:
                raise
            payload.pop("response_format", None)
            response_payload = self._post_chat_completion(payload)
        content = response_payload["choices"][0]["message"]["content"]
        return self._parse_json_content(content)

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload

    @staticmethod
    def _parse_json_content(content: str) -> Any:
        cleaned = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(cleaned[start:end + 1])
