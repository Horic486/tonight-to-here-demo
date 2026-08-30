from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from advice import AdviceService
from audio import AudioService
from config import AUDIO_DIR, DB_PATH, KNOWLEDGE_DIR, USER_AUDIO_DIR, VECTOR_PATH, ensure_directories, model_mode
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from models import AudioPreference, ConversationRound, WorkflowResult
from rag import GuidanceRAG
from search import WebSearchClient
from vector_store import LocalVectorStore
from workflow import WorkflowEngine


FLOW_STATE_KEYS = {
    "session_id",
    "round_id",
    "step",
    "items",
    "followup_count",
    "transition_result",
    "followup_result",
    "tonight_action_result",
    "tomorrow_card",
    "closure",
    "still_awake",
    "sleep_feedback",
    "allow_web_search",
    "initial_feeling_input",
    "concern_input",
    "history_date_select",
    "history_round_select",
}

HONG_KONG_TIMEZONE = ZoneInfo("Asia/Hong_Kong")
UI_BACKGROUND_PATH = Path(__file__).parent / "data" / "ui" / "night-bedroom-v2.jpg"


@st.cache_data
def _night_background_data_uri() -> str:
    if not UI_BACKGROUND_PATH.exists():
        return ""
    encoded = base64.b64encode(UI_BACKGROUND_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def apply_night_theme() -> None:
    background_uri = _night_background_data_uri()
    background_image = f"url('{background_uri}')" if background_uri else "none"
    st.markdown(
        f"""
        <style>
        :root {{
            --night-bg: #090a12;
            --night-surface: rgba(16, 17, 29, 0.82);
            --night-surface-strong: rgba(12, 13, 23, 0.94);
            --night-border: rgba(226, 220, 238, 0.16);
            --night-text: #f3f0f5;
            --night-muted: #b7b2bf;
            --night-coral: #ee8f78;
            --night-coral-hover: #f3a08b;
            --night-violet: #aaa0d7;
            --night-green: #91b8a6;
        }}

        html, body, #root, [data-testid="stApp"], [class*="css"] {{
            font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
            letter-spacing: 0;
        }}

        html, body, #root, [data-testid="stApp"] {{
            min-height: 100%;
            background: var(--night-bg) !important;
        }}

        [data-testid="stAppViewContainer"] {{
            color: var(--night-text);
            background-color: var(--night-bg);
            background-image:
                linear-gradient(90deg, rgba(8, 9, 17, 0.96) 0%, rgba(8, 9, 17, 0.90) 42%, rgba(8, 9, 17, 0.62) 100%),
                {background_image};
            background-position: center, center right;
            background-size: cover, cover;
            background-attachment: fixed;
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: linear-gradient(90deg, rgba(9, 10, 18, 0.52), rgba(9, 10, 18, 0.14));
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stToolbar"] button,
        [data-testid="stHeaderActionElements"] button {{
            color: var(--night-text);
        }}

        .stMainBlockContainer {{
            width: min(100%, 820px);
            max-width: 820px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
        }}

        .tonight-brand {{
            padding: 0.25rem 0 1.5rem;
            border-bottom: 1px solid var(--night-border);
            margin-bottom: 1.8rem;
        }}

        .tonight-brand__eyebrow {{
            color: var(--night-coral);
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 0.4rem;
        }}

        .tonight-brand__title {{
            color: var(--night-text);
            font-size: 2.7rem;
            font-weight: 650;
            line-height: 1.16;
            margin: 0;
        }}

        .tonight-brand__subtitle {{
            color: var(--night-muted);
            font-size: 0.94rem;
            line-height: 1.7;
            margin-top: 0.55rem;
        }}

        .tonight-stage {{
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.35rem 0.7rem;
            margin: 0 0 0.9rem;
            color: #ddd7e8;
            font-size: 0.78rem;
            font-weight: 650;
            background: rgba(170, 160, 215, 0.12);
            border: 1px solid rgba(170, 160, 215, 0.24);
            border-radius: 6px;
        }}

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        label, .stSelectbox label, .stTextInput label, .stTextArea label {{
            color: var(--night-text);
        }}

        h3 {{
            font-size: 1.45rem !important;
            font-weight: 640 !important;
            line-height: 1.35 !important;
            margin-top: 0.2rem !important;
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        small {{
            color: var(--night-muted) !important;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(10, 11, 20, 0.93);
            border-right: 1px solid var(--night-border);
            backdrop-filter: blur(18px);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding: 1.5rem 1.25rem 2.5rem;
        }}

        [data-testid="stSidebar"] h3 {{
            color: var(--night-text);
            font-size: 1.05rem !important;
            margin-top: 1.25rem !important;
        }}

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            font-size: 0.78rem;
            line-height: 1.6;
        }}

        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {{
            color: var(--night-text) !important;
            background: var(--night-surface) !important;
            border-color: var(--night-border) !important;
            border-radius: 7px !important;
        }}

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span {{
            color: var(--night-text) !important;
            caret-color: var(--night-coral) !important;
        }}

        input, textarea {{
            color: var(--night-text) !important;
            background: rgba(22, 23, 36, 0.9) !important;
        }}

        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within {{
            border-color: rgba(238, 143, 120, 0.82) !important;
            box-shadow: 0 0 0 1px rgba(238, 143, 120, 0.45) !important;
        }}

        [data-baseweb="popover"] ul,
        [role="listbox"] {{
            background: var(--night-surface-strong) !important;
            border: 1px solid var(--night-border) !important;
        }}

        [role="option"] {{
            color: var(--night-text) !important;
        }}

        [role="option"]:hover {{
            background: rgba(238, 143, 120, 0.14) !important;
        }}

        .stButton > button,
        [data-testid="stFileUploaderDropzone"] button {{
            min-height: 2.65rem;
            color: var(--night-text);
            background: rgba(28, 29, 43, 0.84);
            border: 1px solid rgba(232, 226, 240, 0.24);
            border-radius: 7px;
            font-weight: 650;
            transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease;
        }}

        .stButton > button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {{
            color: #fff;
            background: rgba(46, 46, 64, 0.94);
            border-color: rgba(238, 143, 120, 0.65);
            transform: translateY(-1px);
        }}

        .stButton > button[kind="primary"] {{
            color: #1a1010;
            background: var(--night-coral);
            border-color: var(--night-coral);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
        }}

        .stButton > button[kind="primary"]:hover {{
            color: #160d0d;
            background: var(--night-coral-hover);
            border-color: var(--night-coral-hover);
        }}

        [data-testid="stAlert"] {{
            color: var(--night-text);
            background: rgba(18, 20, 31, 0.88);
            border: 1px solid var(--night-border);
            border-left: 3px solid var(--night-violet);
            border-radius: 7px;
        }}

        [data-testid="stAlert"] p {{
            color: var(--night-text) !important;
        }}

        [data-testid="stWidgetLabel"] p,
        [data-testid="stSlider"] p,
        [data-testid="stCheckbox"] p {{
            color: var(--night-text) !important;
        }}

        [data-testid="stSlider"] [role="slider"] {{
            background: var(--night-coral) !important;
            border-color: var(--night-coral) !important;
        }}

        [data-testid="stCheckbox"] svg {{
            color: var(--night-text);
        }}

        [data-testid="stAudio"] {{
            border-radius: 7px;
            overflow: hidden;
            opacity: 0.9;
            color-scheme: dark;
            filter: invert(0.88) hue-rotate(175deg) saturate(0.7);
        }}

        [data-testid="stHorizontalBlock"] {{
            gap: 0.8rem;
        }}

        hr {{
            border-color: var(--night-border) !important;
        }}

        a {{
            color: #c7bee8 !important;
        }}

        @media (max-width: 760px) {{
            [data-testid="stAppViewContainer"] {{
                background-image:
                    linear-gradient(rgba(8, 9, 17, 0.86), rgba(8, 9, 17, 0.94)),
                    {background_image};
                background-position: center, 62% center;
                background-attachment: scroll;
            }}

            .stMainBlockContainer {{
                width: 100%;
                padding: 1.4rem 1.1rem 3rem;
            }}

            .tonight-brand {{
                padding-top: 1rem;
                margin-bottom: 1.4rem;
            }}

            .tonight-brand__title {{
                font-size: 2rem;
            }}

            [data-testid="stSidebar"] > div:first-child {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                scroll-behavior: auto !important;
                transition: none !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    st.markdown(
        """
        <div class="tonight-brand">
            <div class="tonight-brand__eyebrow">SLEEP WELL TONIGHT</div>
            <div class="tonight-brand__title">今晚到此</div>
            <div class="tonight-brand__subtitle">把今天轻轻放下，慢慢靠近睡意。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_marker(step: str) -> None:
    labels = {
        "capture": "01 · 写下挂念",
        "triage": "02 · 安排去处",
        "wind_down": "03 · 慢慢放松",
        "tonight_action": "03 · 完成最小动作",
        "close": "04 · 今晚到此",
        "history": "过往的夜晚",
    }
    label = labels.get(step)
    if label:
        st.markdown(f'<div class="tonight-stage">{label}</div>', unsafe_allow_html=True)


def user_id_from_identity(identity: str) -> str:
    identity = identity.strip()
    if not identity:
        raise ValueError("记忆空间标识不能为空")
    return "user-" + hashlib.sha256(f"tonight-to-here:{identity}".encode("utf-8")).hexdigest()


def reset_flow_state() -> None:
    for key in list(st.session_state):
        if (
            key in FLOW_STATE_KEYS
            or key.startswith("slot_")
            or key.startswith("history_round_select_")
        ):
            st.session_state.pop(key, None)


def browser_space_token(memory_space_code: str) -> str:
    if memory_space_code.strip():
        token = "code-" + hashlib.sha256(memory_space_code.strip().encode("utf-8")).hexdigest()
    else:
        current = str(st.query_params.get("space", ""))
        valid = current.startswith(("anon-", "code-")) and all(
            character.isalnum() or character == "-" for character in current
        )
        token = current if valid else f"anon-{uuid.uuid4().hex}"
    if st.query_params.get("space") != token:
        st.query_params["space"] = token
    return token


def _run_workflow_step(action):
    try:
        return action()
    except (sqlite3.DatabaseError, RuntimeError, ValueError) as exc:
        st.error(f"本次操作未能完整保存，请稍后重试。已经写入的内容不会被删除。详情：{exc}")
        return None


def begin_new_round(workflow: WorkflowEngine, user_id: str) -> bool:
    started = _run_workflow_step(lambda: workflow.start_round(user_id))
    if started is None:
        return False
    session_id, round_id = started
    reset_flow_state()
    st.session_state.session_id = session_id
    st.session_state.round_id = round_id
    st.session_state.step = "capture"
    st.session_state["items"] = []
    st.session_state["followup_count"] = 0
    return True


def restore_round_state(
    database: Database, user_id: str, history: ConversationRound
) -> None:
    session = database.get_session_for_user(user_id, history.session_id)
    if not session:
        raise ValueError("找不到当前用户的睡前会话")
    st.session_state.session_id = history.session_id
    st.session_state.round_id = history.round_id
    st.session_state["items"] = history.items
    st.session_state["tomorrow_card"] = history.tomorrow_card
    st.session_state["followup_count"] = len(history.followup_feedback)
    if history.wind_down_advice:
        st.session_state["transition_result"] = WorkflowResult(**history.wind_down_advice)
    if history.followup_advice:
        st.session_state["followup_result"] = WorkflowResult(**history.followup_advice[-1])
    if history.tonight_action:
        st.session_state["tonight_action_result"] = WorkflowResult(**history.tonight_action)
    if history.closure_message:
        st.session_state["closure"] = history.closure_message

    if history.status == "completed":
        st.session_state.step = "tonight_action" if history.tonight_action else "close"
        return
    state = session["state"] if session else "CHECK_IN"
    st.session_state.step = {
        "CHECK_IN": "capture",
        "CAPTURE": "triage",
        "TRIAGE": "triage",
        "TOMORROW_PLAN": "wind_down",
        "WIND_DOWN": "wind_down",
        "SLEEP_FOLLOWUP": "wind_down",
        "TONIGHT_ACTION": "tonight_action",
        "CLOSE": "close",
    }.get(state, "capture")


def _history_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(HONG_KONG_TIMEZONE).strftime("%H:%M")
    except (TypeError, ValueError):
        return ""


def _render_advice(advice: dict, title: str) -> None:
    if not advice:
        return
    st.write(title)
    if advice.get("message"):
        st.write(advice["message"])
    if advice.get("action_title"):
        st.info(advice["action_title"])
    for step in advice.get("action_steps", []):
        st.write(f"- {step}")
    mode_labels = {
        "api": "智能生成",
        "mock": "本地离线建议",
        "local": "本地规则",
        "local_fallback": "本地回退建议",
    }
    generation_mode = advice.get("generation_mode")
    if generation_mode in mode_labels:
        st.caption(f"生成方式：{mode_labels[generation_mode]}")
    if advice.get("web_used"):
        st.caption("本次使用了联网检索资料。")
    elif advice.get("allow_web"):
        st.caption("本次允许联网检索，但最终未使用联网资料。")


def render_history(database: Database, user_id: str) -> None:
    st.subheader("历史记录")
    dates = database.list_history_dates(user_id)
    if not dates:
        st.info("还没有完成或正在进行的睡前收尾记录。")
        return

    summaries = {item["local_date"]: item for item in dates}
    selected_date = st.selectbox(
        "日期",
        [item["local_date"] for item in dates],
        format_func=lambda value: (
            f"{value} · {summaries[value]['round_count']} 轮 · {summaries[value]['summary']}"
        ),
        key="history_date_select",
    )
    rounds = database.list_history_rounds(user_id, selected_date)
    status_labels = {"active": "进行中", "completed": "已完成", "abandoned": "未完成"}
    selected_round_id = st.selectbox(
        "当日轮次",
        [history.round_id for history in rounds],
        format_func=lambda round_id: (
            f"第 {next(item for item in rounds if item.round_id == round_id).round_index} 轮 · "
            f"{status_labels.get(next(item for item in rounds if item.round_id == round_id).status, '未完成')} · "
            f"{_history_time(next(item for item in rounds if item.round_id == round_id).started_at)}"
        ),
        key=f"history_round_select_{selected_date}",
    )
    history = next(item for item in rounds if item.round_id == selected_round_id)

    st.divider()
    st.caption(
        f"第 {history.round_index} 轮 · {_history_time(history.started_at)} 开始"
        + (f" · {_history_time(history.completed_at)} 结束" if history.completed_at else "")
    )
    if history.initial_feeling:
        st.write("当时的感受")
        st.write(history.initial_feeling)
    st.write("挂念的事情")
    st.write(history.concern_input)

    if history.items:
        st.write("事项与安排")
        slot_labels = {
            "tonight": "今晚只做最小动作",
            "tomorrow": "明天处理",
            "later": "暂时放下",
        }
        for index, item in enumerate(history.items):
            arrangement = history.arrangements[index] if index < len(history.arrangements) else {}
            slot = arrangement.get("slot", item.suggested_slot)
            st.write(f"- {item.content} · {slot_labels.get(slot, '尚未安排')}")

    if history.tomorrow_card:
        st.write("明日第一步")
        st.success(history.tomorrow_card)
    _render_advice(history.wind_down_advice, "助眠行动建议")
    _render_advice(history.tonight_action, "今晚的最小动作")

    for index, feedback in enumerate(history.followup_feedback):
        st.divider()
        st.write(f"第 {index + 1} 次睡不着反馈")
        st.write(feedback.get("text", ""))
        if index < len(history.followup_advice):
            _render_advice(history.followup_advice[index], "当时给出的下一步")

    if history.closure_message:
        st.divider()
        st.write("结束语")
        st.write(history.closure_message)


@st.cache_resource
def services() -> tuple[Database, AudioService, WorkflowEngine]:
    ensure_directories()
    database = Database(DB_PATH)
    vectors = LocalVectorStore(VECTOR_PATH)
    memories = MemoryService(database, vectors)
    rag = GuidanceRAG(KNOWLEDGE_DIR, vectors)
    audio = AudioService(database, AUDIO_DIR, USER_AUDIO_DIR)
    context = ContextAssembler(database, memories, rag)
    llm = LLMClient(model_mode())
    advice = AdviceService(context, rag, llm, WebSearchClient())
    return database, audio, WorkflowEngine(database, memories, context, llm, audio, advice)


def play_audio(
    audio: AudioService,
    preference: AudioPreference,
    user_id: str,
    fade_timer_active: bool,
) -> None:
    assets = {asset["audio_id"]: asset for asset in audio.catalog(user_id)}
    asset = assets.get(preference.default_audio_id) or next(iter(assets.values()), None)
    if not asset:
        return
    path = audio.path_for(asset)
    mime_type = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(path.suffix.lower(), "application/octet-stream")
    st.caption(f"正在播放：{asset['title']} · 默认音量 {int(preference.volume * 100)}%")
    st.audio(
        path,
        format=mime_type,
        loop=True,
        autoplay=preference.autoplay_enabled,
    )
    html = f"""
    <script>
      (() => {{
        const configurePlayer = () => {{
          const players = document.querySelectorAll("audio");
          const player = players[players.length - 1];
          if (!player) {{
            window.requestAnimationFrame(configurePlayer);
            return;
          }}
          window.clearTimeout(player._tonightFadeTimeout);
          window.clearInterval(player._tonightFadeInterval);
          player.loop = true;
          player.volume = {preference.volume:.2f};
          if ({str(preference.autoplay_enabled).lower()}) {{
            player.play().catch(() => {{}});
          }}
          const fadeTimerActive = {str(fade_timer_active).lower()};
          const fadeOutMinutes = {preference.fade_out_minutes};
          player.dataset.tonightConfigured = "true";
          player.dataset.tonightFadeTimerActive = String(fadeTimerActive);
          player.dataset.tonightFadeOutMinutes = String(fadeOutMinutes);
          if (fadeTimerActive && fadeOutMinutes > 0) {{
            player._tonightFadeTimeout = window.setTimeout(() => {{
              const initialVolume = player.volume;
              const fadeDurationMs = 10000;
              const startedAt = Date.now();
              player._tonightFadeInterval = window.setInterval(() => {{
                const progress = Math.min(1, (Date.now() - startedAt) / fadeDurationMs);
                player.volume = initialVolume * (1 - progress);
                if (progress >= 1) {{
                  window.clearInterval(player._tonightFadeInterval);
                  player.pause();
                }}
              }}, 250);
            }}, fadeOutMinutes * 60 * 1000);
          }}
        }};
        configurePlayer();
      }})();
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)
    if preference.autoplay_enabled:
        st.caption("默认音量较低；若浏览器拦截自动播放，请点击播放器的播放键。")


def main() -> None:
    st.set_page_config(page_title="今晚到此", page_icon="🌙", layout="centered")
    apply_night_theme()
    try:
        database, audio, workflow = services()
    except (sqlite3.DatabaseError, RuntimeError) as exc:
        st.error(f"应用数据初始化失败：{exc}")
        return

    render_brand_header()

    space_token = browser_space_token("")
    user_id = user_id_from_identity(space_token)
    if st.session_state.get("active_user_id") != user_id:
        reset_flow_state()
        st.session_state.active_user_id = user_id
        st.session_state.current_view = "flow"
    if "current_view" not in st.session_state:
        st.session_state.current_view = "flow"
    preference = audio.preference(user_id)

    with st.sidebar:
        if st.session_state.current_view == "history":
            if st.button("返回当前收尾", use_container_width=True):
                st.session_state.current_view = "flow"
                st.rerun()
        elif st.button("历史记录", use_container_width=True):
            st.session_state.current_view = "history"
            st.rerun()

        st.subheader("今晚的声音")
        fade_timer_active = st.session_state.get("step") in {"close", "tonight_action"}
        play_audio(audio, preference, user_id, fade_timer_active)
        assets = audio.catalog(user_id)
        labels = {asset["title"]: asset for asset in assets}
        current = next((asset["title"] for asset in assets if asset["audio_id"] == preference.default_audio_id), assets[0]["title"])
        selected_title = st.selectbox("进入 App 时播放", list(labels), index=list(labels).index(current))
        volume = st.slider("默认音量", 0.0, 1.0, float(preference.volume), 0.01)
        autoplay = st.checkbox("进入时自动播放", value=preference.autoplay_enabled)
        fade_minutes = st.slider(
            "结束后自动淡出（分钟）",
            0,
            120,
            int(preference.fade_out_minutes),
            1,
            help="从结束今天开始计时；0 表示不自动停止。",
        )
        if st.button("保存声音偏好"):
            selected = labels[selected_title]
            audio.save_preference(AudioPreference(
                user_id=user_id, default_audio_id=selected["audio_id"], volume=volume,
                autoplay_enabled=autoplay, fade_out_minutes=fade_minutes,
            ))
            st.success("已保存，下次进入 App 将播放这段声音。")
        uploaded = st.file_uploader("导入我的音频", type=["wav", "mp3", "ogg", "m4a"])
        if uploaded is not None and st.button("保存导入音频"):
            try:
                audio.upload(user_id, uploaded.name, uploaded.getvalue())
                st.success("音频已加入你的目录。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if st.session_state.current_view == "history":
        render_stage_marker("history")
        try:
            render_history(database, user_id)
        except Exception as exc:
            st.error(f"暂时无法读取历史记录：{exc}")
        return

    if "session_id" not in st.session_state:
        latest_round = database.latest_history_round(user_id)
        if latest_round:
            restore_round_state(database, user_id, latest_round)
        else:
            if not begin_new_round(workflow, user_id):
                return

    render_stage_marker(st.session_state.step)

    if st.session_state.step == "capture":
        st.subheader("现在还挂念着什么？")
        feeling = st.text_input("此刻的感受（可选）", key="initial_feeling_input")
        text = st.text_area(
            "可以写工作、明天的安排，或者只是说说现在为什么不想睡。",
            height=130,
            key="concern_input",
        )
        if st.button("开始收尾", type="primary"):
            if not text.strip():
                st.warning("先写下一件脑中放不下的事。")
            else:
                concern = text.strip()
                initial_feeling = feeling.strip()
                current_round = database.get_history_round(
                    user_id, st.session_state.round_id
                )
                if (
                    current_round is None
                    or current_round.local_date != database.current_local_date()
                ):
                    if not begin_new_round(workflow, user_id):
                        return
                items = _run_workflow_step(
                    lambda: workflow.capture(
                        st.session_state.session_id,
                        concern,
                        round_id=st.session_state.round_id,
                        initial_feeling=initial_feeling,
                    )
                )
                if items is None:
                    return
                st.session_state["items"] = items
                st.session_state.step = "triage"
                st.rerun()

    elif st.session_state.step == "triage":
        st.subheader("给每件事一个去处")
        choices = {}
        for index, item in enumerate(st.session_state["items"]):
            slot = st.selectbox(
                item.content,
                ["tonight", "tomorrow", "later"],
                index=["tonight", "tomorrow", "later"].index(item.suggested_slot),
                format_func={"tonight": "今晚只做最小动作", "tomorrow": "明天处理", "later": "暂时放下"}.get,
                key=f"slot_{index}",
            )
            choices[index] = slot
        if st.button("确认安排", type="primary"):
            items = _run_workflow_step(
                lambda: workflow.triage(
                    st.session_state.session_id,
                    choices,
                    round_id=st.session_state.round_id,
                )
            )
            if items is None:
                return
            st.session_state["items"] = items
            tomorrow_card = _run_workflow_step(
                lambda: workflow.tomorrow_plan(
                    st.session_state.session_id,
                    round_id=st.session_state.round_id,
                )
            )
            if tomorrow_card is None:
                return
            st.session_state.tomorrow_card = tomorrow_card
            st.session_state["followup_count"] = 0
            st.session_state.pop("transition_result", None)
            st.session_state.pop("followup_result", None)
            if any(item.suggested_slot == "tonight" for item in st.session_state["items"]):
                tonight_action = _run_workflow_step(
                    lambda: workflow.finish_with_tonight_actions(
                        st.session_state.session_id,
                        user_id,
                        preference,
                        round_id=st.session_state.round_id,
                    )
                )
                if tonight_action is None:
                    return
                st.session_state["tonight_action_result"] = tonight_action
                st.session_state.step = "tonight_action"
            else:
                st.session_state.pop("tonight_action_result", None)
                st.session_state.step = "wind_down"
            st.rerun()

    elif st.session_state.step == "wind_down":
        if "transition_result" not in st.session_state:
            transition = _run_workflow_step(
                lambda: workflow.wind_down(
                    st.session_state.session_id,
                    user_id,
                    preference,
                    round_id=st.session_state.round_id,
                )
            )
            if transition is None:
                return
            st.session_state["transition_result"] = transition
        result = st.session_state["transition_result"]
        st.subheader("现在，慢一点")
        st.write(result.message)
        st.info(result.action_title)
        for step in result.action_steps:
            st.write(f"- {step}")
        st.caption("建议先给这个动作 5 到 10 分钟；如果仍然清醒，可以反馈一次现在的状态。")

        followup = st.session_state.get("followup_result")
        if followup:
            st.divider()
            st.subheader(f"第 {followup.round_index} 次状态反馈")
            st.write(followup.message)
            st.info(followup.action_title)
            for step in followup.action_steps:
                st.write(f"- {step}")
            if followup.sources:
                st.caption("参考资料：" + "；".join(followup.sources[:3]))
            if followup.fallback_used:
                st.caption("本轮检索或模型不可用，已切换到通用行动建议。")

        if st.session_state.get("followup_count", 0) < 2:
            still_awake = st.checkbox("还是睡不着？", key="still_awake")
            if still_awake:
                feedback = st.text_area(
                    "现在的感受或状态",
                    placeholder="例如：脑子还在反复想明天的事情，身体不困，但不想继续刷手机。",
                    height=100,
                    key="sleep_feedback",
                )
                allow_web = st.checkbox(
                    "允许本次联网检索（可选）",
                    value=False,
                    help="只会发送清洗后的简短主题词；关闭时仅使用本地资料。",
                    key="allow_web_search",
                )
                if allow_web:
                    st.caption("联网不可用时会自动回退到本地资料和通用建议。")
                if st.button("根据现在状态给我下一步建议"):
                    if not feedback.strip():
                        st.warning("先写下现在的感受或状态。")
                    else:
                        with st.spinner("正在整理更适合这一刻的建议..."):
                            followup = _run_workflow_step(
                                lambda: workflow.follow_up(
                                    st.session_state.session_id,
                                    user_id,
                                    preference,
                                    feedback,
                                    round_index=st.session_state.get("followup_count", 0) + 1,
                                    allow_web=allow_web,
                                    round_id=st.session_state.round_id,
                                )
                            )
                        if followup is None:
                            return
                        st.session_state["followup_result"] = followup
                        st.session_state["followup_count"] = followup.round_index
                        st.rerun()
        else:
            st.caption("已经完成两轮状态反馈，可以继续播放白噪音，或结束今天。")

        st.divider()
        st.write("明日卡片")
        st.success(st.session_state.get("tomorrow_card", "明天从最重要的一步开始。"))
        if st.button("结束今天", type="primary"):
            closure = _run_workflow_step(
                lambda: workflow.close(
                    st.session_state.session_id,
                    user_id,
                    preference,
                    round_id=st.session_state.round_id,
                )
            )
            if closure is None:
                return
            st.session_state.closure = closure
            st.session_state.step = "close"
            st.rerun()

    elif st.session_state.step == "tonight_action":
        result = st.session_state["tonight_action_result"]
        st.subheader("现在去做吧")
        st.write(result.message)
        st.info(result.action_title)
        for step in result.action_steps:
            st.write(f"- {step}")
        st.success("做到能够放下就停。这一轮到此。")
        if st.button("开启下一次收尾", type="primary"):
            if begin_new_round(workflow, user_id):
                st.rerun()

    else:
        st.subheader("今晚到此")
        st.write(st.session_state.get("closure", "剩下的事情，明天再处理。"))
        if preference.fade_out_minutes > 0:
            st.caption(f"白噪音将在 {preference.fade_out_minutes} 分钟后自动淡出，也可以手动停止。")
        else:
            st.caption("白噪音会继续播放，直到你手动停止。")
        if st.button("开启下一次收尾"):
            if begin_new_round(workflow, user_id):
                st.rerun()


if __name__ == "__main__":
    main()
