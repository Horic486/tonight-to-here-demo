from __future__ import annotations

import tempfile
from pathlib import Path

from audio import AudioService
from advice import AdviceService
from config import AUDIO_DIR, KNOWLEDGE_DIR
from context import ContextAssembler
from database import Database
from llm import LLMClient
from memory import MemoryService
from rag import GuidanceRAG
from search import WebSearchClient
from vector_store import LocalVectorStore
from workflow import WorkflowEngine


def run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        database = Database(root / "demo.sqlite3")
        vectors = LocalVectorStore(root / "vectors.json")
        audio = AudioService(database, root / "audio", root / "audio" / "user")
        memories = MemoryService(database, vectors)
        rag = GuidanceRAG(KNOWLEDGE_DIR, vectors)
        context = ContextAssembler(database, memories, rag)
        advice = AdviceService(context, rag, LLMClient("mock"), WebSearchClient())
        workflow = WorkflowEngine(database, memories, context, LLMClient("mock"), audio, advice)
        user_id = "cli-user"
        preference = audio.preference(user_id)
        session_id = workflow.start(user_id)
        items = workflow.capture(session_id, "明天上午要开会，今晚还有一封邮件没回，我有点放不下。")
        workflow.triage(session_id, {index: "tomorrow" for index in range(len(items))})
        card = workflow.tomorrow_plan(session_id)
        transition = workflow.wind_down(session_id, user_id, preference)
        closure = workflow.close(session_id, user_id, preference)
        print("session:", session_id)
        print("items:", [item.content for item in items])
        print("tomorrow:", card)
        print("transition:", transition.message)
        print("closure:", closure)
        print("audio:", preference.default_audio_id)
        database.close()


if __name__ == "__main__":
    run()
