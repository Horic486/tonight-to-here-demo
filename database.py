from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

from models import (
    DEFAULT_AUDIO_VOLUME,
    AudioPreference,
    ConversationRound,
    LEGACY_DEFAULT_AUDIO_VOLUME,
    MemoryItem,
    TodoItem,
    UserProfileFact,
    utc_now,
)


def _synchronized(method):
    """Serialize access to the shared SQLite connection used by the demo."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class Database:
    def __init__(
        self,
        path: str | Path,
        clock: Optional[Callable[[], datetime]] = None,
        timezone_name: str = "Asia/Hong_Kong",
    ):
        self.path = str(path)
        self._lock = threading.RLock()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.local_timezone = ZoneInfo(timezone_name)
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.initialize()

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _now_text(self) -> str:
        return self._now().isoformat()

    def now_text(self) -> str:
        return self._now_text()

    def current_local_date(self) -> str:
        return self._now().astimezone(self.local_timezone).date().isoformat()

    def _local_date_for_timestamp(self, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(self.local_timezone).date().isoformat()
        except (TypeError, ValueError):
            return self.current_local_date()

    @_synchronized
    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                local_date TEXT,
                state TEXT NOT NULL,
                today_input TEXT DEFAULT '',
                items_json TEXT DEFAULT '[]',
                tomorrow_card TEXT DEFAULT '',
                transition_json TEXT DEFAULT '{}',
                closure_message TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                round_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'system_inference',
                source_ref TEXT,
                confidence REAL NOT NULL,
                importance REAL NOT NULL DEFAULT 0.6,
                consent INTEGER NOT NULL,
                source_session_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                valid_from TEXT NOT NULL,
                valid_until TEXT,
                half_life_days REAL NOT NULL DEFAULT 90.0,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                memory_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            );
            CREATE TABLE IF NOT EXISTS user_profile_facts (
                profile_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                profile_value TEXT NOT NULL,
                source_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_count INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                valid_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audio_assets (
                audio_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                file_name TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT,
                duration_seconds INTEGER NOT NULL,
                loopable INTEGER NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audio_preferences (
                user_id TEXT PRIMARY KEY,
                default_audio_id TEXT NOT NULL,
                volume REAL NOT NULL,
                autoplay_enabled INTEGER NOT NULL,
                fade_out_seconds INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                round_id TEXT,
                from_state TEXT,
                to_state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_rounds (
                round_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                local_date TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                initial_feeling TEXT NOT NULL DEFAULT '',
                concern_input TEXT NOT NULL DEFAULT '',
                items_json TEXT NOT NULL DEFAULT '[]',
                arrangements_json TEXT NOT NULL DEFAULT '[]',
                tomorrow_card TEXT NOT NULL DEFAULT '',
                wind_down_advice_json TEXT NOT NULL DEFAULT '{}',
                followup_feedback_json TEXT NOT NULL DEFAULT '[]',
                followup_advice_json TEXT NOT NULL DEFAULT '[]',
                tonight_action_json TEXT NOT NULL DEFAULT '{}',
                closure_message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()
        self._migrate_structured_memory()
        self._migrate_conversation_history()
        self._migrate_legacy_audio_volume()
        self._migrate_audio_fade_out_minutes()

    def _migrate_structured_memory(self) -> None:
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(memory_items)")
        }
        additions = {
            "source_type": "TEXT NOT NULL DEFAULT 'system_inference'",
            "source_ref": "TEXT",
            "importance": "REAL NOT NULL DEFAULT 0.6",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "valid_from": "TEXT NOT NULL DEFAULT ''",
            "valid_until": "TEXT",
            "half_life_days": "REAL NOT NULL DEFAULT 90.0",
            "evidence_count": "INTEGER NOT NULL DEFAULT 1",
            "memory_key": "TEXT",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
        }
        try:
            for name, definition in additions.items():
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE memory_items ADD COLUMN {name} {definition}"
                    )
            self.connection.execute(
                "UPDATE memory_items SET valid_from = created_at WHERE valid_from = ''"
            )
            self.connection.execute(
                "UPDATE memory_items SET updated_at = created_at WHERE updated_at = ''"
            )
            self.connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_user_status_valid_until
                    ON memory_items(user_id, status, valid_until);
                CREATE INDEX IF NOT EXISTS idx_memory_user_key_status
                    ON memory_items(user_id, memory_key, status);
                CREATE INDEX IF NOT EXISTS idx_profile_user_status_key_valid_until
                    ON user_profile_facts(user_id, status, profile_key, valid_until);
                """
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO app_metadata(key, value) VALUES (?, ?)",
                ("structured_memory_v1", "completed"),
            )
            self.connection.commit()
        except sqlite3.DatabaseError as exc:
            self.connection.rollback()
            raise RuntimeError(f"长期记忆数据库迁移失败：{exc}") from exc

    def _migrate_conversation_history(self) -> None:
        migration_key = "conversation_history_v1"
        try:
            table_columns = {
                "sessions": {row["name"] for row in self.connection.execute("PRAGMA table_info(sessions)")},
                "turns": {row["name"] for row in self.connection.execute("PRAGMA table_info(turns)")},
                "workflow_events": {
                    row["name"]
                    for row in self.connection.execute("PRAGMA table_info(workflow_events)")
                },
            }
            if "local_date" not in table_columns["sessions"]:
                self.connection.execute("ALTER TABLE sessions ADD COLUMN local_date TEXT")
            if "round_id" not in table_columns["turns"]:
                self.connection.execute("ALTER TABLE turns ADD COLUMN round_id TEXT")
            if "round_id" not in table_columns["workflow_events"]:
                self.connection.execute("ALTER TABLE workflow_events ADD COLUMN round_id TEXT")

            sessions = self.connection.execute(
                "SELECT * FROM sessions WHERE local_date IS NULL OR local_date = ''"
            ).fetchall()
            for session in sessions:
                local_date = self._local_date_for_timestamp(session["created_at"])
                self.connection.execute(
                    "UPDATE sessions SET local_date = ? WHERE session_id = ?",
                    (local_date, session["session_id"]),
                )

            migrated = self.connection.execute(
                "SELECT 1 FROM app_metadata WHERE key = ?", (migration_key,)
            ).fetchone()
            if not migrated:
                legacy_sessions = self.connection.execute(
                    "SELECT * FROM sessions ORDER BY created_at"
                ).fetchall()
                for session in legacy_sessions:
                    round_id = f"legacy-{session['session_id']}"
                    exists = self.connection.execute(
                        "SELECT 1 FROM conversation_rounds WHERE round_id = ?", (round_id,)
                    ).fetchone()
                    if exists:
                        continue
                    items = self._safe_json(session["items_json"], [])
                    arrangements = [
                        {
                            "content": str(item.get("content", "")),
                            "slot": str(item.get("suggested_slot", "tomorrow")),
                            "minimum_action": str(item.get("minimum_action", "")),
                        }
                        for item in items
                        if isinstance(item, dict)
                    ]
                    transition = self._safe_json(session["transition_json"], {})
                    state = session["state"]
                    completed = state in {"CLOSE", "TONIGHT_ACTION"}
                    status = "completed" if completed else "abandoned"
                    wind_down = transition if state != "TONIGHT_ACTION" else {}
                    tonight_action = transition if state == "TONIGHT_ACTION" else {}
                    completed_at = session["updated_at"] if completed else None
                    self.connection.execute(
                        """INSERT OR IGNORE INTO conversation_rounds
                        (round_id, session_id, user_id, local_date, round_index, started_at,
                         completed_at, status, initial_feeling, concern_input, items_json,
                         arrangements_json, tomorrow_card, wind_down_advice_json,
                         followup_feedback_json, followup_advice_json, tonight_action_json,
                         closure_message, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            round_id,
                            session["session_id"],
                            session["user_id"],
                            session["local_date"],
                            1,
                            session["created_at"],
                            completed_at,
                            status,
                            session["today_input"],
                            session["today_input"],
                            json.dumps(items, ensure_ascii=False),
                            json.dumps(arrangements, ensure_ascii=False),
                            session["tomorrow_card"],
                            json.dumps(wind_down, ensure_ascii=False),
                            "[]",
                            "[]",
                            json.dumps(tonight_action, ensure_ascii=False),
                            session["closure_message"],
                            session["created_at"],
                            session["updated_at"],
                        ),
                    )
                self.connection.execute(
                    "INSERT OR IGNORE INTO app_metadata(key, value) VALUES (?, ?)",
                    (migration_key, "completed"),
                )

            self.connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_local_date
                    ON sessions(user_id, local_date, updated_at);
                CREATE INDEX IF NOT EXISTS idx_rounds_user_date_status
                    ON conversation_rounds(user_id, local_date, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_rounds_session_index
                    ON conversation_rounds(session_id, round_index);
                CREATE INDEX IF NOT EXISTS idx_rounds_user_round
                    ON conversation_rounds(user_id, round_id);
                CREATE INDEX IF NOT EXISTS idx_turns_session_round
                    ON turns(session_id, round_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_session_round
                    ON workflow_events(session_id, round_id, created_at);
                """
            )
            self.connection.commit()
        except sqlite3.DatabaseError as exc:
            self.connection.rollback()
            raise RuntimeError(f"历史会话数据库迁移失败：{exc}") from exc

    @staticmethod
    def _safe_json(value: str, fallback: Any) -> Any:
        try:
            parsed = json.loads(value or "")
        except (TypeError, json.JSONDecodeError):
            return fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback

    def _migrate_legacy_audio_volume(self) -> None:
        migration_key = "audio_default_volume_v2"
        migrated = self.connection.execute(
            "SELECT 1 FROM app_metadata WHERE key = ?", (migration_key,)
        ).fetchone()
        if migrated:
            return
        # Only the previous built-in default is migrated; later explicit choices remain untouched.
        self.connection.execute(
            "UPDATE audio_preferences SET volume = ? WHERE ABS(volume - ?) < 0.000001",
            (DEFAULT_AUDIO_VOLUME, LEGACY_DEFAULT_AUDIO_VOLUME),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO app_metadata(key, value) VALUES (?, ?)",
            (migration_key, "completed"),
        )
        self.connection.commit()

    def _migrate_audio_fade_out_minutes(self) -> None:
        migration_key = "audio_fade_out_minutes_v1"
        migrated = self.connection.execute(
            "SELECT 1 FROM app_metadata WHERE key = ?", (migration_key,)
        ).fetchone()
        if migrated:
            return
        # The old default was labeled as 20 seconds but represented the intended 20-minute timer.
        # Preserve that default; normalize other legacy second values to whole minutes.
        self.connection.execute(
            """UPDATE audio_preferences
            SET fade_out_seconds = CASE
                WHEN fade_out_seconds = 20 THEN 1200
                WHEN fade_out_seconds <= 0 THEN 0
                ELSE MIN(7200, MAX(60, fade_out_seconds))
            END"""
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO app_metadata(key, value) VALUES (?, ?)",
            (migration_key, "completed"),
        )
        self.connection.commit()

    @_synchronized
    def close(self) -> None:
        self.connection.close()

    @_synchronized
    def ensure_user(self, user_id: str) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, self._now_text()),
        )
        self.connection.commit()

    @_synchronized
    def create_session(self, user_id: str, local_date: Optional[str] = None) -> str:
        user_id = self._require_user_id(user_id)
        session_id = str(uuid.uuid4())
        now = self._now_text()
        local_date = local_date or self.current_local_date()
        self.connection.execute(
            """INSERT INTO sessions
            (session_id, user_id, local_date, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, local_date, "CHECK_IN", now, now),
        )
        self.connection.commit()
        return session_id

    @_synchronized
    def start_conversation_round(self, user_id: str) -> tuple[str, str]:
        user_id = self._require_user_id(user_id)
        now = self._now_text()
        local_date = self.current_local_date()
        self.connection.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, now),
        )
        session = self.connection.execute(
            """SELECT * FROM sessions
            WHERE user_id = ? AND local_date = ?
            ORDER BY updated_at DESC, created_at DESC LIMIT 1""",
            (user_id, local_date),
        ).fetchone()
        if session:
            session_id = session["session_id"]
        else:
            session_id = str(uuid.uuid4())
            self.connection.execute(
                """INSERT INTO sessions
                (session_id, user_id, local_date, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, user_id, local_date, "CHECK_IN", now, now),
            )

        self.connection.execute(
            """UPDATE conversation_rounds
            SET status = 'abandoned', completed_at = ?, updated_at = ?
            WHERE user_id = ? AND session_id = ? AND status = 'active'""",
            (now, now, user_id, session_id),
        )
        row = self.connection.execute(
            "SELECT COALESCE(MAX(round_index), 0) + 1 FROM conversation_rounds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        round_index = int(row[0])
        round_id = str(uuid.uuid4())
        self.connection.execute(
            """INSERT INTO conversation_rounds
            (round_id, session_id, user_id, local_date, round_index, started_at,
             status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                round_id,
                session_id,
                user_id,
                local_date,
                round_index,
                now,
                "active",
                now,
                now,
            ),
        )
        self.connection.execute(
            """UPDATE sessions
            SET state = 'CHECK_IN', today_input = '', items_json = '[]',
                tomorrow_card = '', transition_json = '{}', closure_message = '', updated_at = ?
            WHERE session_id = ? AND user_id = ?""",
            (now, session_id, user_id),
        )
        self.connection.commit()
        return session_id, round_id

    @_synchronized
    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    @_synchronized
    def get_session_for_user(self, user_id: str, session_id: str) -> Optional[sqlite3.Row]:
        user_id = self._require_user_id(user_id)
        return self.connection.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()

    @_synchronized
    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = self._now_text()
        names = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [session_id]
        self.connection.execute(f"UPDATE sessions SET {names} WHERE session_id = ?", values)
        self.connection.commit()

    @_synchronized
    def add_turn(
        self, session_id: str, role: str, content: str, round_id: Optional[str] = None
    ) -> None:
        self.connection.execute(
            """INSERT INTO turns
            (turn_id, session_id, round_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), session_id, round_id, role, content, self._now_text()),
        )
        self.connection.commit()

    @_synchronized
    def recent_turns(self, session_id: str, limit: int = 6) -> list[str]:
        rows = self.connection.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [f"{row['role']}: {row['content']}" for row in reversed(rows)]

    @_synchronized
    def recent_turns_for_user(self, user_id: str, session_id: str, limit: int = 6) -> list[str]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            """SELECT turns.role, turns.content
            FROM turns
            JOIN sessions ON sessions.session_id = turns.session_id
            WHERE turns.session_id = ? AND sessions.user_id = ?
            ORDER BY turns.created_at DESC LIMIT ?""",
            (session_id, user_id, limit),
        ).fetchall()
        return [f"{row['role']}: {row['content']}" for row in reversed(rows)]

    @_synchronized
    def recent_summaries(self, user_id: str, limit: int = 3) -> list[str]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            "SELECT summary FROM session_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [row["summary"] for row in rows]

    @_synchronized
    def save_summary(self, session_id: str, user_id: str, summary: str) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            "INSERT OR REPLACE INTO session_summaries(session_id, user_id, summary, created_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, summary, self._now_text()),
        )
        self.connection.commit()

    @_synchronized
    def latest_round_for_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Optional[ConversationRound]:
        if user_id is None:
            row = self.connection.execute(
                """SELECT * FROM conversation_rounds
                WHERE session_id = ? ORDER BY round_index DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        else:
            user_id = self._require_user_id(user_id)
            row = self.connection.execute(
                """SELECT * FROM conversation_rounds
                WHERE session_id = ? AND user_id = ? ORDER BY round_index DESC LIMIT 1""",
                (session_id, user_id),
            ).fetchone()
        return self._round_from_row(row) if row else None

    @_synchronized
    def get_history_round(self, user_id: str, round_id: str) -> Optional[ConversationRound]:
        user_id = self._require_user_id(user_id)
        row = self.connection.execute(
            "SELECT * FROM conversation_rounds WHERE user_id = ? AND round_id = ?",
            (user_id, round_id),
        ).fetchone()
        return self._round_from_row(row) if row else None

    @_synchronized
    def latest_history_round(self, user_id: str) -> Optional[ConversationRound]:
        user_id = self._require_user_id(user_id)
        row = self.connection.execute(
            """SELECT * FROM conversation_rounds
            WHERE user_id = ?
            ORDER BY local_date DESC, round_index DESC, updated_at DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        return self._round_from_row(row) if row else None

    @_synchronized
    def update_history_round(
        self,
        user_id: str,
        session_id: str,
        round_id: str,
        **fields: Any,
    ) -> None:
        user_id = self._require_user_id(user_id)
        allowed = {
            "completed_at",
            "status",
            "initial_feeling",
            "concern_input",
            "items_json",
            "arrangements_json",
            "tomorrow_card",
            "wind_down_advice_json",
            "followup_feedback_json",
            "followup_advice_json",
            "tonight_action_json",
            "closure_message",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持的历史字段：{', '.join(sorted(unknown))}")
        if not fields:
            return
        json_fields = {
            "items_json",
            "arrangements_json",
            "wind_down_advice_json",
            "followup_feedback_json",
            "followup_advice_json",
            "tonight_action_json",
        }
        normalized = {
            key: (
                value
                if key not in json_fields or isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            for key, value in fields.items()
        }
        if normalized.get("status") in {"completed", "abandoned"} and not normalized.get(
            "completed_at"
        ):
            normalized["completed_at"] = self._now_text()
        normalized["updated_at"] = self._now_text()
        names = ", ".join(f"{key} = ?" for key in normalized)
        values = [*normalized.values(), user_id, session_id, round_id]
        cursor = self.connection.execute(
            f"""UPDATE conversation_rounds SET {names}
            WHERE user_id = ? AND session_id = ? AND round_id = ?""",
            values,
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError("找不到当前用户的历史轮次")
        self.connection.commit()

    @_synchronized
    def append_history_entry(
        self,
        user_id: str,
        session_id: str,
        round_id: str,
        field: str,
        entry: dict[str, Any],
    ) -> None:
        user_id = self._require_user_id(user_id)
        if field not in {"followup_feedback_json", "followup_advice_json"}:
            raise ValueError("该字段不支持追加历史记录")
        row = self.connection.execute(
            f"""SELECT {field} FROM conversation_rounds
            WHERE user_id = ? AND session_id = ? AND round_id = ?""",
            (user_id, session_id, round_id),
        ).fetchone()
        if not row:
            raise ValueError("找不到当前用户的历史轮次")
        values = self._safe_json(row[field], [])
        values.append(entry)
        self.connection.execute(
            f"""UPDATE conversation_rounds SET {field} = ?, updated_at = ?
            WHERE user_id = ? AND session_id = ? AND round_id = ?""",
            (
                json.dumps(values, ensure_ascii=False),
                self._now_text(),
                user_id,
                session_id,
                round_id,
            ),
        )
        self.connection.commit()

    @_synchronized
    def list_history_dates(self, user_id: str) -> list[dict[str, Any]]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            """SELECT local_date, COUNT(*) AS round_count, MAX(updated_at) AS last_updated
            FROM conversation_rounds
            WHERE user_id = ? AND concern_input <> ''
            GROUP BY local_date
            ORDER BY local_date DESC""",
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            latest = self.connection.execute(
                """SELECT concern_input FROM conversation_rounds
                WHERE user_id = ? AND local_date = ? AND concern_input <> ''
                ORDER BY round_index DESC, updated_at DESC LIMIT 1""",
                (user_id, row["local_date"]),
            ).fetchone()
            result.append(
                {
                    "local_date": row["local_date"],
                    "round_count": row["round_count"],
                    "last_updated": row["last_updated"],
                    "summary": (latest["concern_input"][:60] if latest else ""),
                }
            )
        return result

    @_synchronized
    def list_history_rounds(self, user_id: str, local_date: str) -> list[ConversationRound]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            """SELECT * FROM conversation_rounds
            WHERE user_id = ? AND local_date = ? AND concern_input <> ''
            ORDER BY round_index DESC""",
            (user_id, local_date),
        ).fetchall()
        return [self._round_from_row(row) for row in rows]

    @classmethod
    def _round_from_row(cls, row: sqlite3.Row) -> ConversationRound:
        payload = dict(row)
        payload["items"] = [
            TodoItem(**item) for item in cls._safe_json(payload.pop("items_json"), [])
        ]
        payload["arrangements"] = cls._safe_json(payload.pop("arrangements_json"), [])
        payload["wind_down_advice"] = cls._safe_json(
            payload.pop("wind_down_advice_json"), {}
        )
        payload["followup_feedback"] = cls._safe_json(
            payload.pop("followup_feedback_json"), []
        )
        payload["followup_advice"] = cls._safe_json(
            payload.pop("followup_advice_json"), []
        )
        payload["tonight_action"] = cls._safe_json(
            payload.pop("tonight_action_json"), {}
        )
        return ConversationRound(**payload)

    @_synchronized
    def save_memory(self, memory: MemoryItem) -> None:
        self._require_user_id(memory.user_id)
        self.connection.execute(
            """INSERT OR REPLACE INTO memory_items
            (memory_id, user_id, kind, content, source_type, source_ref, confidence,
             importance, consent, source_session_id, status, valid_from, valid_until,
             half_life_days, evidence_count, memory_key, created_at, updated_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.memory_id,
                memory.user_id,
                memory.kind,
                memory.content,
                memory.source_type,
                memory.source_ref,
                memory.confidence,
                memory.importance,
                int(memory.consent),
                memory.source_session_id,
                memory.status,
                memory.valid_from,
                memory.valid_until,
                memory.half_life_days,
                memory.evidence_count,
                memory.memory_key,
                memory.created_at,
                memory.updated_at,
                memory.last_used_at,
            ),
        )
        self.connection.commit()

    @_synchronized
    def get_memory(self, user_id: str, memory_id: str) -> Optional[MemoryItem]:
        user_id = self._require_user_id(user_id)
        row = self.connection.execute(
            "SELECT * FROM memory_items WHERE user_id = ? AND memory_id = ?",
            (user_id, memory_id),
        ).fetchone()
        return MemoryItem(**dict(row)) if row else None

    @_synchronized
    def list_memories(
        self, user_id: str, include_inactive: bool = False, now: Optional[str] = None
    ) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        if include_inactive:
            rows = self.connection.execute(
                "SELECT * FROM memory_items WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            now = now or utc_now()
            rows = self.connection.execute(
                """SELECT * FROM memory_items
                WHERE user_id = ? AND consent = 1 AND status = 'active'
                  AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)
                ORDER BY updated_at DESC""",
                (user_id, now, now),
            ).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    @_synchronized
    def find_active_memory(
        self, user_id: str, content: str, memory_key: Optional[str], now: str
    ) -> Optional[MemoryItem]:
        user_id = self._require_user_id(user_id)
        if memory_key:
            row = self.connection.execute(
                """SELECT * FROM memory_items
                WHERE user_id = ? AND memory_key = ? AND content = ? AND status = 'active'
                  AND consent = 1 AND valid_from <= ?
                  AND (valid_until IS NULL OR valid_until > ?)
                ORDER BY updated_at DESC LIMIT 1""",
                (user_id, memory_key, content, now, now),
            ).fetchone()
        else:
            row = self.connection.execute(
                """SELECT * FROM memory_items
                WHERE user_id = ? AND content = ? AND status = 'active' AND consent = 1
                  AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)
                ORDER BY updated_at DESC LIMIT 1""",
                (user_id, content, now, now),
            ).fetchone()
        return MemoryItem(**dict(row)) if row else None

    @_synchronized
    def active_memories_for_key(
        self, user_id: str, memory_key: str, now: str
    ) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            """SELECT * FROM memory_items
            WHERE user_id = ? AND memory_key = ? AND status = 'active' AND consent = 1
              AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)""",
            (user_id, memory_key, now, now),
        ).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    @_synchronized
    def active_memories_for_source_session(
        self, user_id: str, session_id: str, now: str
    ) -> list[MemoryItem]:
        user_id = self._require_user_id(user_id)
        rows = self.connection.execute(
            """SELECT * FROM memory_items
            WHERE user_id = ? AND source_session_id = ? AND status = 'active' AND consent = 1
              AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)""",
            (user_id, session_id, now, now),
        ).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    @_synchronized
    def update_memory_status(
        self, user_id: str, memory_id: str, status: str, updated_at: str
    ) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            "UPDATE memory_items SET status = ?, updated_at = ? WHERE user_id = ? AND memory_id = ?",
            (status, updated_at, user_id, memory_id),
        )
        self.connection.commit()

    @_synchronized
    def expire_memories(self, user_id: str, now: str) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            """UPDATE memory_items SET status = 'expired', updated_at = ?
            WHERE user_id = ? AND status = 'active' AND valid_until IS NOT NULL
              AND valid_until <= ?""",
            (now, user_id, now),
        )
        self.connection.execute(
            """UPDATE user_profile_facts SET status = 'expired', updated_at = ?
            WHERE user_id = ? AND status = 'active' AND valid_until IS NOT NULL
              AND valid_until <= ?""",
            (now, user_id, now),
        )
        self.connection.commit()

    @_synchronized
    def touch_memory(self, user_id: str, memory_id: str, last_used_at: str) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            "UPDATE memory_items SET last_used_at = ? WHERE user_id = ? AND memory_id = ?",
            (last_used_at, user_id, memory_id),
        )
        self.connection.commit()

    @_synchronized
    def save_profile_fact(self, fact: UserProfileFact) -> None:
        self._require_user_id(fact.user_id)
        self.connection.execute(
            """INSERT OR REPLACE INTO user_profile_facts
            (profile_id, user_id, profile_key, profile_value, source_type, confidence,
             source_memory_ids_json, evidence_count, status, valid_until, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.profile_id,
                fact.user_id,
                fact.profile_key,
                fact.profile_value,
                fact.source_type,
                fact.confidence,
                json.dumps(fact.source_memory_ids, ensure_ascii=False),
                fact.evidence_count,
                fact.status,
                fact.valid_until,
                fact.created_at,
                fact.updated_at,
            ),
        )
        self.connection.commit()

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> UserProfileFact:
        payload = dict(row)
        payload["source_memory_ids"] = json.loads(payload.pop("source_memory_ids_json") or "[]")
        return UserProfileFact(**payload)

    @_synchronized
    def list_profile_facts(
        self, user_id: str, now: str, include_inactive: bool = False
    ) -> list[UserProfileFact]:
        user_id = self._require_user_id(user_id)
        if include_inactive:
            rows = self.connection.execute(
                "SELECT * FROM user_profile_facts WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """SELECT * FROM user_profile_facts
                WHERE user_id = ? AND status = 'active'
                  AND (valid_until IS NULL OR valid_until > ?)
                ORDER BY confidence DESC, evidence_count DESC, updated_at DESC""",
                (user_id, now),
            ).fetchall()
        return [self._profile_from_row(row) for row in rows]

    @_synchronized
    def find_profile_fact(
        self, user_id: str, profile_key: str, profile_value: str, now: str
    ) -> Optional[UserProfileFact]:
        user_id = self._require_user_id(user_id)
        row = self.connection.execute(
            """SELECT * FROM user_profile_facts
            WHERE user_id = ? AND profile_key = ? AND profile_value = ? AND status = 'active'
              AND (valid_until IS NULL OR valid_until > ?)
            ORDER BY updated_at DESC LIMIT 1""",
            (user_id, profile_key, profile_value, now),
        ).fetchone()
        return self._profile_from_row(row) if row else None

    @_synchronized
    def supersede_profile_key(
        self, user_id: str, profile_key: str, except_value: str, updated_at: str
    ) -> None:
        user_id = self._require_user_id(user_id)
        self.connection.execute(
            """UPDATE user_profile_facts SET status = 'superseded', updated_at = ?
            WHERE user_id = ? AND profile_key = ? AND profile_value <> ? AND status = 'active'""",
            (updated_at, user_id, profile_key, except_value),
        )
        self.connection.commit()

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")
        return user_id.strip()

    @_synchronized
    def get_audio_assets(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        if user_id:
            rows = self.connection.execute(
                "SELECT * FROM audio_assets WHERE owner_type = 'developer' OR owner_id = ? ORDER BY owner_type, title",
                (user_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM audio_assets WHERE owner_type = 'developer' ORDER BY title"
            ).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def save_audio_asset(self, asset: dict[str, Any]) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO audio_assets
            (audio_id, title, category, file_name, owner_type, owner_id, duration_seconds, loopable, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset["audio_id"], asset["title"], asset["category"], asset["file_name"],
                asset["owner_type"], asset.get("owner_id"), asset.get("duration_seconds", 30),
                int(asset.get("loopable", True)), asset.get("source", "built-in"), asset.get("created_at", utc_now()),
            ),
        )
        self.connection.commit()

    @_synchronized
    def get_audio_preference(self, user_id: str, fallback_audio_id: str) -> AudioPreference:
        row = self.connection.execute(
            "SELECT * FROM audio_preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            preference = AudioPreference(user_id=user_id, default_audio_id=fallback_audio_id)
            self.save_audio_preference(preference)
            return preference
        return AudioPreference(
            user_id=row["user_id"], default_audio_id=row["default_audio_id"], volume=row["volume"],
            autoplay_enabled=bool(row["autoplay_enabled"]),
            fade_out_minutes=min(120, max(0, round(row["fade_out_seconds"] / 60))),
        )

    @_synchronized
    def save_audio_preference(self, preference: AudioPreference) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO audio_preferences
            (user_id, default_audio_id, volume, autoplay_enabled, fade_out_seconds, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                preference.user_id, preference.default_audio_id, preference.volume,
                int(preference.autoplay_enabled), preference.fade_out_minutes * 60, utc_now(),
            ),
        )
        self.connection.commit()

    @_synchronized
    def log_event(
        self,
        session_id: str,
        from_state: Optional[str],
        to_state: str,
        payload: dict[str, Any],
        round_id: Optional[str] = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO workflow_events
            (event_id, session_id, round_id, from_state, to_state, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), session_id, round_id, from_state, to_state,
                json.dumps(payload, ensure_ascii=False), self._now_text(),
            ),
        )
        self.connection.commit()
