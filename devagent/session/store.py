"""SQLite-backed session persistence.

Schema:
  sessions       -- one row per session (id, project, model, created_at, updated_at, metadata)
  events         -- append-only event log (turn messages, tool calls, results)
  memory_items   -- structured session memory (key-value with type/scope)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from devagent.core.storage import get_sessions_db_path


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    if not isinstance(db_path, (str, Path)):
        db_path = None
    path = db_path or get_sessions_db_path()
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: Path | None = None) -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    with _conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id                  TEXT PRIMARY KEY,
                project             TEXT NOT NULL DEFAULT '',
                model               TEXT NOT NULL DEFAULT '',
                provider            TEXT NOT NULL DEFAULT '',
                title               TEXT NOT NULL DEFAULT '',
                created_at          REAL NOT NULL,
                updated_at          REAL NOT NULL,
                metadata            TEXT NOT NULL DEFAULT '{}',
                compressed_summary  TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                seq           INTEGER NOT NULL,
                role          TEXT NOT NULL,
                content       TEXT NOT NULL DEFAULT '',
                tool_calls    TEXT NOT NULL DEFAULT '[]',
                tool_call_id  TEXT NOT NULL DEFAULT '',
                tool_name     TEXT NOT NULL DEFAULT '',
                tokens_in     INTEGER NOT NULL DEFAULT 0,
                tokens_out    INTEGER NOT NULL DEFAULT 0,
                created_at    REAL NOT NULL,
                is_compressed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);

            CREATE TABLE IF NOT EXISTS memory_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                scope       TEXT NOT NULL DEFAULT 'session',
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                item_type   TEXT NOT NULL DEFAULT 'fact',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                UNIQUE(session_id, scope, key),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS task_graph (
                id              TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                worker_type     TEXT NOT NULL DEFAULT 'implementer',
                depends_on      TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'pending',
                result          TEXT NOT NULL DEFAULT '',
                output_files    TEXT NOT NULL DEFAULT '[]',
                assigned_session TEXT NOT NULL DEFAULT '',
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL,
                PRIMARY KEY (session_id, id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS file_locks (
                session_id  TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                worker_id   TEXT NOT NULL,
                locked_at   REAL NOT NULL,
                PRIMARY KEY (session_id, file_path)
            );
        """)
        # Migrate existing databases that predate Phase 8/9 columns
        for sql in (
            "ALTER TABLE sessions ADD COLUMN compressed_summary TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE events ADD COLUMN is_compressed INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                conn.execute(sql)
            except Exception:
                pass  # column already exists — safe to ignore


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def create_session(
    session_id: str,
    project: str = "",
    model: str = "",
    provider: str = "",
    title: str = "",
    metadata: dict | None = None,
    db_path: Path | None = None,
) -> None:
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions (id, project, model, provider, title, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, project, model, provider, title, now, now, json.dumps(metadata or {})),
        )


def get_session(session_id: str, db_path: Path | None = None) -> dict | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"])
    return d


def update_session(
    session_id: str,
    title: str | None = None,
    metadata: dict | None = None,
    db_path: Path | None = None,
) -> None:
    now = time.time()
    with _conn(db_path) as conn:
        if title is not None:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
        if metadata is not None:
            conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), now, session_id),
            )
        if title is None and metadata is None:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )


def list_sessions(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def list_sessions_by_project(project: str, limit: int = 50, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE project = ? ORDER BY updated_at DESC LIMIT ?",
            (project, limit),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def delete_session(session_id: str, db_path: Path | None = None) -> None:
    """Delete a session and every row that belongs to it.

    Every table that carries a ``session_id`` is purged, not just the two the
    original implementation happened to name. ``task_graph`` and
    ``file_locks`` have no ``ON DELETE CASCADE`` (and SQLite does not enforce
    foreign keys at all unless ``PRAGMA foreign_keys=ON`` is set, which
    ``_conn`` does not do), so rows left behind here are unreachable orphans
    that would never be cleaned up. Children are deleted before the parent row
    so the order is correct if the pragma is ever enabled.
    """
    with _conn(db_path) as conn:
        conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM memory_items WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM task_graph WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM file_locks WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ---------------------------------------------------------------------------
# Event log (messages + tool calls)
# ---------------------------------------------------------------------------

def append_event(
    session_id: str,
    role: str,
    content: str = "",
    tool_calls: list[dict] | None = None,
    tool_call_id: str = "",
    tool_name: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    db_path: Path | None = None,
) -> int:
    """Append one event; returns its seq number."""
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE session_id = ?",
            (session_id,),
        )
        seq = cur.fetchone()[0]
        conn.execute(
            """INSERT INTO events
               (session_id, seq, role, content, tool_calls, tool_call_id, tool_name,
                tokens_in, tokens_out, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, seq, role, content,
                json.dumps(tool_calls or []),
                tool_call_id, tool_name,
                tokens_in, tokens_out, now,
            ),
        )
        conn.commit()
        return seq


def get_events(session_id: str, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["tool_calls"] = json.loads(d["tool_calls"])
        result.append(d)
    return result


def copy_session_events(
    from_id: str,
    to_id: str,
    db_path: Path | None = None,
) -> int:
    """Copy all events from one session to another, re-sequencing from 0.

    Returns the number of events copied.
    """
    events = get_events(from_id, db_path)
    now = time.time()
    with _conn(db_path) as conn:
        for i, ev in enumerate(events):
            conn.execute(
                """INSERT INTO events
                   (session_id, seq, role, content, tool_calls, tool_call_id, tool_name,
                    tokens_in, tokens_out, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    to_id, i, ev["role"], ev["content"],
                    json.dumps(ev.get("tool_calls") or []),
                    ev.get("tool_call_id", ""), ev.get("tool_name", ""),
                    ev.get("tokens_in", 0), ev.get("tokens_out", 0), now,
                ),
            )
    return len(events)


def trim_events(session_id: str, turns_to_remove: int, db_path: Path | None = None) -> int:
    """Delete the last N user-turn groups from the event log.

    A 'turn' is one user message plus all following assistant/tool events.
    Returns the number of events deleted.
    """
    events = get_events(session_id, db_path)
    if not events:
        return 0
    user_seqs = [e["seq"] for e in events if e["role"] == "user"]
    if not user_seqs:
        return 0
    n = min(turns_to_remove, len(user_seqs))
    cutoff_seq = user_seqs[-n]
    with _conn(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ? AND seq >= ?",
            (session_id, cutoff_seq),
        )
        count = cur.fetchone()[0]
        conn.execute(
            "DELETE FROM events WHERE session_id = ? AND seq >= ?",
            (session_id, cutoff_seq),
        )
    return count


def get_token_totals(session_id: str, db_path: Path | None = None) -> dict[str, int]:
    with _conn(db_path) as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0)
               FROM events WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
    return {"tokens_in": row[0], "tokens_out": row[1]}


# ---------------------------------------------------------------------------
# Structured memory
# ---------------------------------------------------------------------------

def upsert_memory(
    session_id: str,
    key: str,
    value: Any,
    scope: str = "session",
    item_type: str = "fact",
    db_path: Path | None = None,
) -> None:
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """INSERT INTO memory_items (session_id, scope, key, value, item_type, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, scope, key)
               DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (session_id, scope, key, json.dumps(value), item_type, now, now),
        )


def get_memory(
    session_id: str,
    scope: str = "session",
    db_path: Path | None = None,
) -> dict[str, Any]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT key, value FROM memory_items WHERE session_id = ? AND scope = ?",
            (session_id, scope),
        ).fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def delete_memory_key(
    session_id: str,
    key: str,
    scope: str = "session",
    db_path: Path | None = None,
) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "DELETE FROM memory_items WHERE session_id = ? AND scope = ? AND key = ?",
            (session_id, scope, key),
        )


# ---------------------------------------------------------------------------
# Phase 8 — context auto-compression helpers
# ---------------------------------------------------------------------------

def get_compressed_summary(session_id: str, db_path: Path | None = None) -> str:
    """Return the stored compressed summary for this session, or '' if none."""
    try:
        with _conn(db_path) as conn:
            row = conn.execute(
                "SELECT compressed_summary FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return (row["compressed_summary"] or "") if row else ""
    except Exception:
        return ""


def save_compressed_summary(
    session_id: str,
    summary: str,
    db_path: Path | None = None,
) -> None:
    """Persist the compressed summary on the session row."""
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET compressed_summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, session_id),
        )


def mark_events_compressed(
    session_id: str,
    max_seq: int,
    db_path: Path | None = None,
) -> None:
    """Flag all events with seq <= max_seq as compressed (part of the summary)."""
    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE events SET is_compressed = 1 WHERE session_id = ? AND seq <= ?",
            (session_id, max_seq),
        )


def get_active_events(session_id: str, db_path: Path | None = None) -> list[dict]:
    """Return only events that have NOT been folded into the compressed summary."""
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE session_id = ? AND is_compressed = 0 ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["tool_calls"] = json.loads(d["tool_calls"])
        result.append(d)
    return result


def count_compressed_events(session_id: str, db_path: Path | None = None) -> int:
    """Return how many events have been compressed so far."""
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ? AND is_compressed = 1",
            (session_id,),
        ).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Phase 9 — task graph persistence
# ---------------------------------------------------------------------------

def upsert_task(
    session_id: str,
    task_id: str,
    description: str,
    worker_type: str = "implementer",
    depends_on: list[str] | None = None,
    status: str = "pending",
    result: str = "",
    output_files: list[str] | None = None,
    assigned_session: str = "",
    db_path: Path | None = None,
) -> None:
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """INSERT INTO task_graph
               (id, session_id, description, worker_type, depends_on, status,
                result, output_files, assigned_session, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, id)
               DO UPDATE SET status=excluded.status, result=excluded.result,
                             output_files=excluded.output_files,
                             assigned_session=excluded.assigned_session,
                             updated_at=excluded.updated_at""",
            (
                task_id, session_id, description, worker_type,
                json.dumps(depends_on or []), status,
                result, json.dumps(output_files or []),
                assigned_session, now, now,
            ),
        )


def get_tasks(session_id: str, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM task_graph WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["depends_on"] = json.loads(d["depends_on"])
        d["output_files"] = json.loads(d["output_files"])
        result.append(d)
    return result


def update_task_status(
    session_id: str,
    task_id: str,
    status: str,
    result: str = "",
    output_files: list[str] | None = None,
    assigned_session: str = "",
    db_path: Path | None = None,
) -> None:
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            """UPDATE task_graph SET status=?, result=?, output_files=?,
               assigned_session=?, updated_at=?
               WHERE session_id=? AND id=?""",
            (
                status, result,
                json.dumps(output_files or []),
                assigned_session, now,
                session_id, task_id,
            ),
        )


# ---------------------------------------------------------------------------
# Phase 9 — file lock table
# ---------------------------------------------------------------------------

def acquire_file_lock(
    session_id: str,
    file_path: str,
    worker_id: str,
    db_path: Path | None = None,
) -> bool:
    """Try to acquire an exclusive write lock. Returns True on success."""
    now = time.time()
    try:
        with _conn(db_path) as conn:
            conn.execute(
                "INSERT INTO file_locks (session_id, file_path, worker_id, locked_at) VALUES (?, ?, ?, ?)",
                (session_id, file_path, worker_id, now),
            )
        return True
    except Exception:
        return False  # UNIQUE constraint violation — already locked


def release_file_lock(
    session_id: str,
    file_path: str,
    worker_id: str,
    db_path: Path | None = None,
) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "DELETE FROM file_locks WHERE session_id=? AND file_path=? AND worker_id=?",
            (session_id, file_path, worker_id),
        )


def release_all_worker_locks(
    session_id: str,
    worker_id: str,
    db_path: Path | None = None,
) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "DELETE FROM file_locks WHERE session_id=? AND worker_id=?",
            (session_id, worker_id),
        )


def get_file_locks(session_id: str, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM file_locks WHERE session_id=?", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]
