"""All path resolution using platformdirs.

This module is the single source of truth for all file paths in DevAgent.
It uses platformdirs to resolve correct data/config directories per OS.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import platformdirs

APP_NAME = "devagent"


def get_config_path() -> Path:
    """Return the path to the config file.

    On Windows: %APPDATA%\\devagent\\config.toml
    Can be overridden via DEVAGENT_CONFIG_PATH or SPECSYNC_CONFIG_PATH environment variable.
    """
    # Check for environment variable override first (support both for backward compatibility)
    if "DEVAGENT_CONFIG_PATH" in os.environ:
        return Path(os.environ["DEVAGENT_CONFIG_PATH"])
    if "SPECSYNC_CONFIG_PATH" in os.environ:
        return Path(os.environ["SPECSYNC_CONFIG_PATH"])
    config_dir = Path(platformdirs.user_config_dir(APP_NAME))
    return config_dir / "config.toml"


def get_data_dir() -> Path:
    """Return the base data directory.

    On Windows: %LOCALAPPDATA%\\devagent
    """
    return Path(platformdirs.user_data_dir(APP_NAME))


def get_project_hash(project_root: Path) -> str:
    """Compute a deterministic SHA256 hash of the absolute project path.

    This ensures the same project always maps to the same folder
    regardless of how DevAgent is invoked.
    """
    absolute = str(project_root.resolve())
    return hashlib.sha256(absolute.encode("utf-8")).hexdigest()


def get_project_dir(project_root: Path) -> Path:
    """Return the project-specific data directory.

    data_dir/projects/{project_hash}
    """
    return get_data_dir() / "projects" / get_project_hash(project_root)


def get_chroma_dir(project_root: Path) -> Path:
    """Return the ChromaDB directory for a project.

    project_dir/chroma
    """
    return get_project_dir(project_root) / "chroma"


def get_sqlite_path(project_root: Path) -> Path:
    """Return the SQLite database path for a project.

    project_dir/devagent.db
    """
    return get_project_dir(project_root) / "devagent.db"


def get_reports_dir(project_root: Path) -> Path:
    """Return the reports directory for a project.

    data_dir/reports/{project_hash}
    """
    return get_data_dir() / "reports" / get_project_hash(project_root)


def ensure_dirs(project_root: Path) -> None:
    """Create all required directories for a project if they don't exist."""
    dirs = [
        get_config_path().parent,
        get_project_dir(project_root),
        get_chroma_dir(project_root),
        get_reports_dir(project_root),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SQLite database helpers (for CLI-level index status queries)
# ---------------------------------------------------------------------------

async def init_db(sqlite_path: Path) -> None:
    """Create the SQLite tables if they don't exist."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path, timeout=5) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                last_modified REAL NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS project_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        db.commit()


async def get_index_status(sqlite_path: Path) -> dict:
    """Return a summary of the current index state.

    Returns dict with keys: total_files, total_chunks, last_indexed, exists.
    """
    if not sqlite_path.is_file():
        return {"total_files": 0, "total_chunks": 0, "last_indexed": None, "exists": False}

    with sqlite3.connect(sqlite_path, timeout=5) as db:
        row = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM indexed_files"
        ).fetchone()
        total_files = row[0] if row else 0
        total_chunks = row[1] if row else 0
        row = db.execute("SELECT MAX(indexed_at) FROM indexed_files").fetchone()
        last_indexed = row[0] if row else None

    return {
        "total_files": total_files,
        "total_chunks": total_chunks,
        "last_indexed": last_indexed,
        "exists": True,
    }


async def get_changed_files_count(project_root: Path, sqlite_path: Path) -> dict:
    """Compare files on disk against the SQLite index to find changes.

    Returns dict with keys: changed, new, deleted.
    """
    import pathspec as _pathspec

    if not sqlite_path.is_file():
        return {"changed": 0, "new": 0, "deleted": 0}

    # Load indexed files
    indexed: dict[str, float] = {}
    with sqlite3.connect(sqlite_path, timeout=5) as db:
        rows = db.execute("SELECT file_path, last_modified FROM indexed_files").fetchall()
        indexed.update({row[0]: row[1] for row in rows})

    # Load gitignore spec
    gitignore_path = project_root / ".gitignore"
    gitignore_spec = None
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
            lines = f.readlines()
        gitignore_spec = _pathspec.PathSpec.from_lines(_pathspec.patterns.GitWildMatchPattern, lines)

    built_in_ignore = {
        "node_modules", ".venv", "venv", "env", ".env", "__pycache__", ".git",
        "dist", "build", ".idea", ".vscode",
    }
    binary_exts = {".pyc", ".pyo", ".egg-info", ".exe", ".dll", ".so", ".png", ".jpg", ".jpeg"}

    # Walk disk
    disk_files: set[str] = set()
    changed = 0
    new = 0

    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(project_root).as_posix()

        # Skip ignored
        if any(p in built_in_ignore for p in Path(rel).parts):
            continue
        if file_path.suffix in binary_exts:
            continue
        if gitignore_spec and gitignore_spec.match_file(rel):
            continue

        disk_files.add(rel)
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            continue

        if rel in indexed:
            if mtime > indexed[rel]:
                changed += 1
        else:
            new += 1

    deleted = len(set(indexed.keys()) - disk_files)

    return {"changed": changed, "new": new, "deleted": deleted}


def clear_project_index(project_root: Path) -> None:
    """Delete the ChromaDB directory and SQLite file for a project."""
    import shutil

    chroma_dir = get_chroma_dir(project_root)
    sqlite_path = get_sqlite_path(project_root)

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    if sqlite_path.exists():
        sqlite_path.unlink()


# ---------------------------------------------------------------------------
# F3 — Watcher path helpers
# ---------------------------------------------------------------------------

def get_sessions_db_path() -> Path:
    """Returns path to the global agent sessions SQLite database.

    data_dir/sessions/sessions.db
    """
    p = get_data_dir() / "sessions" / "sessions.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_watcher_db_path() -> Path:
    """Returns path to the watcher SQLite database.

    data_dir/watcher/watcher.db
    """
    return get_data_dir() / "watcher" / "watcher.db"


def get_watcher_reports_dir(owner: str, repo: str) -> Path:
    """Returns path to watcher reports for a specific repo.

    data_dir/watcher/reports/{owner}-{repo}
    """
    return get_data_dir() / "watcher" / "reports" / f"{owner}-{repo}"

