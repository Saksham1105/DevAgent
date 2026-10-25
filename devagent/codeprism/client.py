"""CodePrism client — sync wrapper around the async CodePrism Python API.

Uses a dedicated background thread with its own event loop so tool handlers
(which are sync) can call async CodePrism methods without creating a new
event loop per call or fighting an existing one.

Design:
  _loop / _thread  — module-level daemon thread; started once on first import
  CodePrismClient  — per-project singleton; lazy-initialises CodePrism on first use
  _run(coro)       — submit a coroutine and block until result (30s timeout)
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Background event loop (started once per process)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(
                target=_loop.run_forever,
                daemon=True,
                name="codeprism-loop",
            )
            _thread.start()
        return _loop


def _run(coro, timeout: float = 30.0) -> Any:
    """Submit coroutine to the background loop and block until done."""
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise


# ---------------------------------------------------------------------------
# Result formatters (mirror codeprism.mcp.server helpers)
# ---------------------------------------------------------------------------

def _fmt_symbol(s) -> dict:
    return {
        "name": s.name,
        "kind": s.kind.value if hasattr(s.kind, "value") else str(s.kind),
        "file": getattr(s, "file_path", ""),
        "line": s.line_start,
        "signature": s.signature or "",
        "is_public": s.is_public,
    }


def _fmt_context(r) -> dict:
    return {
        "symbol": _fmt_symbol(r.symbol),
        "direct_callers": [_fmt_symbol(s) for s in r.direct_callers],
        "direct_callees": [_fmt_symbol(s) for s in r.direct_callees],
        "related_types": [_fmt_symbol(s) for s in r.related_types],
        "relevant_variables": [_fmt_symbol(s) for s in r.relevant_variables],
        "estimated_tokens": r.estimated_token_count,
    }


def _fmt_impact(r) -> dict:
    return {
        "symbol": r.symbol.name,
        "severity": r.severity,
        "public_api_affected": r.public_api_affected,
        "estimated_change_surface": r.estimated_change_surface,
        "direct_dependents": [_fmt_symbol(s) for s in r.direct_dependents],
        "transitive_dependents": [_fmt_symbol(s) for s in r.transitive_dependents],
        "affected_test_files": r.affected_test_files,
    }


def _fmt_summary(r) -> dict:
    return {
        "file": r.file.path,
        "purpose": r.purpose,
        "public_api": [_fmt_symbol(s) for s in r.public_api],
        "dependencies": r.dependencies,
        "complexity_score": r.complexity_score,
        "test_coverage_file": r.test_coverage_file,
        "key_classes": [_fmt_symbol(s) for s in r.key_classes],
    }


def _fmt_file_map(r) -> dict:
    entries = []
    for e in r.entries:
        entries.append({
            "path": e.path,
            "role": getattr(e, "role", ""),
            "symbols": getattr(e, "symbol_count", 0),
        })
    return {
        "project_path": r.project_path,
        "total_files": r.total_files,
        "total_symbols": r.total_symbols,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# CodePrismClient
# ---------------------------------------------------------------------------

class CodePrismClient:
    """Per-project CodePrism client.  Lazy-initialises on first query.

    Call `attach_session(session_id)` once the DevAgent session ID is known
    so reads/writes are tracked against that session.
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root).resolve()
        self._prism = None
        self._session_obj = None
        self._session_id: str | None = None
        self._ready = False   # True after first _ensure_ready()
        self._indexed = False  # True if CodePrism's project database exists

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> bool:
        if self._ready:
            return self._indexed

        db_path = self._db_path()
        if not db_path.exists():
            self._ready = True
            self._indexed = False
            return False

        try:
            from codeprism import CodePrism
            prism = CodePrism(str(self.project_root))
            _run(prism.initialize(), timeout=10)
            self._prism = prism
            self._ready = True
            self._indexed = True
            # Reattach session if already set
            if self._session_id:
                self._session_obj = self._prism.session(self._session_id)
            return True
        except Exception:
            self._ready = True
            self._indexed = False
            return False

    @property
    def is_indexed(self) -> bool:
        if self._ready:
            return self._indexed
        return self._db_path().exists()

    def _db_path(self) -> Path:
        try:
            from codeprism.core.paths import get_db_path
            return get_db_path(self.project_root)
        except Exception:
            return self.project_root / ".codeprism" / "codeprism.db"

    def _not_indexed(self) -> dict:
        return {
            "error": (
                "CodePrism knowledge graph not found for this project. "
                f"Run:  codeprism index {self.project_root}  "
                "then restart the agent session to enable graph-powered tools."
            )
        }

    def attach_session(self, session_id: str) -> None:
        self._session_id = session_id
        if self._prism:
            self._session_obj = self._prism.session(session_id)

    # ------------------------------------------------------------------
    # Query helpers (use engine directly for full API surface)
    # ------------------------------------------------------------------

    def _engine(self):
        return self._prism.engine  # QueryEngine

    # Context
    def get_context(self, file: str, symbol: str, depth: int = 2) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_context(file, symbol, depth))
            return {"error": f"Symbol '{symbol}' not found in {file}"} if r is None else _fmt_context(r)
        except Exception as exc:
            return {"error": str(exc)}

    # Impact
    def get_impact(self, file: str, symbol: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_impact(file, symbol))
            return {"error": f"Symbol '{symbol}' not found in {file}"} if r is None else _fmt_impact(r)
        except Exception as exc:
            return {"error": str(exc)}

    # Module summary
    def get_module_summary(self, file: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_module_summary(file))
            return {"error": f"File '{file}' not indexed"} if r is None else _fmt_summary(r)
        except Exception as exc:
            return {"error": str(exc)}

    # Callers
    def get_callers(self, file: str, function: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            callers = _run(self._engine().get_callers(file, function))
            return {
                "function": function,
                "file": file,
                "count": len(callers),
                "callers": [_fmt_symbol(s) for s in callers],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # Callees
    def get_callees(self, file: str, function: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            callees = _run(self._engine().get_callees(file, function))
            return {
                "function": function,
                "file": file,
                "count": len(callees),
                "callees": [_fmt_symbol(s) for s in callees],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # Search
    def search_symbol(self, query: str, kind: str | None = None) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            matches = _run(self._engine().search_symbols(query, kind))
            return {
                "count": len(matches),
                "matches": [
                    {
                        "name": m.symbol.name,
                        "kind": m.symbol.kind.value if hasattr(m.symbol.kind, "value") else str(m.symbol.kind),
                        "file": m.file_path,
                        "line": m.symbol.line_start,
                        "score": m.score,
                        "docstring": m.docstring_excerpt or "",
                    }
                    for m in matches
                ],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # Data flow
    def get_data_flow(self, file: str, symbol: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_data_flow(file, symbol))
            if r is None:
                return {"error": f"Symbol '{symbol}' not found in {file}"}
            return {
                "symbol": symbol,
                "sources": [getattr(n, "name", str(n)) for n in r.sources],
                "sinks": [getattr(n, "name", str(n)) for n in r.sinks],
                "intermediate_nodes": [getattr(n, "name", str(n)) for n in r.intermediate_nodes],
                "flow_paths": [
                    [getattr(n, "name", str(n)) for n in path]
                    for path in r.flow_paths
                ],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # File map
    def get_file_map(self, project_path: str = "") -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_file_map(project_path or str(self.project_root)))
            return _fmt_file_map(r)
        except Exception as exc:
            return {"error": str(exc)}

    # Dependencies
    def get_dependencies(self, file: str) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            r = _run(self._engine().get_dependencies(file))
            if r is None:
                return {"error": f"File '{file}' not indexed"}
            return {
                "file": file,
                "internal_deps": [getattr(d, "path", str(d)) for d in r.internal_deps],
                "external_deps": r.external_deps,
                "circular_deps": [getattr(d, "path", str(d)) for d in r.circular_deps],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # Stats
    def get_stats(self) -> dict:
        if not self._ensure_ready():
            return self._not_indexed()
        try:
            return _run(self._engine().get_stats())
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def record_read(self, file: str, symbol: str) -> None:
        if not self._ensure_ready() or self._session_obj is None:
            return
        try:
            _run(self._session_obj.record_read(file, symbol))
        except Exception:
            pass  # Never block agent on tracking failures

    def record_write(self, file: str, content_before: str, content_after: str) -> dict:
        """Record a write and run the security gate.  Returns security scan result."""
        if not self._ensure_ready() or self._session_obj is None:
            return {"status": "PASS", "issues": []}
        try:
            result = _run(self._session_obj.record_write(file, content_before, content_after))
            # result is a dict from the SessionManager with {status, issues, graph_update}
            return result if isinstance(result, dict) else {"status": "PASS", "issues": []}
        except Exception as exc:
            return {"status": "PASS", "issues": [], "note": str(exc)}

    def get_session_context(self) -> dict:
        if not self._ensure_ready() or self._session_obj is None:
            return {}
        try:
            ctx = _run(self._session_obj.get_context())
            return {
                "files_read": ctx.files_read,
                "files_written": ctx.files_written,
                "read_count": ctx.read_count,
                "write_count": ctx.write_count,
                "summary": ctx.summary,
            }
        except Exception:
            return {}

    def undo_write(self, steps: int = 1) -> dict:
        if not self._ensure_ready() or self._session_obj is None:
            return {"error": "No active CodePrism session"}
        try:
            result = _run(self._session_obj.undo(steps=steps))
            return {
                "files_restored": getattr(result, "files_restored", []),
                "steps_undone": getattr(result, "steps_undone", 0),
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Security gate (standalone — for pre-write checks)
    # ------------------------------------------------------------------

    def scan_diff(self, original: str, proposed: str, file: str = "") -> dict:
        """Run security diff scan on proposed content changes.

        Returns: {status: PASS|WARN|BLOCK, new_issues: [...]}
        """
        if not self._ensure_ready():
            return {"status": "PASS", "new_issues": []}
        try:
            from codeprism.security import SecurityScanner
            scanner = SecurityScanner()
            report = scanner.scan_diff(original, proposed, file)
            return {
                "status": report.status,
                "new_issues": [
                    {
                        "severity": i.severity.value if hasattr(i.severity, "value") else str(i.severity),
                        "category": getattr(i, "category", ""),
                        "line": getattr(i, "line_number", None),
                        "description": i.description,
                        "fix": getattr(i, "fix_suggestion", ""),
                    }
                    for i in report.issues
                ],
            }
        except Exception as exc:
            return {"status": "PASS", "new_issues": [], "note": str(exc)}

    def close(self) -> None:
        if self._prism:
            try:
                _run(self._prism.close(), timeout=5)
            except Exception:
                pass
            self._prism = None
