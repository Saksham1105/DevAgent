"""DevAgent REST server — stdlib HTTP, no extra dependencies.

Endpoints:
  GET  /api/health          — liveness check
  GET  /api/status          — config + feature flags
  GET  /api/sessions        — recent sessions (last 20)
  GET  /api/sessions/<id>   — session detail + token totals
  GET  /api/tools           — registered tool names + descriptions
  GET  /api/graph/stats     — CodePrism graph stats (if indexed)
  GET  /api/graph/files     — CodePrism file map (if indexed)

This server is intentionally minimal — it is a stub designed to power a future
graph-visualisation UI.  All endpoints are read-only.  The server runs in the
foreground; stop it with Ctrl-C.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_VERSION = "0.4.0-dev"


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Single-threaded JSON handler.  Subclasses inject config + project_root."""

    # Set by create_server()
    config = None
    project_root: Path = Path(".")

    def log_message(self, fmt: str, *args) -> None:  # silence default access log
        pass

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")

        routes: dict[str, Any] = {
            "/api/health":       self._health,
            "/api/status":       self._status,
            "/api/sessions":     self._sessions,
            "/api/tools":        self._tools,
            "/api/graph/stats":  self._graph_stats,
            "/api/graph/files":  self._graph_files,
        }

        # /api/sessions/<id>
        if path.startswith("/api/sessions/") and path != "/api/sessions/":
            session_id = path.split("/")[-1]
            self._session_detail(session_id)
            return

        handler = routes.get(path)
        if handler is None:
            self._json({"error": f"No route: {path}"}, status=404)
        else:
            handler()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _health(self) -> None:
        self._json({"status": "ok", "version": _VERSION})

    def _status(self) -> None:
        cfg = self.config
        if cfg is None:
            self._json({"error": "no config loaded"}, status=503)
            return

        cp_indexed = False
        try:
            from devagent.codeprism.client import CodePrismClient
            cp = CodePrismClient(str(self.project_root))
            cp_indexed = cp.is_indexed
        except Exception:
            pass

        self._json({
            "version": _VERSION,
            "project": str(self.project_root),
            "llm": {
                "provider": cfg.llm.provider,
                "model": cfg.llm.model,
                "offline_capable": cfg.llm.provider == "ollama",
            },
            "github": {"configured": bool(getattr(cfg.github, "token", ""))},
            "graph": {"indexed": cp_indexed},
        })

    def _sessions(self) -> None:
        try:
            from devagent.session.manager import SessionManager
            mgr = SessionManager()
            sessions = mgr.list(limit=20)
            self._json({"sessions": sessions})
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _session_detail(self, session_id: str) -> None:
        try:
            from devagent.session.manager import SessionManager
            mgr = SessionManager()
            session = mgr.get(session_id)
            if session is None:
                self._json({"error": f"session not found: {session_id}"}, status=404)
                return
            events = mgr.get_events(session["id"])
            totals = mgr.get_token_totals(session["id"])
            self._json({"session": session, "events": events, "token_totals": totals})
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _tools(self) -> None:
        try:
            from devagent.tools.registry import build_registry
            registry = build_registry(str(self.project_root))
            tools = [
                {"name": t.name, "description": t.description}
                for t in registry.get_definitions()
            ]
            self._json({"tools": tools, "count": len(tools)})
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _graph_stats(self) -> None:
        try:
            from devagent.codeprism.client import CodePrismClient
            cp = CodePrismClient(str(self.project_root))
            if not cp.is_indexed:
                self._json({"error": "project not indexed — run: codeprism index ."}, status=404)
                return
            self._json(cp.get_stats())
        except ImportError:
            self._json({"error": "codeprism-ai not installed"}, status=501)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _graph_files(self) -> None:
        try:
            from devagent.codeprism.client import CodePrismClient
            cp = CodePrismClient(str(self.project_root))
            if not cp.is_indexed:
                self._json({"error": "project not indexed — run: codeprism index ."}, status=404)
                return
            self._json(cp.get_file_map())
        except ImportError:
            self._json({"error": "codeprism-ai not installed"}, status=501)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_server(
    host: str = "127.0.0.1",
    port: int = 7331,
    config=None,
    project_root: str | Path = ".",
) -> HTTPServer:
    """Build an HTTPServer with config injected into the handler class."""

    # Create a fresh handler subclass so multiple servers in tests don't share state
    handler_cls = type("_BoundHandler", (_Handler,), {
        "config": config,
        "project_root": Path(project_root).resolve(),
    })
    server = HTTPServer((host, port), handler_cls)
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 7331,
    config=None,
    project_root: str | Path = ".",
    open_ui: bool = False,
) -> None:
    """Start the server in the foreground. Blocks until Ctrl-C."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    server = create_server(host, port, config, project_root)

    console.print(Panel(
        f"[bold cyan]DevAgent REST server[/bold cyan]\n"
        f"Listening on [link]http://{host}:{port}[/link]\n\n"
        f"[dim]Endpoints:[/dim]\n"
        f"  [dim]GET /api/health[/dim]\n"
        f"  [dim]GET /api/status[/dim]\n"
        f"  [dim]GET /api/sessions[/dim]\n"
        f"  [dim]GET /api/tools[/dim]\n"
        f"  [dim]GET /api/graph/stats[/dim]\n"
        f"  [dim]GET /api/graph/files[/dim]\n\n"
        f"[dim]Press Ctrl-C to stop.[/dim]",
        border_style="cyan",
    ))

    if open_ui:
        console.print(
            "[yellow]Note: the graph visualisation UI is not yet bundled. "
            "Point your browser at the API directly for now.[/yellow]"
        )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    finally:
        server.server_close()
