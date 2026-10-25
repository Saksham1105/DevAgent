"""Phase 6 tests: REST server, benchmark scripts, plugin tool interface."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

# ---------------------------------------------------------------------------
# Helper — find a free port
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# REST server — create_server + endpoint tests
# ---------------------------------------------------------------------------

def _start_server(port: int, config=None, project_root: str = ".") -> threading.Thread:
    from devagent.server.app import create_server
    srv = create_server("127.0.0.1", port, config=config, project_root=project_root)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)  # let the server bind
    return t


def _get(port: int, path: str) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return resp.status, body


def test_server_health():
    port = _free_port()
    _start_server(port)
    status, body = _get(port, "/api/health")
    assert status == 200
    assert body["status"] == "ok"
    assert "version" in body


def test_server_status_no_config():
    port = _free_port()
    _start_server(port, config=None)
    status, body = _get(port, "/api/status")
    assert status == 503
    assert "error" in body


def test_server_status_with_config():
    from devagent.core.config import DevAgentConfig, LLMConfig
    port = _free_port()
    cfg = DevAgentConfig(llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"))
    _start_server(port, config=cfg)
    status, body = _get(port, "/api/status")
    assert status == 200
    assert body["llm"]["provider"] == "ollama"
    assert body["llm"]["offline_capable"] is True


def test_server_status_cloud_not_offline():
    from devagent.core.config import DevAgentConfig, LLMConfig
    port = _free_port()
    cfg = DevAgentConfig(llm=LLMConfig(provider="anthropic", model="claude-3-haiku"))
    _start_server(port, config=cfg)
    status, body = _get(port, "/api/status")
    assert status == 200
    assert body["llm"]["offline_capable"] is False


def test_server_sessions_endpoint():
    port = _free_port()
    _start_server(port)
    status, body = _get(port, "/api/sessions")
    assert status == 200
    assert "sessions" in body
    assert isinstance(body["sessions"], list)


def test_server_tools_endpoint():
    port = _free_port()
    _start_server(port)
    status, body = _get(port, "/api/tools")
    assert status == 200
    assert "tools" in body
    assert isinstance(body["tools"], list)
    assert body["count"] >= 10  # at least file + shell + git + search tools
    tool_names = [t["name"] for t in body["tools"]]
    assert "read_file" in tool_names
    assert "run_shell" in tool_names
    assert "grep" in tool_names


def test_server_unknown_route():
    port = _free_port()
    _start_server(port)
    status, body = _get(port, "/api/nonexistent")
    assert status == 404
    assert "error" in body


def test_server_graph_stats_not_indexed():
    port = _free_port()
    _start_server(port, project_root=".")
    status, body = _get(port, "/api/graph/stats")
    # Either 404 (not indexed) or 501 (codeprism not installed)
    assert status in (404, 501)
    assert "error" in body


def test_server_graph_files_not_indexed():
    port = _free_port()
    _start_server(port, project_root=".")
    status, body = _get(port, "/api/graph/files")
    assert status in (404, 501)
    assert "error" in body


def test_server_session_not_found():
    port = _free_port()
    _start_server(port)
    status, body = _get(port, "/api/sessions/nonexistent-session-id-xyz")
    assert status == 404
    assert "error" in body


def test_server_session_detail_exact_match_and_prefix_rejection():
    from devagent.session import store as session_store

    port = _free_port()
    _start_server(port)

    prefix_id = f"test-prefix-{int(time.time() * 1000)}"
    full_id = f"{prefix_id}-extended"

    session_store.init_schema()
    session_store.create_session(
        full_id,
        project="/test/project",
        model="qwen2.5-coder:7b",
        provider="ollama",
        title="Exact match test session",
    )

    try:
        # Prefix lookup must return 404 rather than matching full_id
        status, body = _get(port, f"/api/sessions/{prefix_id}")
        assert status == 404
        assert "error" in body

        # Exact ID lookup succeeds and returns the session details
        status, body = _get(port, f"/api/sessions/{full_id}")
        assert status == 200
        assert body["session"]["id"] == full_id
        assert "events" in body
        assert "token_totals" in body
    finally:
        try:
            session_store.delete_session(full_id)
        except Exception:
            pass


def test_server_cors_headers():
    port = _free_port()
    _start_server(port)
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("OPTIONS", "/api/health")
    resp = conn.getresponse()
    conn.close()
    assert resp.status == 200
    # CORS header should be set
    headers = dict(resp.getheaders())
    assert "Access-Control-Allow-Origin" in headers


def test_server_json_content_type():
    port = _free_port()
    _start_server(port)
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/api/health")
    resp = conn.getresponse()
    ct = resp.getheader("Content-Type", "")
    conn.close()
    assert "application/json" in ct


# ---------------------------------------------------------------------------
# Benchmark scripts — import + run without crashing
# ---------------------------------------------------------------------------

def _load_bench(name: str):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).parent.parent / "benchmarks" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Must register in sys.modules BEFORE exec_module so that @dataclass
    # decorators can look up the module's __dict__ (required on Python 3.13+).
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def test_bench_token_usage_runs():
    mod = _load_bench("bench_token_usage")
    results = mod.run_benchmark()
    assert len(results) == 5
    for r in results:
        assert r.saved_tokens > 0
        assert 0 < r.savings_pct < 100


def test_bench_security_runs():
    mod = _load_bench("bench_security")
    results = mod.run_benchmark()
    assert len(results) == 20
    correct = sum(1 for r in results if r.correct)
    # Expect at least 80% accuracy from our pattern-based benchmark
    assert correct >= 16, f"Only {correct}/20 correct"


def test_bench_tasks_runs():
    mod = _load_bench("bench_tasks")
    results = mod.run_benchmark()
    assert len(results) == 10
    completed = sum(1 for r in results if r.completed)
    # At least 9/10 should complete with the mock LLM
    assert completed >= 9, f"Only {completed}/10 tasks completed"


# ---------------------------------------------------------------------------
# Plugin tool interface contract
# ---------------------------------------------------------------------------

def test_plugin_tool_register_and_call():
    from devagent.tools.registry import ToolRegistry

    registry = ToolRegistry()

    def my_handler(args: dict) -> str:
        name = args.get("name", "world")
        return f"Hello, {name}!"

    registry.register(
        name="greet",
        description="Greet someone by name.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"},
            },
            "required": ["name"],
        },
        handler=my_handler,
    )

    assert "greet" in registry.names()
    result = registry.call("greet", {"name": "DevAgent"})
    assert result == "Hello, DevAgent!"


def test_plugin_tool_error_does_not_raise():
    from devagent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        "boom", "Always explodes", {"type": "object", "properties": {}},
        lambda args: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    result = registry.call("boom", {})
    assert "[tool_error]" in result
    assert "kaboom" in result


def test_plugin_tool_unknown_name():
    from devagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    result = registry.call("no_such_tool", {})
    assert "[tool_error]" in result


def test_plugin_tool_get_definitions():
    from devagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.register("t1", "Tool one", {"type": "object", "properties": {}}, lambda a: "ok")
    registry.register("t2", "Tool two", {"type": "object", "properties": {}}, lambda a: "ok")
    defs = registry.get_definitions()
    names = [d.name for d in defs]
    assert "t1" in names and "t2" in names


def test_serve_command_importable():
    """The serve CLI command must be importable (wires the server correctly)."""
    from devagent.cli import app
    # CommandInfo.name may be None when not explicitly set; fall back to callback name
    def _name(c):
        return c.name if c.name is not None else (c.callback.__name__ if c.callback else "")
    cmds = [_name(c) for c in app.registered_commands]
    assert "serve" in cmds


# ---------------------------------------------------------------------------
# CI workflow files exist
# ---------------------------------------------------------------------------

def test_ci_workflow_exists():
    ci = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    assert ci.exists(), "ci.yml not found"
    content = ci.read_text(encoding="utf-8")
    assert "pytest" in content
    assert "ruff" in content


def test_publish_workflow_exists():
    pub = Path(__file__).parent.parent / ".github" / "workflows" / "publish.yml"
    assert pub.exists()
    content = pub.read_text(encoding="utf-8")
    assert "pypi" in content.lower() or "PyPI" in content


def test_issue_templates_exist():
    tpl_dir = Path(__file__).parent.parent / ".github" / "ISSUE_TEMPLATE"
    assert (tpl_dir / "bug_report.yml").exists()
    assert (tpl_dir / "feature_request.yml").exists()


def test_contributing_md_exists():
    assert (Path(__file__).parent.parent / "CONTRIBUTING.md").exists()


def test_plugin_tools_doc_exists():
    assert (Path(__file__).parent.parent / "docs" / "plugin_tools.md").exists()
