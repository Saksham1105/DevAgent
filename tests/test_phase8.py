"""Phase 8 tests: context auto-compression."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_cfg(
    auto_compress=True,
    compression_threshold=0.6,
    compression_window_size=5,
    compression_model="cheap",
):
    from devagent.core.config import SessionConfig
    return SessionConfig(
        auto_compress=auto_compress,
        compression_threshold=compression_threshold,
        compression_window_size=compression_window_size,
        compression_model=compression_model,
    )


def _make_llm_cfg(provider="ollama", model="qwen2.5-coder:7b"):
    from devagent.core.config import LLMConfig
    return LLMConfig(provider=provider, model=model)


def _seed_events(session_id: str, n: int, db_path: Path) -> None:
    """Write n alternating user/assistant events into the DB."""
    from devagent.session import store
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        store.append_event(
            session_id,
            role=role,
            content=f"Event {i}: some content about file_path/module.py function_name error_message",
            db_path=db_path,
        )


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_session_config_compression_defaults():
    from devagent.core.config import SessionConfig
    cfg = SessionConfig()
    assert cfg.auto_compress is True
    assert cfg.compression_threshold == 0.45
    assert cfg.compression_window_size == 20
    assert cfg.compression_model == "cheap"


# ---------------------------------------------------------------------------
# 2. Store — schema migration & helpers
# ---------------------------------------------------------------------------

def test_init_schema_creates_compressed_summary_column(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    store.init_schema(db_path=db)
    # verify column exists by writing and reading
    sid = "sess-abc"
    store.create_session(sid, db_path=db)
    store.save_compressed_summary(sid, "test summary", db_path=db)
    result = store.get_compressed_summary(sid, db_path=db)
    assert result == "test summary"


def test_get_compressed_summary_returns_empty_when_none(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    store.init_schema(db_path=db)
    store.create_session("s1", db_path=db)
    assert store.get_compressed_summary("s1", db_path=db) == ""


def test_mark_events_compressed_flags_correct_rows(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    store.init_schema(db_path=db)
    sid = "sess-mark"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 6, db)

    all_events = store.get_events(sid, db_path=db)
    seqs = [e["seq"] for e in all_events]
    # Mark first 3 as compressed
    store.mark_events_compressed(sid, max_seq=seqs[2], db_path=db)

    active = store.get_active_events(sid, db_path=db)
    assert len(active) == 3
    assert all(e["is_compressed"] == 0 for e in active)


def test_get_active_events_returns_all_when_none_compressed(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    store.init_schema(db_path=db)
    sid = "sess-active"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 4, db)
    active = store.get_active_events(sid, db_path=db)
    assert len(active) == 4


def test_count_compressed_events(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    store.init_schema(db_path=db)
    sid = "sess-count"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 8, db)
    all_events = store.get_events(sid, db_path=db)
    store.mark_events_compressed(sid, all_events[3]["seq"], db_path=db)
    assert store.count_compressed_events(sid, db_path=db) == 4


# ---------------------------------------------------------------------------
# 3. History — build_messages with / without summary
# ---------------------------------------------------------------------------

def test_build_messages_no_summary_uses_all_events(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.history import build_messages
    store.init_schema(db_path=db)
    sid = "sess-hist"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 4, db)

    msgs = build_messages(sid, "sys-prompt", db_path=db)
    # 1 system + 4 events
    assert len(msgs) == 5
    assert msgs[0].role == "system"
    assert msgs[0].content == "sys-prompt"


def test_build_messages_with_summary_injects_it_into_system(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.history import build_messages
    store.init_schema(db_path=db)
    sid = "sess-sum"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 6, db)

    # Compress first 3 events
    all_events = store.get_events(sid, db_path=db)
    store.mark_events_compressed(sid, all_events[2]["seq"], db_path=db)
    store.save_compressed_summary(sid, "Compressed: read module.py, found bug.", db_path=db)

    msgs = build_messages(sid, "sys-prompt", db_path=db)
    # 1 system + 3 active events
    assert len(msgs) == 4
    assert "compressed summary" in msgs[0].content.lower()
    assert "Compressed: read module.py" in msgs[0].content


def test_build_messages_summary_not_replayed_as_events(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.history import build_messages
    store.init_schema(db_path=db)
    sid = "sess-noreplay"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 8, db)

    all_events = store.get_events(sid, db_path=db)
    store.mark_events_compressed(sid, all_events[4]["seq"], db_path=db)
    store.save_compressed_summary(sid, "Old summary.", db_path=db)

    msgs = build_messages(sid, "sys", db_path=db)
    roles = [m.role for m in msgs]
    # Should not have more than 1 system message
    assert roles.count("system") == 1
    # 1 system + 3 remaining active events
    assert len(msgs) == 4


# ---------------------------------------------------------------------------
# 4. Compressor — compress_session logic
# ---------------------------------------------------------------------------

def _make_mock_llm(summary_text="Summary: did stuff."):
    from devagent.core.llm import LLMConfig, LLMResponse
    mock = MagicMock()
    mock.cfg = LLMConfig(provider="ollama", model="qwen2.5-coder:7b")
    mock.complete_with_tools.return_value = LLMResponse(
        content=summary_text,
        input_tokens=100,
        output_tokens=50,
    )
    return mock


def test_compress_session_returns_none_when_too_few_events(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import compress_session
    store.init_schema(db_path=db)
    sid = "sess-short"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 3, db)

    llm = _make_mock_llm()
    result = compress_session(sid, llm, keep_last_n=5, db_path=db)
    assert result is None
    llm.complete_with_tools.assert_not_called()


def test_compress_session_summarises_old_events(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import compress_session
    store.init_schema(db_path=db)
    sid = "sess-comp"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 10, db)

    llm = _make_mock_llm("Compressed: 5 events summarised.")
    result = compress_session(sid, llm, keep_last_n=5, db_path=db)

    assert result is not None
    assert result.events_compressed == 5
    assert result.events_remaining == 5
    assert "Compressed: 5 events summarised." in result.summary
    llm.complete_with_tools.assert_called_once()


def test_compress_session_marks_events_in_db(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import compress_session
    store.init_schema(db_path=db)
    sid = "sess-flag"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 8, db)

    llm = _make_mock_llm()
    compress_session(sid, llm, keep_last_n=3, db_path=db)

    assert store.count_compressed_events(sid, db_path=db) == 5
    assert len(store.get_active_events(sid, db_path=db)) == 3


def test_compress_session_persists_summary(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import compress_session
    store.init_schema(db_path=db)
    sid = "sess-persist"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 6, db)

    llm = _make_mock_llm("My summary.")
    compress_session(sid, llm, keep_last_n=2, db_path=db)

    saved = store.get_compressed_summary(sid, db_path=db)
    assert "My summary." in saved


def test_compress_session_incremental_appends(tmp_path):
    """Second compression appends to the first summary, not overwrites."""
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import compress_session
    store.init_schema(db_path=db)
    sid = "sess-incr"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 10, db)

    llm = _make_mock_llm("Round 1 summary.")
    compress_session(sid, llm, keep_last_n=5, db_path=db)

    # Add more events so there's something to compress in round 2
    _seed_events(sid, 8, db)
    llm2 = _make_mock_llm("Round 2 summary.")
    compress_session(sid, llm2, keep_last_n=5, db_path=db)

    saved = store.get_compressed_summary(sid, db_path=db)
    assert "Round 1 summary." in saved
    assert "Round 2 summary." in saved


# ---------------------------------------------------------------------------
# 5. maybe_compress — threshold gating
# ---------------------------------------------------------------------------

def test_maybe_compress_skips_when_below_threshold(tmp_path):
    db = tmp_path / "test.db"
    from devagent.session import store
    from devagent.session.compressor import maybe_compress
    store.init_schema(db_path=db)
    sid = "sess-thresh"
    store.create_session(sid, db_path=db)
    # Only 3 events — well below any threshold
    _seed_events(sid, 3, db)

    llm = _make_mock_llm()
    cfg = _make_session_cfg(compression_window_size=5)
    result = maybe_compress(sid, llm, cfg, db_path=db)
    assert result is None
    llm.complete_with_tools.assert_not_called()


def test_maybe_compress_triggers_when_over_threshold(tmp_path):
    """Patch _estimate_tokens and _context_window_for to force triggering."""
    db = tmp_path / "test.db"
    from devagent.session import compressor as comp_mod
    from devagent.session import store
    store.init_schema(db_path=db)
    sid = "sess-over"
    store.create_session(sid, db_path=db)
    _seed_events(sid, 12, db)

    llm = _make_mock_llm("Triggered summary.")
    cfg = _make_session_cfg(compression_window_size=5, compression_threshold=0.001)

    # Force a tiny context window so the threshold is exceeded
    with patch.object(comp_mod, "_context_window_for", return_value=10):
        result = comp_mod.maybe_compress(sid, llm, cfg, db_path=db)

    assert result is not None
    assert result.events_compressed > 0


# ---------------------------------------------------------------------------
# 6. Context window lookup
# ---------------------------------------------------------------------------

def test_context_window_lookup_claude():
    from devagent.session.compressor import _context_window_for
    assert _context_window_for("anthropic", "claude-sonnet-4-6") == 200_000


def test_context_window_lookup_gpt4o():
    from devagent.session.compressor import _context_window_for
    assert _context_window_for("openai", "gpt-4o") == 128_000


def test_context_window_lookup_ollama_fallback():
    from devagent.session.compressor import _context_window_for
    # Unknown model falls back to the "ollama" key or "default"
    result = _context_window_for("ollama", "unknown-model-xyz")
    assert result > 0


def test_default_compression_threshold_limits_history_growth():
    from devagent.core.config import SessionConfig

    assert SessionConfig().compression_threshold == 0.45


def test_context_window_lookup_default():
    from devagent.session.compressor import _context_window_for
    result = _context_window_for("unknown-provider", "unknown-model")
    assert result == 32_000


# ---------------------------------------------------------------------------
# 7. CLI — session compress command importable and has correct signature
# ---------------------------------------------------------------------------

def test_session_compress_command_exists():
    import inspect

    from devagent.cli import session_compress
    sig = inspect.signature(session_compress)
    assert "session_id" in sig.parameters
    assert "keep" in sig.parameters


def test_fit_request_context_counts_tools_and_keeps_newest_messages():
    from devagent.core.llm import AgentMessage, ToolDef
    from devagent.session.history import fit_request_context

    messages = [
        AgentMessage(role="system", content="system prompt"),
        AgentMessage(role="user", content="old " * 500),
        AgentMessage(role="assistant", content="middle " * 500),
        AgentMessage(role="user", content="latest request"),
    ]
    tools = [ToolDef(name="read_file", description="x" * 100, parameters={"type": "object"})]
    fitted, fitted_tools = fit_request_context(messages, tools, max_tokens=100)

    assert fitted_tools == tools
    assert fitted[0].role == "system"
    assert "latest request" in fitted[-1].content
    assert len(fitted) < len(messages)
