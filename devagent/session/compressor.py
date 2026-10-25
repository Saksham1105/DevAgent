"""Phase 8 — Context auto-compression.

When a session's event history grows large enough to crowd the LLM context
window, this module summarises the older portion into a compact text block
and marks those events as compressed in the DB.  Subsequent turns only replay
the most-recent (hot) events, with the summary injected into the system prompt.

Compression is lossless for facts: the LLM is explicitly instructed to preserve
every file path, function name, decision, and error message verbatim.  Narrative
and reasoning are summarised to dense statements.

Usage
-----
From DevAgentSession (automatic):
    result = maybe_compress(session_id, llm, cfg.session, db_path)

Manual via CLI:
    devagent session compress <session-id>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devagent.core.llm import AgentMessage, LLMClient
from devagent.session import store

# Rough token-window sizes per provider/model pattern.
# Used to compute the absolute threshold in tokens when only a fraction is given.
_CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "claude": 200_000,
    "gpt-4o": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
    "gemini": 128_000,
    "qwen2.5-coder:14b": 32_000,
    "qwen2.5-coder:7b": 32_000,
    "qwen2.5-coder:3b": 8_000,
    "ollama": 32_000,   # generic fallback for unknown Ollama models
    "default": 32_000,
}

_SUMMARISE_SYSTEM = """\
You are a session summariser for an AI coding agent.
Your job is to compress a list of past conversation events into a compact,
dense summary that preserves EVERY factual detail a developer needs.

Rules:
- Preserve VERBATIM: file paths, function/class names, error messages, decisions made,
  test results, branch names, commit hashes, and any specific values.
- Summarise CONCISELY: reasoning steps, exploratory reads, repeated failures.
- Deduplicate: if the same file was read 5 times, say "read X (×5)".
- Use bullet points. One bullet per logical action or finding.
- End with a "## Open items" section listing anything unresolved at the end of this window.
- Do NOT invent information. If uncertain, omit rather than guess.
- Target length: 300–600 tokens.
"""


@dataclass
class CompressionResult:
    summary: str
    events_compressed: int   # number of old events folded into the summary
    events_remaining: int    # hot-window events still in full form
    tokens_saved: int        # rough estimate of tokens freed


def _estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _estimate_event_tokens(events: list[dict[str, Any]]) -> int:
    """Estimate the full serialized event payload sent back to the model."""
    total = 0
    for event in events:
        total += len(event.get("role", ""))
        total += len(event.get("content") or "")
        total += len(str(event.get("tool_calls") or []))
        total += len(event.get("tool_name") or "")
        total += len(event.get("tool_call_id") or "")
    return _estimate_tokens("x" * total)


def _context_window_for(provider: str, model: str) -> int:
    """Return the estimated context window size for the given provider/model."""
    model_lower = model.lower()
    for key, size in _CONTEXT_WINDOW_TOKENS.items():
        if key in model_lower:
            return size
    provider_lower = provider.lower()
    for key, size in _CONTEXT_WINDOW_TOKENS.items():
        if key in provider_lower:
            return size
    return _CONTEXT_WINDOW_TOKENS["default"]


def _format_events_for_summary(events: list[dict[str, Any]]) -> str:
    """Convert raw DB event rows into a readable text block for the LLM."""
    lines: list[str] = []
    for ev in events:
        role = ev["role"]
        content = (ev.get("content") or "").strip()
        tool_calls: list[dict] = ev.get("tool_calls") or []

        if role == "user":
            lines.append(f"[USER] {content[:500]}")
        elif role == "assistant":
            if content:
                lines.append(f"[ASSISTANT] {content[:300]}")
            for tc in tool_calls:
                args_str = str(tc.get("args", {}))[:200]
                lines.append(f"  → tool_call: {tc.get('name')}({args_str})")
        elif role == "tool_result":
            tool_name = ev.get("tool_name", "")
            lines.append(f"  ← {tool_name}: {content[:300]}")

    return "\n".join(lines)


def maybe_compress(
    session_id: str,
    llm: LLMClient,
    session_cfg: Any,   # SessionConfig
    db_path: Path | None = None,
) -> CompressionResult | None:
    """Auto-compress if history tokens exceed the configured threshold.

    Returns a CompressionResult if compression happened, else None.
    """
    all_events = store.get_events(session_id, db_path=db_path)
    keep_n = session_cfg.compression_window_size

    if len(all_events) <= keep_n:
        return None

    # Estimate total history tokens
    history_tokens = _estimate_event_tokens(all_events)

    context_window = _context_window_for(llm.cfg.provider, llm.cfg.model)
    threshold_tokens = int(context_window * session_cfg.compression_threshold)

    if history_tokens < threshold_tokens:
        return None  # not crowded yet

    return compress_session(
        session_id=session_id,
        llm=llm,
        keep_last_n=keep_n,
        db_path=db_path,
    )


def compress_session(
    session_id: str,
    llm: LLMClient,
    keep_last_n: int = 20,
    db_path: Path | None = None,
    focus: str | None = None,
) -> CompressionResult | None:
    """Summarise events older than the hot window and persist the result.

    focus: optional topic hint — the summariser is asked to preserve content
    related to this topic in extra detail.

    Returns None if there is nothing old enough to compress.
    """
    all_events = store.get_events(session_id, db_path=db_path)

    if len(all_events) <= keep_last_n:
        return None

    to_compress = all_events[:-keep_last_n]
    hot_events = all_events[-keep_last_n:]

    # Token estimate before compression
    old_text = _format_events_for_summary(to_compress)
    tokens_before = _estimate_tokens(old_text)

    # Build the LLM prompt
    focus_line = (
        f"\nPay special attention to content related to: {focus}\n"
        "Preserve that content in extra detail even if it would normally be abbreviated."
        if focus else ""
    )
    user_content = (
        "Summarise the following conversation events from an AI coding session.\n"
        f"Follow your summarisation rules exactly.{focus_line}\n\n"
        "--- EVENTS TO SUMMARISE ---\n"
        f"{old_text}\n"
        "--- END ---"
    )

    messages = [
        AgentMessage(role="system", content=_SUMMARISE_SYSTEM),
        AgentMessage(role="user", content=user_content),
    ]

    try:
        response = llm.complete_with_tools(messages, tools=[])
        summary = (response.content or "").strip()
    except Exception as exc:
        summary = f"[Compression failed: {exc}]\n\n" + old_text[:2000]

    if not summary:
        summary = old_text[:2000]  # fallback: truncated raw text

    # Append to any existing summary (incremental compression)
    existing = store.get_compressed_summary(session_id, db_path=db_path)
    if existing:
        summary = existing + "\n\n---\n\n" + summary

    store.save_compressed_summary(session_id, summary, db_path=db_path)

    max_compressed_seq = to_compress[-1]["seq"]
    store.mark_events_compressed(session_id, max_compressed_seq, db_path=db_path)

    tokens_after = _estimate_tokens(summary)

    return CompressionResult(
        summary=summary,
        events_compressed=len(to_compress),
        events_remaining=len(hot_events),
        tokens_saved=max(0, tokens_before - tokens_after),
    )
