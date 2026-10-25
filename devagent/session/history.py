"""Convert stored session events to AgentMessage lists for the LLM.

Handles the full set of roles: system, user, assistant (with tool calls), tool_result.

Phase 8: When a compressed summary exists for a session, only the hot-window
events (is_compressed=0) are replayed.  The summary is appended to the system
prompt so the LLM retains context from earlier turns without paying the full
token cost of the original event log.
"""

from __future__ import annotations

import json

from devagent.core.llm import AgentMessage, ToolCallRequest
from devagent.session import store


def build_messages(
    session_id: str,
    system_prompt: str,
    db_path=None,
) -> list[AgentMessage]:
    """Reconstruct the conversation as AgentMessage objects for the LLM.

    If a compressed summary exists the system prompt is augmented with it
    and only uncompressed (hot-window) events are replayed — reducing the
    number of tokens sent on every turn for long sessions.
    """
    compressed_summary = store.get_compressed_summary(session_id, db_path=db_path)

    if compressed_summary:
        # Inject summary into system prompt; load only the hot window
        full_system = (
            system_prompt
            + "\n\n## Context from earlier in this session (compressed summary):\n"
            + compressed_summary
        )
        events = store.get_active_events(session_id, db_path=db_path)
    else:
        full_system = system_prompt
        events = store.get_events(session_id, db_path=db_path)

    messages: list[AgentMessage] = [
        AgentMessage(role="system", content=full_system)
    ]

    for ev in events:
        role = ev["role"]
        content = ev["content"]
        raw_tool_calls = ev.get("tool_calls") or []

        if role == "tool_result":
            messages.append(AgentMessage(
                role="tool_result",
                content=content,
                tool_call_id=ev.get("tool_call_id", ""),
                tool_name=ev.get("tool_name", ""),
            ))
        elif role == "assistant":
            tool_calls = [
                ToolCallRequest(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    args=tc.get("args", {}),
                )
                for tc in raw_tool_calls
            ]
            messages.append(AgentMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            ))
        else:
            messages.append(AgentMessage(role=role, content=content))

    return messages


def estimate_tokens(messages: list[AgentMessage]) -> int:
    """Estimate message tokens, including tool-call arguments."""
    total = 0
    for message in messages:
        total += len(message.content or "")
        total += sum(len(json.dumps(call.args, ensure_ascii=False)) for call in message.tool_calls)
        total += len(message.tool_call_id or "") + len(message.tool_name or "")
    return (total + 3) // 4


def estimate_request_tokens(messages: list[AgentMessage], tools: list | None = None) -> int:
    """Estimate the prompt payload, including serialized tool definitions."""
    total = estimate_tokens(messages) * 4
    for tool in tools or []:
        total += len(tool.name) + len(tool.description)
        total += len(json.dumps(tool.parameters or {}, ensure_ascii=False))
    return (total + 3) // 4


def fit_request_context(
    messages: list[AgentMessage],
    tools: list,
    max_tokens: int,
) -> tuple[list[AgentMessage], list]:
    """Keep the system prompt and newest messages within a request budget.

    Tool definitions are retained because removing them can make tool calls
    invalid. Older conversation messages are dropped first; durable history
    remains in SQLite and can be compressed for later turns.
    """
    if estimate_request_tokens(messages, tools) <= max_tokens:
        return messages, tools

    system = messages[:1]
    retained: list[AgentMessage] = []
    for message in reversed(messages[1:]):
        candidate = system + [message] + list(reversed(retained))
        if estimate_request_tokens(candidate, tools) > max_tokens:
            break
        retained.append(message)

    retained.reverse()
    result = system + retained
    if len(result) == 1 and result[0].content:
        tool_tokens = estimate_request_tokens([], tools)
        allowed_chars = max(0, (max_tokens - tool_tokens) * 4)
        result[0] = AgentMessage(
            role="system",
            content=result[0].content[:allowed_chars],
        )
    return result, tools
