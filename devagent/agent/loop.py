"""ReAct agent loop.

Drives the Reason + Act cycle:
  1. Build messages from history
  2. Call LLM (with tools) — via MultiModelRouter when available
  3. If the LLM requests tool calls, execute them and loop
  4. If the LLM produces a final text response, yield it and stop

The loop is a sync generator that yields AgentEvent objects so the
caller (CLI) can stream output while driving the loop.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devagent.agent.permissions import PermissionManager
from devagent.core.llm import LLMClient
from devagent.session.budget import BudgetExceeded, TokenBudget
from devagent.session.history import build_messages, fit_request_context
from devagent.session.manager import SessionManager
from devagent.session.memory import MemoryBlock
from devagent.tools.registry import ToolRegistry

# Optional — only present when CodePrism is integrated
try:
    from devagent.codeprism.client import CodePrismClient  # noqa: F401
    from devagent.codeprism.session_overlay import build_session_overlay
    _HAS_CODEPRISM = True
except ImportError:
    _HAS_CODEPRISM = False

# Optional — MultiModelRouter
try:
    from devagent.core.router import MultiModelRouter  # noqa: F401
    _HAS_ROUTER = True
except ImportError:
    _HAS_ROUTER = False


_DEFAULT_MAX_ITERATIONS = 30  # used when no value passed to constructor
MAX_ITERATIONS = _DEFAULT_MAX_ITERATIONS  # public alias kept for backward compatibility
MAX_REPAIR = 3                # max consecutive auto-test retries after a write


# ---------------------------------------------------------------------------
# Event types emitted by the loop
# ---------------------------------------------------------------------------

@dataclass
class ThinkingEvent:
    """LLM produced reasoning text before (or without) tool calls."""
    text: str


@dataclass
class ToolCallEvent:
    """LLM is calling a tool."""
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Tool execution completed."""
    id: str
    name: str
    result: str
    success: bool = True
    diff: str = ""  # unified diff text (Phase 13); empty when not a file edit


@dataclass
class FinalAnswerEvent:
    """LLM has produced a final text answer (no more tool calls)."""
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class BudgetWarningEvent:
    """Token budget is running low."""
    used: int
    remaining: int | None
    limit: int | None


@dataclass
class StatusEvent:
    """Live status bar update — emitted after each LLM call."""
    status_line: str      # e.g. "tokens: 4,200 | cost: $0.0021 | calls: 3"
    task: str = ""        # router task name, e.g. "coding"
    iteration: int = 0


@dataclass
class ErrorEvent:
    """Unrecoverable error."""
    message: str


@dataclass
class ApprovalNeededEvent:
    """Permission gate requires user approval before tool execution."""
    call_id: str
    tool_name: str
    args: dict
    primary_arg: str = ""   # the path / command being acted on


AgentEvent = (
    ThinkingEvent
    | ToolCallEvent
    | ToolResultEvent
    | FinalAnswerEvent
    | BudgetWarningEvent
    | StatusEvent
    | ErrorEvent
    | ApprovalNeededEvent
)

# Tool names that trigger the auto-test repair loop
_WRITE_TOOL_NAMES = {"write_file", "edit_file"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        session_mgr: SessionManager,
        session_id: str,
        memory: MemoryBlock,
        budget: TokenBudget,
        system_prompt: str,
        codeprism_client=None,        # CodePrismClient | None
        router=None,                  # MultiModelRouter | None
        max_iterations: int = 0,      # 0 = use _DEFAULT_MAX_ITERATIONS; pass cfg value
        max_repair: int | None = None,  # None = use MAX_REPAIR; pass cfg.agent.max_repair_iterations
        loop_detection: bool = True,
        permission_mgr: PermissionManager | None = None,
        bare: bool = False,           # Phase 15: skip memory/overlay injection
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.session_mgr = session_mgr
        self.session_id = session_id
        self.memory = memory
        self.budget = budget
        self.system_prompt = system_prompt
        self._cp_client = codeprism_client
        self._router = router
        self._repair_attempts: dict[str, int] = {}  # test-repair attempts per file after write
        self._max_iterations = max_iterations if max_iterations > 0 else _DEFAULT_MAX_ITERATIONS
        self._max_repair = MAX_REPAIR if max_repair is None else max(0, max_repair)
        self._loop_detection = loop_detection
        self._permission_mgr = permission_mgr
        self._bare = bare
        # Loop detection: ring buffer of (tool_name, args_fingerprint) pairs
        self._recent_calls: list[tuple[str, str]] = []

    def run(self, user_message: str) -> Generator[AgentEvent, None, None]:
        """Drive one user turn through the ReAct loop.

        Yields AgentEvents as they happen. The caller consumes and renders them.
        """
        # Persist user message
        self.session_mgr.record_user(self.session_id, user_message)

        tools = self.registry.get_definitions()
        iteration = 0
        last_tool_names: list[str] = []

        # Reload full system prompt with latest memory + session overlay
        full_system = self.system_prompt
        if not self._bare:
            memory_block = self.memory.as_prompt_block()
            if memory_block:
                full_system = full_system + memory_block

        while iteration < self._max_iterations:
            iteration += 1

            # Refresh session overlay from CodePrism each turn (skipped in bare mode)
            current_system = full_system
            if not self._bare and _HAS_CODEPRISM and self._cp_client:
                overlay = build_session_overlay(self._cp_client)
                if overlay:
                    current_system = full_system + overlay

            # Select LLM for this iteration (router or default)
            task = "fallback"
            if _HAS_ROUTER and self._router is not None:
                current_llm, task = self._router.get_llm_for_iteration(last_tool_names, iteration)
            else:
                current_llm = self.llm

            db_path = getattr(self.session_mgr, "db_path", None)
            if not isinstance(db_path, (str, Path)):
                db_path = None
            messages = build_messages(
                self.session_id,
                current_system,
                db_path=db_path,
            )
            from devagent.session.compressor import _context_window_for
            provider = getattr(getattr(current_llm, "cfg", None), "provider", None) or "anthropic"
            model = getattr(getattr(current_llm, "cfg", None), "model", None) or "claude-sonnet-4-6"
            if not isinstance(provider, str):
                provider = "anthropic"
            if not isinstance(model, str):
                model = "claude-sonnet-4-6"
            context_limit = int(_context_window_for(provider, model) * 0.55)
            messages, tools = fit_request_context(messages, tools, context_limit)

            # Budget check
            try:
                self.budget.check()
            except BudgetExceeded as exc:
                yield ErrorEvent(str(exc))
                return

            # LLM call
            try:
                response = current_llm.complete_with_tools(messages, tools)
            except Exception as exc:
                yield ErrorEvent(f"LLM error: {exc}")
                return

            # Record token usage (with provider/model for cost tracking)
            self.budget.record(
                response.input_tokens,
                response.output_tokens,
                provider=current_llm.cfg.provider,
                model=current_llm.cfg.model,
            )

            # Emit live status bar
            yield StatusEvent(
                status_line=self.budget.status_line(),
                task=task,
                iteration=iteration,
            )

            # Check budget warning threshold
            threshold = self.budget.warn_threshold
            if threshold and self.budget.max_tokens:
                frac = self.budget.total_used / self.budget.max_tokens
                if frac >= threshold:
                    yield BudgetWarningEvent(
                        used=self.budget.total_used,
                        remaining=self.budget.remaining,
                        limit=self.budget.max_tokens,
                    )

            # Extended thinking blocks (Anthropic only, Phase 15)
            if response.thinking:
                yield ThinkingEvent(f"<thinking>\n{response.thinking}\n</thinking>")

            # Emit regular reasoning text if any
            if response.content:
                yield ThinkingEvent(response.content)

            if response.has_tool_calls:
                # Persist the assistant message with tool calls
                self.session_mgr.record_assistant(
                    self.session_id,
                    content=response.content,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "args": tc.args}
                        for tc in response.tool_calls
                    ],
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                )

                # Execute each tool call and track names for router
                last_tool_names = []
                for tc in response.tool_calls:
                    last_tool_names.append(tc.name)

                    # Loop detection: track recent (tool_name, args_fingerprint) pairs
                    if self._loop_detection:
                        fingerprint = str(sorted(tc.args.items()))[:120]
                        self._recent_calls.append((tc.name, fingerprint))
                        if len(self._recent_calls) > 8:
                            self._recent_calls.pop(0)
                        # Count occurrences of this exact call in the recent window
                        count = sum(
                            1 for n, f in self._recent_calls
                            if n == tc.name and f == fingerprint
                        )
                        if count >= 3:
                            yield ErrorEvent(
                                f"Loop detected: agent called '{tc.name}' with identical "
                                f"arguments {count} times in the last {len(self._recent_calls)} calls. "
                                "Stopping to avoid an infinite loop."
                            )
                            return

                    yield ToolCallEvent(id=tc.id, name=tc.name, args=tc.args)

                    # Permission gate — check before executing
                    if self._permission_mgr is not None:
                        from devagent.agent.permissions import _primary_arg
                        action = self._permission_mgr.check(tc.name, tc.args)
                        if action == "deny":
                            result = (
                                f"[blocked] Tool '{tc.name}' call denied by permission rule."
                            )
                            yield ToolResultEvent(
                                id=tc.id, name=tc.name, result=result, success=False
                            )
                            self.session_mgr.record_tool_result(
                                self.session_id,
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                content=result,
                            )
                            continue
                        if action == "ask":
                            primary = _primary_arg(tc.name, tc.args)
                            self._permission_mgr.request_approval(tc.id)
                            yield ApprovalNeededEvent(
                                call_id=tc.id,
                                tool_name=tc.name,
                                args=tc.args,
                                primary_arg=primary,
                            )
                            # Generator suspended here. CLI/UI resolves and resumes.
                            approved = self._permission_mgr.wait_for_decision(tc.id, timeout=300.0)
                            if not approved:
                                result = (
                                    f"[blocked] Tool '{tc.name}' call denied by user."
                                )
                                yield ToolResultEvent(
                                    id=tc.id, name=tc.name, result=result, success=False
                                )
                                self.session_mgr.record_tool_result(
                                    self.session_id,
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    content=result,
                                )
                                continue

                    raw = self.registry.call(tc.name, tc.args)

                    # Split diff from the clean result so the LLM history stays lean
                    result, _, diff = raw.partition("\n---diff---\n")

                    # Auto-test repair loop: after every file write, run relevant tests
                    if tc.name in _WRITE_TOOL_NAMES:
                        file_path = tc.args.get("path", tc.args.get("file_path", ""))
                        test_note = self._auto_test_after_write(file_path)
                        if test_note:
                            result = result + test_note

                    success = not result.startswith("[error]") and not result.startswith("[blocked]")

                    yield ToolResultEvent(id=tc.id, name=tc.name, result=result, success=success, diff=diff)

                    # Persist only the clean result (no diff noise for the LLM)
                    self.session_mgr.record_tool_result(
                        self.session_id,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=result,
                    )

                # Loop back to let the LLM continue
                continue

            else:
                # Final answer -- no more tool calls
                self.session_mgr.record_assistant(
                    self.session_id,
                    content=response.content,
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                )
                yield FinalAnswerEvent(
                    text=response.content,
                    tokens_in=response.input_tokens,
                    tokens_out=response.output_tokens,
                )
                return

        # Exceeded max iterations
        yield ErrorEvent(f"Agent loop exceeded {self._max_iterations} iterations without finishing")

    # ------------------------------------------------------------------
    # Test-driven repair helpers
    # ------------------------------------------------------------------

    def _auto_test_after_write(self, file_path: str) -> str:
        """Run relevant tests after a write. Returns a note to append to the tool result.

        Returns empty string if CodePrism is unavailable, no test file found,
        or the repair limit has been reached.
        """
        if not self._cp_client or not file_path:
            return ""
        current_attempts = self._repair_attempts.get(file_path, 0)
        if current_attempts >= self._max_repair:
            return ""

        try:
            summary = self._cp_client.get_module_summary(file_path)
        except Exception:
            return ""

        test_file = summary.get("test_coverage_file", "")
        if not test_file:
            return ""

        test_result = self.registry.call(
            "run_shell",
            {"command": f"python -m pytest {test_file} -x -q --tb=short 2>&1"},
        )

        low = test_result.lower()
        passed = "passed" in low and "failed" not in low and "error" not in low

        if passed:
            self._repair_attempts[file_path] = 0
            return f"\n\n[auto_test] {test_file}: all tests pass."

        current_attempts += 1
        self._repair_attempts[file_path] = current_attempts
        remaining = self._max_repair - current_attempts
        note = (
            f"\n\n[auto_test] Tests failed after your edit "
            f"(attempt {current_attempts}/{self._max_repair}):\n"
            f"Test file: {test_file}\n"
            f"Output:\n{test_result[:600]}"
        )
        if remaining > 0:
            note += (
                f"\n\nPlease fix the failing tests. "
                f"{remaining} auto-repair attempt(s) remaining."
            )
        else:
            note += (
                "\n\n[WARNING] Max repair attempts reached. "
                "Proceeding — manual review recommended."
            )
        return note
