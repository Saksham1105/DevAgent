"""High-level agent flows built on top of AgentLoop + DevAgentSession.

Each flow:
  1. Parses the input (issue URL, PR URL, repo string, CI run URL)
  2. Pre-fetches the relevant GitHub data (issue body, PR diff, run logs, …)
  3. Builds a flow-specific system prompt extension
  4. Creates a DevAgentSession and seeds it with a rich first message
  5. Drops into the interactive REPL so the user can follow up

Flows:
  run_implement  — fetch issue → CodePrism context → edit → test → PR
  run_review     — fetch PR diff → graph analysis → post inline review
  run_triage     — classify open issues by effort → post triage comments
  run_fix_ci     — read failed logs → locate code → propose fix
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# ---------------------------------------------------------------------------
# REPL command list — used for tab completion (Phase 29)
# ---------------------------------------------------------------------------

_REPL_COMMANDS = [
    "/explain", "/test", "/review", "/commit", "/summarize", "/deep-research", "/help",
    "/status", "/history", "/context", "/clear", "/rewind", "/permissions", "/goal", "/btw", "/autocompact",
    "/model", "/effort", "/think", "/memory", "/tokens", "/security", "/undo", "/diff", "/exit",
    "/fork", "/tasks", "/loop", "/rename", "/sessions",
    "/fast", "/compact", "/recap", "/branch", "/theme", "/batch", "/background",
    "/keybindings",
]

# Phase 42 — /keybindings reference table: (key combo, action description)
_KEYBINDINGS_TABLE: list[tuple[str, str]] = [
    ("Tab",          "Cycle through /command completions"),
    ("Up / Down",    "Navigate command history"),
    ("Ctrl+A",       "Move cursor to beginning of line"),
    ("Ctrl+E",       "Move cursor to end of line"),
    ("Ctrl+K",       "Delete from cursor to end of line"),
    ("Ctrl+U",       "Delete from cursor to beginning of line"),
    ("Ctrl+W",       "Delete word before cursor"),
    ("Alt+B",        "Move one word backward"),
    ("Alt+F",        "Move one word forward"),
    ("Ctrl+R",       "Reverse history search (incremental)"),
    ("Ctrl+L",       "Clear the terminal screen"),
    ("Ctrl+C",       "Cancel current input / interrupt agent"),
    ("Ctrl+D",       "Exit session (same as /exit)"),
    ("F1",           "Insert /help into the prompt"),
    ("F2",           "Insert /status into the prompt"),
]

# Phase 36 — prompt_toolkit colour themes
# Values are dicts passed to Style.from_dict(); None means use the default style.
_THEME_STYLES: dict[str, dict[str, str] | None] = {
    "default": None,
    "dracula": {
        "completion-menu.completion": "bg:#282a36 #f8f8f2",
        "completion-menu.completion.current": "bg:#44475a #f8f8f2",
        "prompt": "#50fa7b bold",
    },
    "monokai": {
        "completion-menu.completion": "bg:#272822 #f8f8f2",
        "completion-menu.completion.current": "bg:#49483e #f8f8f2",
        "prompt": "#a6e22e bold",
    },
    "solarized": {
        "completion-menu.completion": "bg:#002b36 #839496",
        "completion-menu.completion.current": "bg:#073642 #93a1a1",
        "prompt": "#268bd2 bold",
    },
}

# ---------------------------------------------------------------------------
# URL parsers (issue, PR, CI run)
# ---------------------------------------------------------------------------

@dataclass
class _ParsedIssue:
    owner: str
    repo: str
    number: int
    kind: str  # "issue" | "pull"


@dataclass
class _ParsedRun:
    owner: str
    repo: str
    run_id: str


def _parse_issue_or_pr_url(url: str) -> _ParsedIssue:
    """Parse https://github.com/owner/repo/issues/N or /pull/N."""
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)",
        url.strip(),
    )
    if not m:
        raise ValueError(
            f"Cannot parse GitHub issue/PR URL: {url!r}\n"
            "Expected: https://github.com/owner/repo/issues/N"
        )
    return _ParsedIssue(
        owner=m.group(1),
        repo=m.group(2),
        number=int(m.group(4)),
        kind=m.group(3),
    )


def _parse_run_url(url: str) -> _ParsedRun:
    """Parse https://github.com/owner/repo/actions/runs/RUN_ID."""
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/actions/runs/(\d+)",
        url.strip(),
    )
    if not m:
        raise ValueError(
            f"Cannot parse GitHub Actions run URL: {url!r}\n"
            "Expected: https://github.com/owner/repo/actions/runs/RUN_ID"
        )
    return _ParsedRun(owner=m.group(1), repo=m.group(2), run_id=m.group(3))


def _parse_repo(repo_or_url: str) -> tuple[str, str]:
    """Accept 'owner/repo' or 'https://github.com/owner/repo'."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?$", repo_or_url.strip())
    if m:
        return m.group(1), m.group(2)
    parts = repo_or_url.strip().split("/")
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse repo: {repo_or_url!r}. Use 'owner/repo'.")


# ---------------------------------------------------------------------------
# DevAgentSession — the shared agent stack
# ---------------------------------------------------------------------------

class DevAgentSession:
    """Full agent session: LLM + tools + memory + budget + security gate.

    Usage:
        session = DevAgentSession(cfg, project_root)
        session.interactive_repl(first_message="Implement issue #42 …")
    """

    def __init__(
        self,
        cfg,                          # DevAgentConfig
        project_root: str | Path,
        *,
        max_tokens: int | None = None,
        extra_system: str = "",
        resume_id: str | None = None,
        no_limit: bool = False,       # Phase 7.2: remove iteration cap
        plan_mode: bool = False,      # Phase 7.3: generate plan before first task
        allow: list[str] | None = None,   # Phase 10: auto-allow patterns
        deny: list[str] | None = None,    # Phase 10: auto-deny patterns
        interactive_approval: bool = False,  # Phase 10: pause for approval when no rule matches
        # Phase 15 — effort, bare mode, CI hardening
        effort: str | None = None,
        bare: bool = False,
        allow_tools: list[str] | None = None,
        output_format: str = "rich",
        system_prompt_override: str | None = None,
        # Phase 10 — permission mode shorthand
        permission_mode: str = "default",
        # Phase 20 — per-agent persistent memory
        agent_name: str | None = None,
        # Phase 29 — session name
        name: str | None = None,
        # Phase 41 — extra dirs + diff preview
        extra_dirs: list[str] | None = None,
        diff_preview: bool = False,
    ) -> None:
        from devagent.agent import permissions as perm_registry
        from devagent.agent.loop import AgentLoop
        from devagent.agent.permissions import PermissionManager, parse_rule
        from devagent.agent.system_prompt import build_system_prompt
        from devagent.core.llm import LLMClient
        from devagent.session.budget import TokenBudget
        from devagent.session.manager import SessionManager
        from devagent.session.memory import MemoryBlock
        from devagent.tools.registry import build_registry

        self._cfg = cfg
        self._project_root = Path(project_root).resolve()
        self._console = Console()
        self.security_log: list = []
        self._output_format = output_format

        # Apply effort override before anything reads cfg.llm.effort
        if effort is not None:
            cfg.llm.effort = effort

        # ── Session ──────────────────────────────────────────────────
        mgr = SessionManager()
        if resume_id:
            match = mgr.find_by_name(resume_id) or next(
                (s for s in mgr.list(limit=200) if s["id"].startswith(resume_id)), None
            )
            if not match:
                raise ValueError(f"Session not found: {resume_id}")
            session_id = match["id"]
        else:
            session_id = mgr.new(
                project=str(self._project_root),
                model=cfg.llm.model,
                provider=cfg.llm.provider,
                title=name or "",
            )
        self._mgr = mgr
        self.session_id = session_id

        # ── Permission manager (Phase 10 / Phase 15) ─────────────────
        rules = []
        for spec in (deny or []):
            rules.append(parse_rule(spec, "deny"))
        for spec in (allow or []):
            rules.append(parse_rule(spec, "allow"))
        # --allow-tools is a shorthand: auto-allow exact tool name matches
        for tool_name in (allow_tools or []):
            rules.append(parse_rule(tool_name, "allow"))
        # Phase 10 — permission_mode applies preset rule bundles
        _interactive = interactive_approval
        if not bare and permission_mode != "default":
            if permission_mode == "read-only":
                for t in ("write_file", "edit_file", "run_shell"):
                    rules.append(parse_rule(t, "deny"))
            elif permission_mode == "accept-edits":
                for t in ("write_file", "edit_file"):
                    rules.append(parse_rule(t, "allow"))
            elif permission_mode == "yolo":
                _interactive = False
                rules.insert(0, parse_rule("*", "allow"))
            elif permission_mode == "auto":
                _interactive = False

        self._permission_mgr: PermissionManager | None = None
        if not bare and (rules or _interactive):
            self._permission_mgr = PermissionManager(
                rules=rules, interactive=_interactive
            )
            perm_registry.register(session_id, self._permission_mgr)

        # ── CodePrism (optional; skipped in bare mode) ───────────────
        cp_client = None
        if not bare:
            try:
                from devagent.codeprism.client import CodePrismClient
                cp = CodePrismClient(str(self._project_root))
                if cp.is_indexed:
                    cp.attach_session(session_id)
                    cp_client = cp
            except Exception:
                pass
        self._cp_client = cp_client

        # ── Security gate confirm callback ────────────────────────────
        def _confirm(msg: str) -> bool:
            self._console.print(f"\n[yellow]{msg}[/yellow]")
            return Confirm.ask("Proceed with write?", default=False)

        # ── Diff preview callback (Phase 41) ──────────────────────────
        _diff_confirm_fn = None
        if diff_preview and not bare:
            from rich.syntax import Syntax
            def _diff_confirm_fn(path: str, diff: str) -> bool:
                self._console.print()
                self._console.print(Syntax(diff, "diff", theme="monokai", line_numbers=False))
                return Confirm.ask(f"Apply change to [cyan]{path}[/cyan]?", default=True)

        # ── Hooks runner (Phase 10) ────────────────────────────────────
        hook_runner = None
        if not bare:
            try:
                from devagent.hooks.runner import HookRunner
                hook_runner = HookRunner(self._project_root)
                if hook_runner.hooks:
                    hook_runner.session_start(session_id)
            except Exception:
                hook_runner = None
        self._hook_runner = hook_runner

        # ── Tool registry ─────────────────────────────────────────────
        gh_token = cfg.github.token or None
        registry = build_registry(
            project_root=str(self._project_root),
            codeprism_client=cp_client,
            security_log=self.security_log,
            confirm_fn=_confirm if cp_client else None,
            github_token=gh_token,
            session_id=session_id,
            provider=cfg.llm.provider,
            brave_api_key=cfg.brave.api_key,
            searchx_api_key=cfg.searchx.api_key,
            searchx_base_url=cfg.searchx.base_url,
            search_provider=cfg.search_provider,
            hook_runner=hook_runner,
            extra_dirs=extra_dirs,
            diff_confirm_fn=_diff_confirm_fn,
        )

        # ── Multi-model router (optional) ─────────────────────────────
        router = None
        try:
            from devagent.core.router import MultiModelRouter
            router = MultiModelRouter(cfg)
        except Exception:
            pass

        # ── Budget, memory, prompt, loop ─────────────────────────────
        budget = TokenBudget(
            max_tokens=max_tokens,
            warn_at_percent=cfg.budget.warn_at_percent,
        )
        self._budget = budget

        from devagent.agent.system_prompt import load_devagent_md
        from devagent.session.project_memory import ProjectMemory
        if agent_name:
            project_memory = ProjectMemory.for_agent(self._project_root, agent_name)
        else:
            project_memory = ProjectMemory(self._project_root)
        memory = MemoryBlock(session_id, project_memory=project_memory)
        self._memory = memory

        # Memory tools — agent can explicitly store/recall facts across turns
        from devagent.tools.memory_tools import (
            register_memory_tools,
            register_persistent_memory_tools,
        )
        register_memory_tools(registry, memory)
        # Phase 12 — persistent cross-session memory backed by .devagent/memory.md
        register_persistent_memory_tools(registry, project_memory)

        # ── Sub-agent spawning (Phase 14) ────────────────────────────
        from devagent.tools.agent_tools import register_agent_tools
        register_agent_tools(registry, cfg, str(self._project_root))

        # ── Undo stack (Phase 13) ─────────────────────────────────────
        # Each entry: {"path": str, "before": str | None}
        # before=None means the file did not exist before the write (new file).
        self._undo_stack: list[dict] = []
        self._last_diff: str = ""
        self._wrap_file_tools_for_undo(registry)

        # Bare mode: omit DEVAGENT.md from the system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            devagent_md = "" if bare else load_devagent_md(self._project_root)
            system_prompt = build_system_prompt(
                project_description=f"Project: {self._project_root.name}",
                extra_context=extra_system,
                devagent_md=devagent_md,
            )

        llm = LLMClient(cfg.llm)
        import sys
        max_iters = sys.maxsize if no_limit else cfg.agent.max_iterations
        self._loop = AgentLoop(
            llm=llm,
            registry=registry,
            session_mgr=mgr,
            session_id=session_id,
            memory=memory,
            budget=budget,
            system_prompt=system_prompt,
            codeprism_client=cp_client,
            router=router,
            max_iterations=max_iters,
            max_repair=cfg.agent.max_repair_iterations,
            loop_detection=cfg.agent.loop_detection,
            permission_mgr=self._permission_mgr,
            bare=bare,
        )
        self._router = router

        # Phase 15 — expose effort and bare for REPL commands
        self._bare = bare
        self._llm_client = llm  # kept so /effort and /think can mutate cfg.llm live

        # Status flags for the REPL
        self._cp_active = cp_client is not None
        self._gh_active = bool(gh_token)
        self._router_active = router is not None
        self._title_set = resume_id is not None
        self._plan_mode = plan_mode
        self._llm = LLMClient(cfg.llm)  # kept for plan generation

        # Phase 32 — /fast mode state
        self._fast_mode = False
        self._pre_fast_model: str = cfg.llm.model
        self._pre_fast_provider: str = cfg.llm.provider

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wrap_file_tools_for_undo(self, registry) -> None:
        """Wrap edit_file and write_file to capture before-state for /undo."""
        _UNDO_TOOLS = ("edit_file", "write_file")
        _MAX_UNDO = 20

        for tool_name in _UNDO_TOOLS:
            original = registry._handlers.get(tool_name)
            if original is None:
                continue

            def _make_wrapper(fn: object, tn: str):
                def wrapper(args: dict) -> str:
                    path = args.get("path", "")
                    target = self._project_root / path
                    before: str | None = None
                    try:
                        if target.exists():
                            before = target.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    result: str = fn(args)  # type: ignore[operator]
                    if not result.startswith("[error]") and not result.startswith("[tool_error]"):
                        self._undo_stack.append({"path": path, "before": before, "tool": tn})
                        if len(self._undo_stack) > _MAX_UNDO:
                            self._undo_stack.pop(0)
                    return result
                return wrapper

            registry._handlers[tool_name] = _make_wrapper(original, tool_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        """Auto-compress history if it has grown past the configured threshold."""
        if not self._cfg.session.auto_compress:
            return
        try:
            from devagent.session.compressor import maybe_compress
            result = maybe_compress(
                session_id=self.session_id,
                llm=self._llm,
                session_cfg=self._cfg.session,
                db_path=self._mgr.db_path,
            )
            if result:
                self._console.print(
                    f"[dim]Auto-compressed {result.events_compressed} events "
                    f"(~{result.tokens_saved} tokens saved).[/dim]"
                )
        except Exception:
            pass  # compression is best-effort; never block a turn

    def show_history(self, limit: int = 8) -> None:
        """Display a bounded resume/history view without changing LLM context."""
        from devagent.session import store

        summary = store.get_compressed_summary(self.session_id, db_path=self._mgr.db_path)
        events = self._mgr.get_events(self.session_id)
        if summary:
            self._console.print("[bold]Earlier session summary[/bold]")
            self._console.print(summary)
        recent = events[-limit:]
        if recent:
            self._console.print(f"[bold]Recent messages (last {len(recent)})[/bold]")
            for event in recent:
                role = event["role"].replace("tool_result", "tool")
                content = (event.get("content") or "").strip().replace("\n", " ")
                if len(content) > 240:
                    content = content[:237] + "..."
                self._console.print(f"[dim]{role}:[/dim] {content}")
        elif not summary:
            self._console.print("[dim]No previous messages in this session.[/dim]")

    def run_message(self, message: str, quiet: bool = False) -> str:
        """Drive one user turn; render events. Returns the final LLM text.

        quiet=True suppresses all terminal output (used by background watcher).
        On the first call when plan_mode=True, generates and presents a plan for
        approval before executing.
        """
        from devagent.agent.loop import FinalAnswerEvent
        from devagent.output.streaming import render_events

        # Auto-compress long sessions before each turn
        if not quiet:
            self._maybe_compress()

        # Plan mode: intercept the first user message
        if self._plan_mode:
            self._plan_mode = False  # only applies once per session
            from devagent.agent.planner import generate_plan
            from devagent.output.plan_renderer import plan_to_first_message, render_plan

            with self._console.status("[cyan]Generating plan...[/cyan]"):
                plan = generate_plan(self._llm, message, self._project_root.name)

            result = render_plan(plan, self._console)
            if result == "cancel":
                self._console.print("[dim]Task cancelled.[/dim]")
                return ""
            message = plan_to_first_message(plan, message)

        if quiet:
            final = ""
            for event in self._loop.run(message):
                if isinstance(event, FinalAnswerEvent):
                    final = event.text
            return final

        # Tee the event stream to capture the most recent diff for /diff command
        from devagent.agent.loop import ToolResultEvent as _TRE

        def _tee(gen):
            for event in gen:
                if isinstance(event, _TRE) and event.diff:
                    self._last_diff = event.diff
                yield event

        return render_events(_tee(self._loop.run(message)), permission_mgr=self._permission_mgr)

    def interactive_repl(self, *, first_message: str | None = None) -> None:
        """Run the interactive REPL, optionally seeding with first_message."""
        if first_message:
            self.run_message(first_message)
            self._title_set = True  # seeded message serves as title

        # Phase 29 — tab completion via prompt_toolkit (falls back to plain input)
        # Phase 36 — theme support: _pt_holder[0] is replaced by /theme live
        _pt_holder: list = [None]
        try:
            from prompt_toolkit import PromptSession as _PSession
            from prompt_toolkit.completion import WordCompleter as _WC
            from prompt_toolkit.document import Document as _Doc
            from prompt_toolkit.key_binding import KeyBindings as _KB
            from prompt_toolkit.styles import Style as _PTStyle

            def _make_pt_session(theme_name: str):
                _style_dict = _THEME_STYLES.get(theme_name)
                _style = _PTStyle.from_dict(_style_dict) if _style_dict else None
                _completer = _WC(_REPL_COMMANDS, sentence=True)
                _kb = _KB()

                @_kb.add("f1")
                def _f1(event):
                    event.current_buffer.set_document(_Doc("/help"))
                    event.current_buffer.validate_and_handle()

                @_kb.add("f2")
                def _f2(event):
                    event.current_buffer.set_document(_Doc("/status"))
                    event.current_buffer.validate_and_handle()

                return _PSession(
                    completer=_completer,
                    complete_while_typing=True,
                    style=_style,
                    key_bindings=_kb,
                )

            _pt_holder[0] = _make_pt_session(getattr(getattr(self._cfg, "ui", None), "theme", "default"))

            def _read_input() -> str:
                return _pt_holder[0].prompt("> ").strip()
        except Exception:
            def _read_input() -> str:
                return self._console.input("[bold cyan]>[/bold cyan] ").strip()

        while True:
            try:
                raw = _read_input()
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[dim]Session ended.[/dim]")
                self._print_exit()
                break

            if not raw:
                continue

            # Phase 25 — @agent-name: spawn a named agent as a background sub-task
            if raw.startswith("@"):
                import threading
                import uuid

                from devagent.agents.loader import load_agent_defs
                from devagent.session.task_store import get_store as _get_store

                mention_parts = raw[1:].split(None, 1)
                agent_mention_name = mention_parts[0].strip() if mention_parts else ""
                agent_task = mention_parts[1].strip() if len(mention_parts) > 1 else ""
                _agent_defs = load_agent_defs(self._project_root)

                if not agent_mention_name:
                    self._console.print("[yellow]Usage: @agent-name <task description>[/yellow]")
                elif agent_mention_name not in _agent_defs:
                    available = ", ".join(sorted(_agent_defs)) or "(none)"
                    self._console.print(
                        f"[yellow]Unknown agent: {agent_mention_name!r}. "
                        f"Available: {available}[/yellow]"
                    )
                else:
                    _ag_def = _agent_defs[agent_mention_name]
                    _mention_task = agent_task or _ag_def.prompt or f"Run the {agent_mention_name} agent."
                    _mention_task_id = uuid.uuid4().hex[:8]
                    _get_store().add(_mention_task_id, f"@{agent_mention_name}: {_mention_task[:50]}")
                    _cfg = self._cfg
                    _root = self._project_root
                    _ag = _ag_def

                    def _bg_mention(tid=_mention_task_id, task=_mention_task, cfg=_cfg,
                                    root=_root, ag=_ag):
                        try:
                            from devagent.agent.flows import DevAgentSession
                            from devagent.agent.worktree import isolated_worktree
                            _an = ag.name if ag.memory == "project" else None
                            if ag.isolation == "worktree":
                                with isolated_worktree(root, branch_prefix=ag.name) as (wt_path, _br):
                                    bg = DevAgentSession(cfg, str(wt_path), bare=True,
                                                         permission_mode=ag.permission, agent_name=_an)
                                    result = bg.run_message(task, quiet=True)
                            else:
                                bg = DevAgentSession(cfg, root, bare=True,
                                                     permission_mode=ag.permission, agent_name=_an)
                                result = bg.run_message(task, quiet=True)
                            _get_store().complete(tid, result or "done")
                        except Exception as exc:
                            _get_store().complete(tid, str(exc), failed=True)

                    threading.Thread(target=_bg_mention, daemon=True).start()
                    self._console.print(
                        f"[dim]Spawned @{agent_mention_name} [{_mention_task_id}]: {_mention_task[:60]}[/dim]"
                    )
                    self._console.print("[dim]Check progress with /tasks[/dim]")
                continue

            # Phase 13 — shell escape: !<command> runs in the project root
            if raw.startswith("!"):
                import subprocess
                shell_cmd = raw[1:].strip()
                if shell_cmd:
                    proc = subprocess.run(
                        shell_cmd, shell=True, capture_output=True, text=True,
                        cwd=str(self._project_root),
                    )
                    if proc.stdout:
                        self._console.print(proc.stdout, end="")
                    if proc.stderr:
                        self._console.print(proc.stderr, style="dim red", end="")
                continue

            cmd = raw.lower()

            if cmd in ("/exit", "/quit", "exit", "quit"):
                self._console.print("[dim]Session ended.[/dim]")
                self._print_exit()
                break

            # Phase 13 — /undo: restore the last file the agent wrote or edited
            if cmd == "/undo":
                if not self._undo_stack:
                    self._console.print("[yellow]Nothing to undo.[/yellow]")
                else:
                    entry = self._undo_stack.pop()
                    path, before = entry["path"], entry["before"]
                    target = self._project_root / path
                    if before is None:
                        target.unlink(missing_ok=True)
                        self._console.print(f"[green]Undone:[/green] deleted {path} (was a new file)")
                    else:
                        target.write_text(before, encoding="utf-8")
                        self._console.print(f"[green]Undone:[/green] restored {path}")
                continue

            # Phase 13 — /diff: replay the last inline diff
            if cmd == "/diff":
                if not self._last_diff:
                    self._console.print("[dim]No diff recorded yet.[/dim]")
                else:
                    from devagent.output.streaming import render_diff
                    render_diff(self._last_diff)
                continue

            if cmd.startswith("/history"):
                parts = raw.split(None, 1)
                try:
                    limit = max(1, min(int(parts[1]), 50)) if len(parts) > 1 else 8
                except ValueError:
                    limit = 8
                self.show_history(limit)
                continue

            # Phase 23 / Phase 28 — /model: hot-swap provider/model; /model alone shows picker
            if cmd.startswith("/model"):
                parts = raw.split(None, 1)
                _valid_providers = ("ollama", "anthropic", "openai", "gemini", "groq")

                # Helper: fetch available models for the current Ollama endpoint.
                def _get_ollama_models() -> list[str]:
                    try:
                        from devagent.core.llm import LLMClient as _LC
                        return _LC(self._cfg.llm).list_ollama_models()
                    except Exception:
                        return []

                if len(parts) < 2:
                    # No arg — show current model + numbered list for Ollama endpoints.
                    self._console.print(
                        f"[dim]Current: [bold]{self._cfg.llm.provider}/{self._cfg.llm.model}[/bold][/dim]"
                    )
                    if self._cfg.llm.provider == "ollama":
                        with self._console.status("[dim]Fetching available models...[/dim]"):
                            _models = _get_ollama_models()
                        if _models:
                            self._console.print()
                            for _i, _m in enumerate(_models, 1):
                                marker = " [green]←[/green]" if _m == self._cfg.llm.model else ""
                                self._console.print(f"  [cyan]{_i}[/cyan]. {_m}{marker}")
                            self._console.print()
                            self._console.print(
                                "[dim]Switch with: [bold]/model <number>[/bold]  or  "
                                "[bold]/model <name>[/bold][/dim]"
                            )
                        else:
                            self._console.print(
                                f"[dim]Usage: /model <provider/model>  or  /model <model>\n"
                                f"Providers: {', '.join(_valid_providers)}[/dim]"
                            )
                    else:
                        self._console.print(
                            f"[dim]Usage: /model <provider/model>  or  /model <model>\n"
                            f"Providers: {', '.join(_valid_providers)}[/dim]"
                        )
                else:
                    spec = parts[1].strip()

                    # /model <N> — pick by index from the Ollama model list.
                    if spec.isdigit() and self._cfg.llm.provider == "ollama":
                        _models = _get_ollama_models()
                        idx = int(spec) - 1
                        if _models and 0 <= idx < len(_models):
                            self._cfg.llm.model = _models[idx]
                            self._console.print(
                                f"[dim]Model switched to [bold]{self._cfg.llm.provider}/"
                                f"{self._cfg.llm.model}[/bold][/dim]"
                            )
                        else:
                            self._console.print(
                                f"[yellow]No model at index {spec}. "
                                f"Run [bold]/model[/bold] to see the list.[/yellow]"
                            )
                    elif "/" in spec:
                        new_provider, new_model = spec.split("/", 1)
                        new_provider = new_provider.strip().lower()
                        new_model = new_model.strip()
                        if new_provider not in _valid_providers:
                            self._console.print(
                                f"[yellow]Unknown provider '{new_provider}'. "
                                f"Valid: {', '.join(_valid_providers)}[/yellow]"
                            )
                        else:
                            self._cfg.llm.provider = new_provider
                            self._cfg.llm.model = new_model
                            self._console.print(
                                f"[dim]Model switched to [bold]{new_provider}/{new_model}[/bold][/dim]"
                            )
                    else:
                        self._cfg.llm.model = spec
                        self._console.print(
                            f"[dim]Model switched to [bold]{self._cfg.llm.provider}/{spec}[/bold][/dim]"
                        )
                if self._router is not None and len(parts) >= 2:
                    self._router.set_override(self._cfg.llm)
                continue

            # Phase 15 — /effort: change effort level for the rest of the session
            if cmd.startswith("/effort"):
                parts = raw.split(None, 1)
                valid = ("low", "medium", "high", "xhigh", "max")
                if len(parts) < 2 or parts[1].strip().lower() not in valid:
                    current = self._cfg.llm.effort
                    self._console.print(
                        f"[dim]Current effort: [bold]{current}[/bold]\n"
                        f"Usage: /effort low|medium|high|xhigh|max[/dim]"
                    )
                else:
                    new_effort = parts[1].strip().lower()
                    self._cfg.llm.effort = new_effort
                    self._console.print(f"[dim]Effort set to [bold]{new_effort}[/bold][/dim]")
                continue

            # Phase 15 — /think: toggle extended thinking (Anthropic only)
            if cmd.startswith("/think"):
                parts = raw.split(None, 1)
                if len(parts) < 2 or parts[1].strip().lower() not in ("on", "off"):
                    state = "on" if self._cfg.llm.extended_thinking else "off"
                    self._console.print(
                        f"[dim]Extended thinking: [bold]{state}[/bold]\n"
                        f"Usage: /think on|off  (Anthropic only; auto-activates for xhigh/max effort)[/dim]"
                    )
                else:
                    enable = parts[1].strip().lower() == "on"
                    self._cfg.llm.extended_thinking = enable
                    state = "enabled" if enable else "disabled"
                    self._console.print(f"[dim]Extended thinking {state}[/dim]")
                continue

            if cmd == "/memory":
                mem = self._memory.all()
                if mem:
                    for k, v in mem.items():
                        self._console.print(f"  [bold]{k}[/bold]: {v}")
                else:
                    self._console.print("[dim]No memory items.[/dim]")
                continue

            if cmd == "/tokens":
                self._console.print(f"  [dim]{self._budget.status_line()}[/dim]")
                for row in self._budget.per_model_summary():
                    self._console.print(
                        f"    {row['provider']}/{row['model']}: "
                        f"{row['input_tokens']:,}in / {row['output_tokens']:,}out  "
                        f"${row['cost_usd']:.4f}  ({row['calls']} calls)"
                    )
                continue

            if cmd == "/security":
                from devagent.tools.security_gate import format_security_report
                self._console.print(
                    Panel(
                        format_security_report(self.security_log),
                        title="Security Gate",
                        border_style="yellow",
                    )
                )
                continue

            # Phase 10 — /permissions: show active permission rules and hooks
            if cmd == "/permissions":
                lines: list[str] = []
                if self._permission_mgr is None:
                    lines.append("[dim]No permission rules active.[/dim]")
                    lines.append("[dim]Use --allow/--deny or --permission-mode to configure.[/dim]")
                else:
                    lines.append("[bold]Permission rules (first match wins):[/bold]")
                    for r in self._permission_mgr.rules:
                        lines.append(f"  [{r.action}] {r.tool}:{r.pattern}")
                if self._hook_runner and self._hook_runner.hooks:
                    lines.append("\n[bold]Active hooks:[/bold]")
                    for h in self._hook_runner.hooks:
                        detail = h.command or h.url
                        tool_str = f" (tool={h.tool})" if h.tool else " (all tools)"
                        lines.append(f"  [{h.event}]{tool_str} [{h.type}] {detail}")
                elif self._hook_runner:
                    lines.append("\n[dim]No hooks configured in .devagent/hooks.toml[/dim]")
                self._console.print(Panel("\n".join(lines), title="/permissions", border_style="cyan"))
                continue

            # Phase 13 — /status: session summary
            if cmd == "/status":
                _sess_row = self._mgr.get(self.session_id)
                _sess_name = (_sess_row or {}).get("title", "") if _sess_row else ""
                lines = [
                    f"Session:  {self.session_id[:8]}" + (f"  ({_sess_name})" if _sess_name else ""),
                    f"Project:  {self._project_root}",
                    f"Model:    {self._cfg.llm.provider}/{self._cfg.llm.model}",
                    f"Effort:   {self._cfg.llm.effort}",
                    f"{self._budget.status_line()}",
                    f"Undo entries: {len(self._undo_stack)}",
                    f"Security events: {len(self.security_log)}",
                ]
                self._console.print(Panel("\n".join(lines), title="/status", border_style="cyan"))
                continue

            # Phase 29 — /rename: give the current session a human-readable name
            if cmd.startswith("/rename"):
                _rename_parts = raw.split(None, 1)
                if len(_rename_parts) < 2 or not _rename_parts[1].strip():
                    self._console.print("[dim]Usage: /rename <name>[/dim]")
                else:
                    _new_name = _rename_parts[1].strip()
                    self._mgr.rename(self.session_id, _new_name)
                    self._console.print(f"[dim]Session renamed to [bold]{_new_name}[/bold][/dim]")
                continue

            # Phase 30 — /sessions: list sessions for this project
            if cmd == "/sessions":
                _proj_sessions = self._mgr.list_by_project(str(self._project_root), limit=20)
                if not _proj_sessions:
                    self._console.print("[dim]No sessions for this project yet.[/dim]")
                else:
                    lines = []
                    import time as _time
                    for _s in _proj_sessions:
                        _sid = _s["id"][:8]
                        _title = _s.get("title") or ""
                        _ts = _s.get("updated_at") or 0
                        _updated = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(float(_ts)))
                        _active = " [green]← current[/green]" if _s["id"] == self.session_id else ""
                        label = f"[bold]{_title}[/bold]  [dim]{_sid}[/dim]" if _title else f"[dim]{_sid}[/dim]"
                        lines.append(f"  {label}  [dim]{_updated}[/dim]{_active}")
                    self._console.print(Panel(
                        "\n".join(lines),
                        title=f"Sessions — {self._project_root.name}",
                        border_style="cyan",
                    ))
                    self._console.print("[dim]Resume with: devagent run --resume <name or id>[/dim]")
                continue

            # Phase 32 — /fast [off]: toggle cheap-tier model for the session
            if cmd.startswith("/fast"):
                _fast_arg = raw.split(None, 1)[1].strip().lower() if len(raw.split(None, 1)) > 1 else ""
                if _fast_arg == "off":
                    if self._fast_mode:
                        self._cfg.llm.provider = self._pre_fast_provider
                        self._cfg.llm.model = self._pre_fast_model
                        self._fast_mode = False
                        self._console.print(
                            f"[dim]Fast mode off — restored "
                            f"[bold]{self._cfg.llm.provider}/{self._cfg.llm.model}[/bold][/dim]"
                        )
                    else:
                        self._console.print("[dim]Fast mode is already off.[/dim]")
                else:
                    if not self._fast_mode:
                        self._pre_fast_provider = self._cfg.llm.provider
                        self._pre_fast_model = self._cfg.llm.model
                    cheap = self._cfg.router.cheap
                    self._cfg.llm.provider = cheap.get("provider", self._cfg.llm.provider)
                    self._cfg.llm.model = cheap.get("model", self._cfg.llm.model)
                    self._fast_mode = True
                    self._console.print(
                        f"[dim]Fast mode on — using "
                        f"[bold]{self._cfg.llm.provider}/{self._cfg.llm.model}[/bold]  "
                        f"(type [bold]/fast off[/bold] to restore)[/dim]"
                    )
                if self._router is not None:
                    self._router.set_override(self._cfg.llm)
                continue

            # Phase 32 — /compact [focus on X]: force-compress session history now
            if cmd.startswith("/compact"):
                _focus = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else None
                from devagent.core.llm import LLMClient as _LC
                from devagent.session.compressor import compress_session as _cs
                _comp_llm = _LC(self._cfg.llm)
                keep_n = self._cfg.session.compression_window_size
                with self._console.status("[cyan]Compressing session history...[/cyan]"):
                    _result = _cs(
                        self.session_id,
                        _comp_llm,
                        keep_last_n=keep_n,
                        focus=_focus,
                    )
                if _result is None:
                    self._console.print("[dim]Nothing to compress yet (history is short).[/dim]")
                else:
                    _focus_note = f"  Focus: {_focus}" if _focus else ""
                    self._console.print(
                        f"[dim]Compressed {_result.events_compressed} events "
                        f"(~{_result.tokens_saved} tokens saved).{_focus_note}[/dim]"
                    )
                continue

            # Phase 34 — /recap [N]: print last N turns so user can re-orient after a long break
            if cmd.startswith("/recap"):
                _recap_parts = raw.split(None, 1)
                _recap_n = 5
                if len(_recap_parts) > 1:
                    try:
                        _recap_n = max(1, int(_recap_parts[1].strip()))
                    except ValueError:
                        pass
                from rich.rule import Rule as _Rule

                from devagent.session import store as _store
                _revents = _store.get_events(self.session_id)
                _visible = [
                    e for e in _revents
                    if e["role"] in ("user", "assistant") and (e.get("content") or "").strip()
                ]
                _to_show = _visible[-_recap_n:]
                if not _to_show:
                    self._console.print("[dim]No history yet.[/dim]")
                else:
                    self._console.print(_Rule(f"[dim]Last {len(_to_show)} messages[/dim]", style="dim"))
                    for _rev in _to_show:
                        _label = (
                            "[bold cyan]You[/bold cyan]"
                            if _rev["role"] == "user"
                            else "[bold green]Agent[/bold green]"
                        )
                        _txt = (_rev.get("content") or "").strip()
                        if len(_txt) > 500:
                            _txt = _txt[:500] + "…"
                        self._console.print(f"{_label}: {_txt}")
                    self._console.print(_Rule(style="dim"))
                continue

            # Phase 36 — /branch: snapshot current session to a new branch session
            if cmd == "/branch":
                from devagent.session import store as _store
                _branch_id = self._mgr.new(
                    project=str(self._project_root),
                    model=self._cfg.llm.model,
                    provider=self._cfg.llm.provider,
                    title=f"Branch of {self.session_id[:8]}",
                )
                _copied = _store.copy_session_events(self.session_id, _branch_id)
                self._console.print(
                    f"[dim]Branch created: [bold]{_branch_id[:8]}[/bold]  "
                    f"({_copied} events copied)[/dim]"
                )
                self._console.print(
                    f"[dim]Resume it with: devagent run --resume {_branch_id[:8]}[/dim]"
                )
                continue

            # Phase 36 — /theme [<name>]: list or switch colour theme
            if cmd.startswith("/theme"):
                _theme_arg = raw.split(None, 1)[1].strip().lower() if len(raw.split(None, 1)) > 1 else ""
                _valid = list(_THEME_STYLES.keys())
                if not _theme_arg or _theme_arg == "list":
                    self._console.print(
                        f"[dim]Themes: {', '.join(_valid)}  |  current: "
                        f"{getattr(getattr(self._cfg, 'ui', None), 'theme', 'default')}[/dim]"
                    )
                elif _theme_arg not in _THEME_STYLES:
                    self._console.print(
                        f"[yellow]Unknown theme '{_theme_arg}'. "
                        f"Available: {', '.join(_valid)}[/yellow]"
                    )
                else:
                    self._cfg.ui.theme = _theme_arg
                    if _pt_holder[0] is not None:
                        try:
                            _pt_holder[0] = _make_pt_session(_theme_arg)
                        except Exception:
                            pass
                    self._console.print(f"[dim]Theme: [bold]{_theme_arg}[/bold][/dim]")
                continue

            # Phase 42 — /keybindings: display keyboard shortcut reference table
            if cmd == "/keybindings":
                from rich.table import Table as _Table
                _kb_table = _Table(title="Keyboard Shortcuts", border_style="dim", show_header=True, header_style="bold cyan")
                _kb_table.add_column("Key", style="bold yellow", no_wrap=True)
                _kb_table.add_column("Action")
                for _key, _action in _KEYBINDINGS_TABLE:
                    _kb_table.add_row(_key, _action)
                self._console.print(_kb_table)
                self._console.print("[dim]F1 and F2 insert /help and /status directly into the prompt.[/dim]")
                continue

            # Phase 13 — /clear: reset conversation (start new session, keep project memory)
            if cmd == "/clear":
                new_sid = self._mgr.new(
                    project=str(self._project_root),
                    model=self._cfg.llm.model,
                    provider=self._cfg.llm.provider,
                )
                self.session_id = new_sid
                self._loop.session_id = new_sid
                self._undo_stack.clear()
                self._last_diff = ""
                self._title_set = False
                self._console.print("[dim]Conversation cleared. Persistent memory preserved.[/dim]")
                continue

            # Phase 13 — /rewind N: remove last N turns
            if cmd.startswith("/rewind"):
                parts = raw.split(None, 1)
                n = 1
                if len(parts) > 1:
                    try:
                        n = max(1, int(parts[1].strip()))
                    except ValueError:
                        self._console.print("[yellow]Usage: /rewind N  (N = number of turns to remove)[/yellow]")
                        continue
                from devagent.session import store as _store
                removed = _store.trim_events(self.session_id, n)
                self._console.print(f"[dim]Removed {removed} events ({n} turn(s) rewound).[/dim]")
                continue

            # Phase 13 — /context: token usage breakdown
            if cmd == "/context":
                budget = self._budget
                total_tokens = budget.total_used
                lines = [
                    f"Total tokens used this session:  {total_tokens:,}",
                ]
                cap = budget.max_tokens
                if cap:
                    pct = int(100 * total_tokens / cap)
                    lines.append(f"Budget cap:  {cap:,} ({pct}% used)")
                else:
                    lines.append("Budget cap:  unlimited")
                lines.append(f"Undo stack:  {len(self._undo_stack)} entries")
                lines.append(f"\n{budget.status_line()}")
                self._console.print(Panel("\n".join(lines), title="/context", border_style="blue"))
                continue

            # Phase 13 — /autocompact [tokens|off]
            if cmd.startswith("/autocompact"):
                parts = raw.split(None, 1)
                if len(parts) < 2 or parts[1].strip().lower() == "off":
                    self._cfg.session.auto_compress = False
                    self._console.print("[dim]Auto-compress disabled.[/dim]")
                else:
                    try:
                        self._cfg.session.auto_compress = True
                        self._console.print("[dim]Auto-compress enabled.[/dim]")
                    except ValueError:
                        self._console.print("[yellow]Usage: /autocompact  or  /autocompact off[/yellow]")
                continue

            # Phase 13 — /goal <condition>: loop until the goal is met
            if cmd.startswith("/goal"):
                goal = raw[5:].strip()
                if not goal:
                    self._console.print("[yellow]Usage: /goal <condition to meet>[/yellow]")
                    continue
                self._console.print(f"[dim]Goal mode: looping until — {goal}[/dim]")
                max_iters = 8
                for gi in range(max_iters):
                    self._console.print(f"[dim]Goal iteration {gi + 1}/{max_iters}…[/dim]")
                    self.run_message(f"Work toward this goal: {goal}")
                    check = self.run_message(
                        f"GOAL CHECK — answer YES or NO only: Is this goal fully met? Goal: {goal}",
                        quiet=True,
                    )
                    if check.strip().upper().startswith("YES"):
                        self._console.print(f"[green]Goal met after {gi + 1} iteration(s).[/green]")
                        break
                else:
                    self._console.print(f"[yellow]Goal not met after {max_iters} iterations.[/yellow]")
                continue

            # Phase 13 — /btw <question>: side question that doesn't enter the history
            if cmd.startswith("/btw"):
                question = raw[4:].strip()
                if not question:
                    self._console.print("[yellow]Usage: /btw <question>  (answer not saved to history)[/yellow]")
                    continue
                from devagent.session import store as _btw_store
                events_before = _btw_store.get_events(self.session_id)
                seq_before = max((e["seq"] for e in events_before), default=-1)
                self.run_message(question)
                # Remove all events added during the /btw turn
                from devagent.session.store import _conn
                try:
                    with _conn() as conn:
                        conn.execute(
                            "DELETE FROM events WHERE session_id = ? AND seq > ?",
                            (self.session_id, seq_before),
                        )
                except Exception:
                    pass
                self._console.print("[dim](response above not added to session history)[/dim]")
                continue

            # Phase 38 — /background [<task>]: launch task as detached subprocess or list jobs
            if cmd.startswith("/background"):
                _bg_task = raw[len("/background"):].strip()
                from devagent.background import jobs as _bg_jobs
                if not _bg_task:
                    # List all background jobs for this project
                    _bg_all = _bg_jobs.list_jobs(self._project_root)
                    if not _bg_all:
                        self._console.print("[dim]No background jobs.[/dim]")
                    else:
                        for _bj in _bg_all:
                            _bc = {"running": "yellow", "done": "green", "failed": "red"}.get(
                                _bj.status, "white"
                            )
                            self._console.print(
                                f"  [{_bc}]{_bj.status}[/{_bc}]  {_bj.id}  "
                                f"pid={_bj.pid}  {_bj.task[:60]}"
                            )
                            self._console.print(f"    [dim]log: {_bj.log_path}[/dim]")
                else:
                    _bj = _bg_jobs.launch(_bg_task, self._project_root)
                    self._console.print(
                        f"[dim]Background job [bold]{_bj.id}[/bold] started  pid={_bj.pid}[/dim]"
                    )
                    self._console.print(f"[dim]Tail output: devagent run — or: tail -f {_bj.log_path}[/dim]")
                    self._console.print("[dim]List jobs: /background (no args)[/dim]")
                continue

            # Phase 14 — /tasks: list background tasks
            if cmd == "/tasks":
                from devagent.session.task_store import get_store as _get_store
                tasks = _get_store().list_tasks()
                if not tasks:
                    self._console.print("[dim]No background tasks.[/dim]")
                else:
                    for t in tasks:
                        color = {"running": "yellow", "done": "green", "failed": "red"}.get(t.status, "white")
                        self._console.print(
                            f"  [{color}]{t.status}[/{color}]  {t.task_id}  {t.label}  ({t.elapsed:.0f}s)"
                        )
                        if t.result and t.status != "running":
                            self._console.print(f"    [dim]{t.result[:120]}[/dim]")
                continue

            # Phase 37 — /batch <task> -- <glob>: fan-out one sub-agent per matched file
            if cmd.startswith("/batch"):
                _batch_rest = raw[len("/batch"):].strip()
                if " -- " not in _batch_rest:
                    self._console.print(
                        "[yellow]Usage: /batch <task> -- <glob>  "
                        "e.g. /batch 'add type hints' -- src/**/*.py[/yellow]"
                    )
                    continue
                _batch_task, _batch_glob = _batch_rest.rsplit(" -- ", 1)
                _batch_task = _batch_task.strip()
                _batch_glob = _batch_glob.strip()

                import glob as _glob_mod
                _matched = sorted(
                    _glob_mod.glob(_batch_glob, root_dir=str(self._project_root), recursive=True)
                )
                if not _matched:
                    self._console.print(f"[yellow]No files matched: {_batch_glob}[/yellow]")
                    continue

                self._console.print(
                    f"[dim]Batch: {len(_matched)} file(s) matched — running up to 4 in parallel…[/dim]"
                )

                import concurrent.futures
                import threading
                _batch_results: dict[str, str] = {}
                _batch_lock = threading.Lock()
                _batch_cfg = self._cfg
                _batch_root = self._project_root

                def _batch_worker(
                    fpath: str,
                    _cfg=_batch_cfg,
                    _root=_batch_root,
                    _task=_batch_task,
                ) -> tuple[str, str]:
                    try:
                        from devagent.agent.flows import DevAgentSession as _DS
                        _sub = _DS(_cfg, _root, bare=True)
                        msg = f"Apply the following change to `{fpath}` only:\n\n{_task}"
                        result = _sub.run_message(msg, quiet=True)
                        return fpath, result or "done"
                    except Exception as _exc:
                        return fpath, f"ERROR: {_exc}"

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as _pool:
                    for _fp, _res in _pool.map(_batch_worker, _matched):
                        with _batch_lock:
                            _batch_results[_fp] = _res

                from rich.table import Table as _Table
                _tbl = _Table(title=f"Batch: {_batch_task[:60]}", show_lines=False)
                _tbl.add_column("File", style="cyan", no_wrap=True)
                _tbl.add_column("Result")
                for _fp in _matched:
                    _r = _batch_results.get(_fp, "—")
                    _style = "red" if str(_r).startswith("ERROR") else "green"
                    _tbl.add_row(_fp, f"[{_style}]{str(_r)[:100]}[/{_style}]")
                self._console.print(_tbl)
                continue

            # Phase 14 — /fork <task>: run a task in a background thread
            if cmd.startswith("/fork"):
                fork_task = raw[5:].strip()
                if not fork_task:
                    self._console.print("[yellow]Usage: /fork <task description>[/yellow]")
                    continue
                import threading
                import uuid

                from devagent.session.task_store import get_store as _get_store
                task_id = uuid.uuid4().hex[:8]
                _get_store().add(task_id, fork_task[:60])
                _cfg = self._cfg
                _root = self._project_root
                def _bg_run(tid=task_id, task=fork_task, cfg=_cfg, root=_root):
                    try:
                        from devagent.agent.flows import DevAgentSession
                        bg = DevAgentSession(cfg, root, bare=True)
                        result = bg.run_message(task, quiet=True)
                        _get_store().complete(tid, result or "done")
                    except Exception as exc:
                        _get_store().complete(tid, str(exc), failed=True)
                threading.Thread(target=_bg_run, daemon=True).start()
                self._console.print(f"[dim]Background task {task_id}: {fork_task[:60]}[/dim]")
                self._console.print("[dim]Check progress with /tasks[/dim]")
                continue

            # Phase 14 — /loop [interval] <command>: run a command on a schedule
            if cmd.startswith("/loop"):
                rest = raw[5:].strip()
                if rest.lower() in ("", "off", "stop"):
                    stop_ev = getattr(self, "_loop_stop_event", None)
                    if stop_ev is not None:
                        stop_ev.set()
                        del self._loop_stop_event
                        self._console.print("[dim]Loop stopped.[/dim]")
                    else:
                        self._console.print("[dim]No loop running.[/dim]")
                    continue
                import re as _re
                import threading
                m = _re.match(r"^(\d+)(m|s)\s+(.+)$", rest)
                if m:
                    interval = int(m.group(1)) * (60 if m.group(2) == "m" else 1)
                    loop_cmd = m.group(3)
                else:
                    interval = 300
                    loop_cmd = rest
                stop_event = threading.Event()
                self._loop_stop_event = stop_event
                def _loop_runner(se=stop_event, cmd_str=loop_cmd, iv=interval, sess=self):
                    while not se.is_set():
                        se.wait(iv)
                        if se.is_set():
                            break
                        try:
                            sess.run_message(cmd_str, quiet=True)
                        except Exception:
                            pass
                threading.Thread(target=_loop_runner, daemon=True).start()
                self._console.print(
                    f"[dim]Loop started: every {interval}s, command: {loop_cmd}  (type /loop off to stop)[/dim]"
                )
                continue

            # Phase 7.4 — Skills / slash commands
            if raw.startswith("/"):
                parts = raw[1:].split(None, 1)
                skill_name = parts[0].lower()
                extra_args = parts[1] if len(parts) > 1 else ""

                from devagent.skills.loader import load_all_skills
                skills = load_all_skills()

                if skill_name == "help":
                    self._console.print()
                    self._console.print("[bold]Available skills:[/bold]")
                    for sn, sk in skills.items():
                        self._console.print(f"  [cyan]/{sn}[/cyan]  {sk.description}")
                    self._console.print()
                    self._console.print("[bold]Built-in commands:[/bold]")
                    self._console.print("[dim]  /memory  /tokens  /security  /undo  /diff  /exit[/dim]")
                    self._console.print("[dim]  /permissions — show permission rules and hooks[/dim]")
                    self._console.print("[dim]  /status  — session summary (model, budget, undo stack)[/dim]")
                    self._console.print("[dim]  /context — token breakdown[/dim]")
                    self._console.print("[dim]  /clear   — reset conversation (keeps project memory)[/dim]")
                    self._console.print("[dim]  /rewind N — remove last N turns from history[/dim]")
                    self._console.print("[dim]  /autocompact [off] — toggle dynamic context compression[/dim]")
                    self._console.print("[dim]  /goal <condition> — loop until goal is met[/dim]")
                    self._console.print("[dim]  /btw <question> — side question (not saved to history)[/dim]")
                    self._console.print("[bold]Background agents:[/bold]")
                    self._console.print("[dim]  /fork <task> — run task in background thread[/dim]")
                    self._console.print("[dim]  /tasks       — list background tasks[/dim]")
                    self._console.print("[dim]  /loop [Ns|Nm] <cmd> — run command on schedule (/loop off to stop)[/dim]")
                    self._console.print("[dim]  /batch <task> -- <glob> — fan-out one sub-agent per matched file[/dim]")
                    self._console.print("[dim]  /background <task> — launch detached subprocess (survives terminal close)[/dim]")
                    self._console.print("[dim]  /background         — list detached background jobs[/dim]")
                    self._console.print("[bold]Other:[/bold]")
                    self._console.print("[dim]  /model <provider/model>  — hot-swap LLM for the rest of the session[/dim]")
                    self._console.print("[dim]  /effort low|medium|high|xhigh|max  — change effort level[/dim]")
                    self._console.print("[dim]  /think on|off  — toggle extended thinking (Anthropic)[/dim]")
                    self._console.print("[dim]  /fast [off]  — switch to cheap tier; /fast off to restore[/dim]")
                    self._console.print("[dim]  /compact [topic]  — compress session history (keep topic in detail)[/dim]")
                    self._console.print("[dim]  /recap [N]  — show last N messages to re-orient (default: 5)[/dim]")
                    self._console.print("[dim]  /branch  — snapshot this session to a new branch (resumable)[/dim]")
                    self._console.print("[dim]  /theme [<name>]  — list themes or switch: default|dracula|monokai|solarized[/dim]")
                    self._console.print("[dim]  /rename <name>  — rename this session[/dim]")
                    self._console.print("[dim]  /sessions  — list sessions in this project[/dim]")
                    self._console.print("[dim]  /keybindings  — show keyboard shortcut reference (F1=help, F2=status)[/dim]")
                    self._console.print("[dim]  !<command>  — run a shell command in the project root[/dim]")
                    continue

                if skill_name in skills:
                    skill = skills[skill_name]
                    skill_msg = skill.prompt
                    if extra_args:
                        skill_msg += f"\n\nTarget / context: {extra_args}"

                    # Restrict tools if skill specifies
                    if skill.tools_only:
                        orig_get_defs = self._loop.registry.get_definitions
                        restricted = self._loop.registry.get_restricted_definitions(skill.tools_only)
                        self._loop.registry.get_definitions = lambda _r=restricted: _r  # type: ignore[method-assign]

                    self._console.print(f"[dim]Running skill: /{skill_name}[/dim]")
                    self.run_message(skill_msg)

                    # Restore unrestricted tools
                    if skill.tools_only:
                        self._loop.registry.get_definitions = orig_get_defs  # type: ignore[method-assign]
                    continue

                self._console.print(f"[yellow]Unknown command: /{skill_name}[/yellow]")
                self._console.print(
                    "[dim]Type /help to see available skills.[/dim]"
                )
                continue

            if not self._title_set:
                self._mgr.set_title(self.session_id, raw[:60])
                self._title_set = True

            self.run_message(raw)

    def print_header(self, title: str = "DevAgent") -> None:
        graph = "active" if self._cp_active else "not indexed"
        gh = "active" if self._gh_active else "no token"
        router = "active" if self._router_active else "off"
        mode = "local (offline)" if self._cfg.llm.provider == "ollama" else f"cloud ({self._cfg.llm.provider})"
        effort = self._cfg.llm.effort
        think_suffix = " + thinking" if self._cfg.llm.extended_thinking else ""
        bare_suffix = "  [dim][bare][/dim]" if self._bare else ""
        self._console.print(
            Panel(
                f"[bold cyan]{title}[/bold cyan]  |  "
                f"{self._cfg.llm.provider}/{self._cfg.llm.model}  |  mode: {mode}"
                f"  |  effort: {effort}{think_suffix}{bare_suffix}\n"
                f"[dim]Project: {self._project_root}[/dim]\n"
                f"[dim]Graph: {graph}  |  GitHub: {gh}  |  Router: {router}[/dim]\n"
                "[dim]Type /help for commands — Tab to autocomplete[/dim]",
                border_style="cyan",
            )
        )

    def _print_exit(self) -> None:
        from devagent.agent import permissions as perm_registry
        perm_registry.unregister(self.session_id)
        if self._hook_runner and self._hook_runner.hooks:
            try:
                self._hook_runner.session_end(self.session_id)
            except Exception:
                pass
        self._console.print(f"[dim]{self._budget.status_line()}[/dim]")
        if self.security_log:
            from devagent.tools.security_gate import format_security_report
            self._console.print(
                Panel(
                    format_security_report(self.security_log),
                    title="Security Gate Report",
                    border_style="yellow",
                )
            )


# ---------------------------------------------------------------------------
# Pre-fetch helpers (direct httpx, before the agent starts)
# ---------------------------------------------------------------------------

def _require_gh_token(cfg) -> str:
    token = cfg.github.token
    if not token:
        raise RuntimeError(
            "GitHub token not configured. Run: devagent init"
        )
    return token


def _slug(text: str, max_len: int = 40) -> str:
    """Convert text to a branch-name-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


# ---------------------------------------------------------------------------
# implement flow
# ---------------------------------------------------------------------------

_IMPLEMENT_SYSTEM = """\
## Flow: Implement GitHub Issue

You are implementing a specific GitHub issue end-to-end. Follow this workflow:

1. **Understand**: Read the issue. Use cp_get_file_map and cp_search_symbol to find relevant code.
2. **Branch**: Create a feature branch with git_branch_create (name: feat/issue-{number}-{slug}).
3. **Implement**: Edit the relevant files. Prefer edit_file over write_file for targeted changes.
4. **Test**: Run the test suite with run_shell after each significant change.
5. **Commit**: `run_shell("git add -A && git commit -m 'feat: <description>'")`
6. **Push**: `run_shell("git push -u origin feat/issue-{number}-{slug}")`
7. **PR**: Call gh_create_pr with body including "Closes #{number}" to link the issue.

Be explicit about every file you change and why. If tests fail, fix them before creating the PR.
"""


def run_implement(cfg, project_root: Path, issue_url: str, max_tokens: int | None = None, plan_mode: bool = False, no_limit: bool = False) -> None:
    """Fetch a GitHub issue and run the implement flow."""
    console = Console()
    token = _require_gh_token(cfg)

    parsed = _parse_issue_or_pr_url(issue_url)
    if parsed.kind == "pull":
        raise ValueError("URL points to a PR, not an issue. Use 'devagent review' for PR review.")

    from devagent.tools.github_tools import GitHubAPI
    gh = GitHubAPI(token)

    repo_str = f"{parsed.owner}/{parsed.repo}"

    with console.status(f"[cyan]Fetching issue #{parsed.number} from {repo_str}...[/cyan]"):
        try:
            issue = gh.get(f"/repos/{parsed.owner}/{parsed.repo}/issues/{parsed.number}")
        except Exception as exc:
            raise RuntimeError(f"Could not fetch issue: {exc}") from exc

    title = issue.get("title", "(no title)")
    body = issue.get("body") or "(no body)"
    labels = ", ".join(l["name"] for l in issue.get("labels", []))
    branch_name = f"feat/issue-{parsed.number}-{_slug(title)}"
    extra_system = _IMPLEMENT_SYSTEM.replace("{number}", str(parsed.number)).replace("{slug}", _slug(title))

    console.print()
    console.print(
        Panel(
            f"[bold]#{parsed.number}: {title}[/bold]\n"
            f"[dim]Labels: {labels or 'none'}  |  Repo: {repo_str}[/dim]\n"
            f"[dim]Suggested branch: {branch_name}[/dim]",
            title="Implementing GitHub Issue",
            border_style="green",
        )
    )

    first_message = (
        f"Implement GitHub issue #{parsed.number} in {repo_str}.\n\n"
        f"**Title:** {title}\n\n"
        f"**Body:**\n{body}\n\n"
        f"Suggested branch name: `{branch_name}`\n"
        "Start by exploring the codebase structure, then proceed with the implementation."
    )

    session = DevAgentSession(
        cfg,
        project_root,
        max_tokens=max_tokens,
        extra_system=extra_system,
        plan_mode=plan_mode,
        no_limit=no_limit,
    )
    session._mgr.set_title(session.session_id, f"impl: #{parsed.number} {title[:40]}")
    session._title_set = True
    session.print_header(f"Implement: #{parsed.number} — {title[:50]}")
    session.interactive_repl(first_message=first_message)


# ---------------------------------------------------------------------------
# review flow
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """\
## Flow: Review Pull Request

You are reviewing a GitHub pull request. Your goal is to provide high-quality, actionable feedback.

Workflow:
1. Read the PR diff (provided in your first message)
2. For each changed file, use cp_get_impact to understand the blast radius of the change
3. Check for: bugs, missing error handling, security issues, naming/style problems, missing tests
4. Use gh_review_pr to post your review with precise inline comments (file + line + specific issue)
5. Set event=APPROVE if the code is solid, REQUEST_CHANGES if it needs work, COMMENT for observations

Be constructive and specific. Quote the relevant code in your inline comments. Suggest concrete fixes.
"""


def run_review(cfg, project_root: Path, pr_url: str, max_tokens: int | None = None) -> None:
    """Fetch a PR diff and run the code-review flow."""
    console = Console()
    token = _require_gh_token(cfg)

    parsed = _parse_issue_or_pr_url(pr_url)
    repo_str = f"{parsed.owner}/{parsed.repo}"

    from devagent.tools.github_tools import GitHubAPI
    gh = GitHubAPI(token)

    with console.status(f"[cyan]Fetching PR #{parsed.number} from {repo_str}...[/cyan]"):
        try:
            pr = gh.get(f"/repos/{parsed.owner}/{parsed.repo}/pulls/{parsed.number}")
            files = gh.get(f"/repos/{parsed.owner}/{parsed.repo}/pulls/{parsed.number}/files")
        except Exception as exc:
            raise RuntimeError(f"Could not fetch PR: {exc}") from exc

    title = pr.get("title", "(no title)")
    pr_body = pr.get("body") or "(no description)"
    head_sha = pr["head"]["sha"]
    base_branch = pr["base"]["ref"]
    head_branch = pr["head"]["ref"]

    # Build a compact diff summary for the first message
    diff_lines: list[str] = []
    total_add = total_del = 0
    for f in files[:30]:
        total_add += f.get("additions", 0)
        total_del += f.get("deletions", 0)
        diff_lines.append(
            f"[{f['status'].upper()}] {f['filename']}  "
            f"+{f.get('additions',0)} -{f.get('deletions',0)}"
        )
        patch = f.get("patch", "")
        if patch:
            patch_head = "\n".join(patch.splitlines()[:30])
            diff_lines.append(patch_head)
        diff_lines.append("")
    if len(files) > 30:
        diff_lines.append(f"... and {len(files) - 30} more files")

    console.print()
    console.print(
        Panel(
            f"[bold]PR #{parsed.number}: {title}[/bold]\n"
            f"[dim]{head_branch} → {base_branch}  |  "
            f"+{total_add} -{total_del}  |  {len(files)} files[/dim]",
            title="Reviewing Pull Request",
            border_style="blue",
        )
    )

    first_message = (
        f"Review PR #{parsed.number} in {repo_str}.\n\n"
        f"**Title:** {title}\n"
        f"**Branch:** {head_branch} → {base_branch}\n"
        f"**Head SHA:** {head_sha}\n\n"
        f"**PR Description:**\n{pr_body}\n\n"
        f"**Diff ({len(files)} files, +{total_add} -{total_del}):**\n"
        + "\n".join(diff_lines)
        + "\n\nUse cp_get_impact to assess the blast radius of each change, then submit your review."
    )

    session = DevAgentSession(
        cfg,
        project_root,
        max_tokens=max_tokens,
        extra_system=_REVIEW_SYSTEM,
    )
    session._mgr.set_title(session.session_id, f"review: PR #{parsed.number} {title[:40]}")
    session._title_set = True
    session.print_header(f"Review: PR #{parsed.number} — {title[:50]}")
    session.interactive_repl(first_message=first_message)


# ---------------------------------------------------------------------------
# triage flow
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = """\
## Flow: Triage Open Issues

You are triaging a repository's open issues. For EVERY issue:

1. Read the issue title and body carefully
2. Classify the effort: **trivial** (<1h) | **small** (1-4h) | **medium** (1-2d) | **large** (>2d) | **unclear** (needs more info)
3. Suggest appropriate labels (e.g. bug, enhancement, documentation, good-first-issue, help-wanted)
4. Use gh_comment_issue to post a triage comment with:
   - Effort estimate and rationale
   - Suggested labels
   - Clarifying questions if needed (for "unclear" issues)
5. Move to the next issue

Work systematically. Post one comment per issue. Do not skip issues.
"""


def run_triage(cfg, project_root: Path, repo: str, max_tokens: int | None = None) -> None:
    """Fetch open issues and run the triage flow."""
    console = Console()
    token = _require_gh_token(cfg)

    owner, repo_name = _parse_repo(repo)
    repo_str = f"{owner}/{repo_name}"

    from devagent.tools.github_tools import GitHubAPI
    gh = GitHubAPI(token)

    with console.status(f"[cyan]Fetching open issues for {repo_str}...[/cyan]"):
        try:
            issues = gh.get(f"/repos/{owner}/{repo_name}/issues", state="open", per_page=50)
        except Exception as exc:
            raise RuntimeError(f"Could not fetch issues: {exc}") from exc

    issues = [i for i in issues if "pull_request" not in i]
    if not issues:
        console.print(f"[yellow]No open issues found for {repo_str}.[/yellow]")
        return

    issue_list = "\n".join(
        f"  #{i['number']}  {i['title']}  "
        f"[{', '.join(l['name'] for l in i.get('labels', [])) or 'no labels'}]"
        for i in issues
    )

    console.print()
    console.print(
        Panel(
            f"[bold]{len(issues)} open issues in {repo_str}[/bold]",
            title="Triaging Issues",
            border_style="magenta",
        )
    )

    first_message = (
        f"Triage the following {len(issues)} open issues in {repo_str}.\n\n"
        f"{issue_list}\n\n"
        f"For each issue above, use gh_get_issue to read the full body, "
        f"then post a triage comment with your effort estimate and label suggestions."
    )

    session = DevAgentSession(
        cfg,
        project_root,
        max_tokens=max_tokens,
        extra_system=_TRIAGE_SYSTEM,
    )
    session._mgr.set_title(session.session_id, f"triage: {repo_str} ({len(issues)} issues)")
    session._title_set = True
    session.print_header(f"Triage: {repo_str}")
    session.interactive_repl(first_message=first_message)


# ---------------------------------------------------------------------------
# fix-ci flow
# ---------------------------------------------------------------------------

_FIX_CI_SYSTEM = """\
## Flow: Fix CI Failure

You are fixing a GitHub Actions CI failure. Workflow:

1. The failure logs are in your first message — identify the root cause
2. Use grep and cp_search_symbol to locate the failing code
3. Read the relevant files to understand the context
4. Make targeted, minimal fixes
5. Verify by running the specific failing test locally: run_shell("pytest path/to/test -x -v")
6. Once fixed, commit and push the fix
7. Optionally comment on the failed run's PR if there is one

Focus on the specific error. Avoid sweeping refactors unless the root cause requires it.
"""


def run_fix_ci(cfg, project_root: Path, run_url: str, max_tokens: int | None = None) -> None:
    """Fetch a failed CI run's logs and run the fix-ci flow."""
    console = Console()
    token = _require_gh_token(cfg)

    parsed_run = _parse_run_url(run_url)
    repo_str = f"{parsed_run.owner}/{parsed_run.repo}"

    import httpx as _httpx

    from devagent.tools.github_tools import GitHubAPI

    gh = GitHubAPI(token)

    with console.status(f"[cyan]Fetching run {parsed_run.run_id} from {repo_str}...[/cyan]"):
        try:
            run_info = gh.get(f"/repos/{parsed_run.owner}/{parsed_run.repo}/actions/runs/{parsed_run.run_id}")
            jobs_data = gh.get(f"/repos/{parsed_run.owner}/{parsed_run.repo}/actions/runs/{parsed_run.run_id}/jobs")
        except Exception as exc:
            raise RuntimeError(f"Could not fetch run: {exc}") from exc

    all_jobs = jobs_data.get("jobs", [])
    failed_jobs = [j for j in all_jobs if j.get("conclusion") in ("failure", "timed_out")]
    if not failed_jobs:
        failed_jobs = [j for j in all_jobs if j.get("conclusion") not in ("success", "skipped", None)]

    run_name = run_info.get("display_title") or run_info.get("name", f"Run #{parsed_run.run_id}")
    conclusion = run_info.get("conclusion", "unknown")
    branch = run_info.get("head_branch", "")
    commit_sha = run_info.get("head_sha", "")[:7]

    log_sections: list[str] = []
    for job in failed_jobs[:3]:
        job_id = job["id"]
        bad_steps = [s["name"] for s in job.get("steps", []) if s.get("conclusion") in ("failure", "timed_out")]
        log_sections.append(f"Job: {job.get('name', '?')} — failed steps: {', '.join(bad_steps) or 'none'}")
        try:
            r = _httpx.get(
                f"https://api.github.com/repos/{parsed_run.owner}/{parsed_run.repo}/actions/jobs/{job_id}/logs",
                headers=gh._headers,
                follow_redirects=True,
                timeout=60,
            )
            if r.status_code == 200:
                tail = r.text.splitlines()[-120:]
                log_sections.append("\n".join(tail))
        except Exception:
            log_sections.append("(could not fetch log)")

    console.print()
    console.print(
        Panel(
            f"[bold]Run: {run_name}[/bold]\n"
            f"[dim]Conclusion: {conclusion}  |  Branch: {branch}  |  Commit: {commit_sha}[/dim]\n"
            f"[dim]Failed jobs: {len(failed_jobs)}[/dim]",
            title="Fix CI Failure",
            border_style="red",
        )
    )

    first_message = (
        f"Fix the CI failure in run {parsed_run.run_id} of {repo_str}.\n\n"
        f"**Run:** {run_name}\n"
        f"**Branch:** {branch}  |  **Commit:** {commit_sha}\n\n"
        f"**Failed job logs:**\n"
        + "\n\n---\n\n".join(log_sections)
        + "\n\nIdentify the root cause, locate the failing code, and implement a fix."
    )

    session = DevAgentSession(
        cfg,
        project_root,
        max_tokens=max_tokens,
        extra_system=_FIX_CI_SYSTEM,
    )
    session._mgr.set_title(session.session_id, f"fix-ci: run {parsed_run.run_id} ({repo_str})")
    session._title_set = True
    session.print_header(f"Fix CI: {run_name[:50]}")
    session.interactive_repl(first_message=first_message)


# ---------------------------------------------------------------------------
# autofix-pr flow  (Phase 24)
# ---------------------------------------------------------------------------

_AUTOFIX_PR_SYSTEM = """\
## Flow: Auto-Fix Pull Request

You are watching a pull request and fixing issues automatically.
Rules:
1. Fix only what is reported — no extra refactoring.
2. After each fix: run the relevant tests, commit with a clear message, push.
3. Keep fixes minimal and targeted; prefer the simplest correct change.
"""

_AUTOFIX_CI_MSG = """\
The following CI checks failed on PR #{pr_number} ({pr_title}).
Branch: {branch}  |  Commit: {sha}

Failed job logs:
{logs}

Identify the root cause, fix it, run the tests to verify, then commit and push.
"""

_AUTOFIX_REVIEW_MSG = """\
The following review comment(s) were left on PR #{pr_number} ({pr_title}).
Branch: {branch}

{comments}

Address each comment, make the required changes, then commit and push.
"""


def _fetch_failed_job_logs(gh, owner: str, repo: str, run_id: str) -> tuple[list[dict], str]:
    """Return (failed_jobs, concatenated_log_text) for a CI run."""
    import httpx as _httpx
    jobs_data = gh.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs")
    all_jobs = jobs_data.get("jobs", [])
    failed_jobs = [j for j in all_jobs if j.get("conclusion") in ("failure", "timed_out")]
    if not failed_jobs:
        failed_jobs = [j for j in all_jobs if j.get("conclusion") not in ("success", "skipped", None)]

    log_parts: list[str] = []
    for job in failed_jobs[:3]:
        job_id = job["id"]
        bad_steps = [s["name"] for s in job.get("steps", []) if s.get("conclusion") in ("failure", "timed_out")]
        log_parts.append(f"Job: {job.get('name', '?')} — failed steps: {', '.join(bad_steps) or 'none'}")
        try:
            r = _httpx.get(
                f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
                headers=gh._headers,
                follow_redirects=True,
                timeout=60,
            )
            if r.status_code == 200:
                log_parts.append("\n".join(r.text.splitlines()[-120:]))
        except Exception:
            log_parts.append("(could not fetch log)")

    return failed_jobs, "\n\n---\n\n".join(log_parts)


def run_autofix_pr(
    cfg,
    project_root: Path,
    pr_url: str,
    poll_interval: int = 120,
    max_polls: int | None = None,
) -> None:
    """Watch a PR for CI failures and review comments, auto-fix and push.

    Polls every `poll_interval` seconds (default 120).
    Stops when max_polls reached or on KeyboardInterrupt.
    """
    import subprocess
    import time

    from rich.console import Console as _Console

    from devagent.tools.github_tools import GitHubAPI

    console = _Console()
    token = _require_gh_token(cfg)
    gh = GitHubAPI(token)

    parsed = _parse_issue_or_pr_url(pr_url)
    owner, repo, pr_number = parsed.owner, parsed.repo, parsed.number
    repo_str = f"{owner}/{repo}"

    # Fetch PR metadata
    pr_data = gh.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    pr_title = pr_data.get("title", f"PR #{pr_number}")
    branch = pr_data.get("head", {}).get("ref", "")
    if not branch:
        raise RuntimeError("Could not determine PR branch from GitHub API.")

    console.print(
        Panel(
            f"[bold]Watching PR #{pr_number}: {pr_title}[/bold]\n"
            f"[dim]Repo: {repo_str}  |  Branch: {branch}[/dim]\n"
            f"[dim]Poll interval: {poll_interval}s  |  Press Ctrl-C to stop[/dim]",
            title="autofix-pr",
            border_style="cyan",
        )
    )

    # Checkout PR branch locally so fixes land on the right branch
    try:
        subprocess.run(
            ["git", "fetch", "origin", branch], cwd=str(project_root),
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "checkout", branch], cwd=str(project_root),
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not checkout branch {branch!r}: {exc.stderr.decode().strip()}") from exc

    seen_run_ids: set[str] = set()
    seen_comment_ids: set[int] = set()

    # Seed seen_run_ids with already-completed runs so we don't re-process them
    try:
        runs_data = gh.get(f"/repos/{owner}/{repo}/actions/runs?branch={branch}&per_page=20")
        for run in runs_data.get("workflow_runs", []):
            if run.get("conclusion") in ("failure", "timed_out", "success", "cancelled"):
                seen_run_ids.add(str(run["id"]))
    except Exception:
        pass

    # Seed seen_comment_ids
    try:
        comments = gh.get(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments")
        for c in comments:
            seen_comment_ids.add(c["id"])
    except Exception:
        pass

    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        console.print(f"[dim]Poll #{polls} — checking PR #{pr_number}…[/dim]")

        # ---- Check for new failed CI runs ----
        try:
            runs_data = gh.get(f"/repos/{owner}/{repo}/actions/runs?branch={branch}&per_page=20")
            for run in runs_data.get("workflow_runs", []):
                run_id = str(run["id"])
                conclusion = run.get("conclusion")
                status = run.get("status")
                if run_id in seen_run_ids:
                    continue
                if status != "completed":
                    continue
                seen_run_ids.add(run_id)
                if conclusion not in ("failure", "timed_out"):
                    continue

                console.print(f"[yellow]CI failure detected: run {run_id} — {run.get('name', '?')}[/yellow]")

                # Pull latest before fixing
                subprocess.run(
                    ["git", "pull", "origin", branch], cwd=str(project_root),
                    capture_output=True,
                )

                failed_jobs, log_text = _fetch_failed_job_logs(gh, owner, repo, run_id)
                if not failed_jobs:
                    console.print("[dim]No failed jobs found; skipping.[/dim]")
                    continue

                head_sha = run.get("head_sha", "")[:7]
                message = _AUTOFIX_CI_MSG.format(
                    pr_number=pr_number,
                    pr_title=pr_title,
                    branch=branch,
                    sha=head_sha,
                    logs=log_text,
                )
                session = DevAgentSession(
                    cfg, project_root, bare=True, extra_system=_AUTOFIX_PR_SYSTEM,
                )
                session.run_message(message, quiet=True)
                console.print(f"[green]CI fix applied for run {run_id}.[/green]")

        except Exception as exc:
            console.print(f"[red]Error checking CI runs: {exc}[/red]")

        # ---- Check for new review comments ----
        try:
            new_comments: list[dict] = []
            comments = gh.get(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments")
            for c in comments:
                if c["id"] not in seen_comment_ids:
                    seen_comment_ids.add(c["id"])
                    new_comments.append(c)

            if new_comments:
                console.print(f"[yellow]{len(new_comments)} new review comment(s) found.[/yellow]")

                # Pull latest before fixing
                subprocess.run(
                    ["git", "pull", "origin", branch], cwd=str(project_root),
                    capture_output=True,
                )

                comment_text = "\n\n".join(
                    f"File: {c.get('path', '?')} (line {c.get('line', '?')})\n"
                    f"Comment: {c.get('body', '(empty)')}"
                    for c in new_comments
                )
                message = _AUTOFIX_REVIEW_MSG.format(
                    pr_number=pr_number,
                    pr_title=pr_title,
                    branch=branch,
                    comments=comment_text,
                )
                session = DevAgentSession(
                    cfg, project_root, bare=True, extra_system=_AUTOFIX_PR_SYSTEM,
                )
                session.run_message(message, quiet=True)
                console.print(f"[green]Review comments addressed ({len(new_comments)} comment(s)).[/green]")

        except Exception as exc:
            console.print(f"[red]Error checking review comments: {exc}[/red]")

        if max_polls is not None and polls >= max_polls:
            break

        try:
            console.print(f"[dim]Sleeping {poll_interval}s…[/dim]")
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            console.print("\n[dim]autofix-pr stopped.[/dim]")
            break
