"""Shell execution tool — streaming output, stateful cwd/env, background processes.

Improvements over the original:
- Large output streamed to a temp file; last 50 lines returned inline
- ShellSession maintains persistent cwd and env across calls
- read_shell_output tool for paginating large output files
- Background process support via background=True + poll_shell tool
- Configurable timeout (default 300s, 0 = no timeout)
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Self

from devagent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Blocklist — never allowed regardless of other settings
# ---------------------------------------------------------------------------

_BLOCKLIST: list[re.Pattern] = [
    re.compile(r"\brm\s+-rf\s+/"),             # rm -rf /
    re.compile(r"\bdd\b.*\bof=/dev/"),          # dd to device
    re.compile(r"\b(shutdown|reboot|halt)\b"),  # system control
    re.compile(r"\bmkfs\b"),                    # format disk
    re.compile(r"\bchmod\s+[0-7]*7[0-7]{2}\s+/"),  # world-writable root files
    re.compile(r">\s*/etc/"),                   # overwrite system files
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh"),      # curl | bash
    re.compile(r"\bwget\b.*\|\s*(ba)?sh"),      # wget | bash
    re.compile(r"\bcat\s+.*\.env\b"),           # read .env files
    re.compile(r"\bprintenv\b"),                # dump environment
]


def _is_blocked(command: str) -> str | None:
    for pattern in _BLOCKLIST:
        if pattern.search(command):
            return pattern.pattern
    return None


# ---------------------------------------------------------------------------
# ShellSession — stateful cwd + env per registry instance
# ---------------------------------------------------------------------------

class ShellSession:
    """Maintains persistent cwd and environment variables across shell calls."""

    def __init__(self, project_root: str) -> None:
        self.cwd = str(Path(project_root).resolve())
        self.env: dict[str, str] = os.environ.copy()
        # Temp dir for this session's output files
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="devagent_shell_"))
        self._cmd_counter = 0

    def next_output_path(self) -> Path:
        self._cmd_counter += 1
        return self._tmp_dir / f"cmd_{self._cmd_counter:04d}.txt"

    def apply_side_effects(self, command: str) -> None:
        """Parse cd and export statements and update session state."""
        # Handle `cd <path>` — last cd wins.
        # Separators are the shell sequencing operators plus a newline: a `cd`
        # on its own line of a multi-line command (heredoc / script joined with
        # \n or \r\n) changes the shell's cwd just like one after `;`. Note this
        # is deliberately not a character class including `|` — `cd` in a
        # pipeline runs in a subshell and must not move the session cwd.
        for m in re.finditer(r"(?:^|;|&&|\|\||\n)\s*cd\s+([^\s;|&]+)", command):
            target = m.group(1).strip().strip('"').strip("'")
            if target == "-":
                continue
            if target.startswith("~"):
                target = str(Path.home()) + target[1:]
            new_cwd = str((Path(self.cwd) / target).resolve())
            if Path(new_cwd).is_dir():
                self.cwd = new_cwd

        # Handle `export KEY=VALUE`
        for m in re.finditer(r"export\s+([A-Za-z_][A-Za-z0-9_]*)=([^\s;|&]*)", command):
            key = m.group(1)
            val = m.group(2).strip('"').strip("'")
            self.env[key] = val

    @property
    def tmp_dir(self) -> Path:
        """Return the path to this session's temporary directory."""
        return self._tmp_dir

    def cleanup(self) -> None:
        """Safely remove this session's temporary directory if it exists."""
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Background process registry (module-level, keyed by string PID)
# ---------------------------------------------------------------------------

_bg_processes: dict[str, dict] = {}  # pid_str -> {proc, output_path, thread}


def _stream_to_file(proc: subprocess.Popen, output_path: Path) -> None:
    """Thread target: stream proc stdout+stderr to a file."""
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        for line in proc.stdout:  # type: ignore[union-attr]
            f.write(line)
            f.flush()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_shell_tool(
    registry: ToolRegistry,
    project_root: str = ".",
    timeout_sec: int = 300,
) -> None:
    session = ShellSession(project_root)
    atexit.register(session.cleanup)

    # ------------------------------------------------------------------
    # run_shell
    # ------------------------------------------------------------------
    def run_shell(args: dict) -> str:
        command = (args.get("command") or args.get("cmd") or "").strip()
        timeout = args.get("timeout", timeout_sec)
        background = bool(args.get("background", False))

        if not command:
            return "[error] No command provided"

        blocked = _is_blocked(command)
        if blocked:
            return f"[blocked] Command matches blocklist pattern: {blocked!r}"

        # Apply cd/export side effects to session state
        session.apply_side_effects(command)

        if background:
            output_path = session.next_output_path()
            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=session.cwd,
                    env=session.env,
                )
                t = threading.Thread(
                    target=_stream_to_file,
                    args=(proc, output_path),
                    daemon=True,
                )
                t.start()
                pid_str = str(proc.pid)
                _bg_processes[pid_str] = {
                    "proc": proc,
                    "output_path": output_path,
                    "thread": t,
                }
                return (
                    f"[background] Started PID {pid_str}.\n"
                    f"Output: {output_path}\n"
                    f"Use poll_shell(pid=\"{pid_str}\") to check status."
                )
            except Exception as exc:
                return f"[error] Could not start background process: {exc}"

        # Foreground execution
        timeout_arg = timeout if timeout and timeout > 0 else None
        output_path = session.next_output_path()

        try:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_arg,
                cwd=session.cwd,
                env=session.env,
            )
            combined = result.stdout or ""
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            return f"[error] Command timed out after {timeout_arg}s"
        except Exception as exc:
            return f"[error] {exc}"

        lines = combined.splitlines()
        total = len(lines)

        # Small output — return inline, no temp file
        if total <= 50:
            text = combined.strip()
            if exit_code != 0:
                text = (text + f"\n[exit code: {exit_code}]") if text else f"[exit code: {exit_code}]"
            return text or "(no output)"

        # Large output — write to temp file, return tail
        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(combined)

        tail = "\n".join(lines[-50:])
        header = (
            f"[Command completed. Exit: {exit_code}. "
            f"Total: {total} lines. "
            f"Full output: {output_path}]\n\n"
            "--- Last 50 lines ---\n"
        )
        return header + tail

    registry.register(
        "run_shell",
        (
            "Execute a shell command in the project directory. "
            "Supports background=true for long-running commands. "
            "Large output is saved to a temp file; use read_shell_output to page through it. "
            "State (cwd, env) persists across calls within a session. "
            "Dangerous commands (rm -rf /, curl|bash, etc.) are blocked."
        ),
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (0 = no timeout, default 300)",
                    "default": 300,
                },
                "background": {
                    "type": "boolean",
                    "description": "Start in background and return immediately",
                    "default": False,
                },
            },
            "required": ["command"],
        },
        run_shell,
    )

    # ------------------------------------------------------------------
    # read_shell_output
    # ------------------------------------------------------------------
    def read_shell_output(args: dict) -> str:
        path = args.get("path", "").strip()
        offset = int(args.get("offset", 0))
        lines_to_read = int(args.get("lines", 100))

        if not path:
            return "[error] path is required"

        p = Path(path)
        if not p.exists():
            return f"[error] File not found: {path}"

        try:
            all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"[error] Could not read file: {exc}"

        total = len(all_lines)
        slice_lines = all_lines[offset: offset + lines_to_read]
        has_more = (offset + lines_to_read) < total

        result = "\n".join(slice_lines)
        footer = (
            f"\n\n[Lines {offset + 1}–{offset + len(slice_lines)} of {total}]"
        )
        if has_more:
            footer += f" — use offset={offset + lines_to_read} to read more"
        return result + footer

    registry.register(
        "read_shell_output",
        (
            "Read lines from a shell output file (created by run_shell for large outputs). "
            "Paginate with offset and lines parameters."
        ),
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the shell output file"},
                "offset": {"type": "integer", "description": "Line offset to start reading from", "default": 0},
                "lines": {"type": "integer", "description": "Number of lines to read", "default": 100},
            },
            "required": ["path"],
        },
        read_shell_output,
    )

    # ------------------------------------------------------------------
    # poll_shell
    # ------------------------------------------------------------------
    def poll_shell(args: dict) -> str:
        pid_str = str(args.get("pid", "")).strip()
        if not pid_str or pid_str not in _bg_processes:
            known = ", ".join(_bg_processes.keys()) or "none"
            return f"[error] Unknown PID: {pid_str!r}. Known background PIDs: {known}"

        entry = _bg_processes[pid_str]
        proc: subprocess.Popen = entry["proc"]
        output_path: Path = entry["output_path"]

        poll_result = proc.poll()
        is_done = poll_result is not None

        # Count lines written so far
        try:
            lines = output_path.read_text(encoding="utf-8", errors="replace").splitlines()
            line_count = len(lines)
            tail = "\n".join(lines[-20:]) if lines else "(no output yet)"
        except Exception:
            line_count = 0
            tail = "(output file not readable yet)"

        if is_done:
            del _bg_processes[pid_str]
            return (
                f"[done] PID {pid_str} exited with code {poll_result}. "
                f"Total: {line_count} lines. Full output: {output_path}\n\n"
                f"--- Last 20 lines ---\n{tail}"
            )

        return (
            f"[running] PID {pid_str} still running. "
            f"Lines so far: {line_count}. Full output: {output_path}\n\n"
            f"--- Last 20 lines ---\n{tail}"
        )

    registry.register(
        "poll_shell",
        "Check the status of a background shell process started with run_shell(background=true).",
        {
            "type": "object",
            "properties": {
                "pid": {"type": "string", "description": "PID string returned by run_shell(background=true)"},
            },
            "required": ["pid"],
        },
        poll_shell,
    )
