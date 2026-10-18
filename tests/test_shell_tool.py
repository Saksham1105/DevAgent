"""Tests for ShellSession side-effect tracking (cd / export parsing).

Regression coverage for issue #70: a `cd` appearing on its own line of a
multi-line command must update the session cwd, exactly like one after `;`.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from devagent.tools.registry import ToolRegistry
from devagent.tools.shell_tool import ShellSession, register_shell_tool


@pytest.fixture()
def session(tmp_path: Path) -> Generator[ShellSession, None, None]:
    """A session rooted in a temp dir that contains a `sub` subdirectory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "other").mkdir()
    sess = ShellSession(str(tmp_path))
    try:
        yield sess
    finally:
        sess.cleanup()


def test_starts_in_project_root(session: ShellSession, tmp_path: Path) -> None:
    assert session.cwd == str(tmp_path.resolve())


@pytest.mark.parametrize(
    "command",
    [
        "echo hi; cd sub",
        "true && cd sub",
        "false || cd sub",
    ],
)
def test_cd_after_sequencing_operator(
    session: ShellSession, tmp_path: Path, command: str
) -> None:
    session.apply_side_effects(command)
    assert session.cwd == str((tmp_path / "sub").resolve())


@pytest.mark.parametrize(
    "command",
    [
        "echo hi\ncd sub\npwd",       # \n
        "echo hi\r\ncd sub\r\npwd",   # \r\n (Windows newlines from tools)
        "\ncd sub",                    # leading newline
        "echo hi\n    cd sub",         # indented line inside a script/heredoc
    ],
)
def test_cd_on_its_own_line(
    session: ShellSession, tmp_path: Path, command: str
) -> None:
    """Issue #70: a newline-joined `cd` was previously not detected."""
    session.apply_side_effects(command)
    assert session.cwd == str((tmp_path / "sub").resolve())


def test_multiple_cd_apply_in_order(session: ShellSession, tmp_path: Path) -> None:
    """Each `cd` resolves against the cwd left by the previous one (shell order)."""
    session.apply_side_effects("cd sub\ncd ../other")
    assert session.cwd == str((tmp_path / "other").resolve())


def test_cd_in_single_pipe_does_not_move(session: ShellSession, tmp_path: Path) -> None:
    """`cd` in a pipeline runs in a subshell — it must not change session cwd."""
    session.apply_side_effects("echo x | cd sub")
    assert session.cwd == str(tmp_path.resolve())


def test_cd_in_subshell_does_not_move(session: ShellSession, tmp_path: Path) -> None:
    session.apply_side_effects("( cd sub; pwd )")
    assert session.cwd == str(tmp_path.resolve())


def test_cd_dash_is_ignored(session: ShellSession, tmp_path: Path) -> None:
    session.apply_side_effects("cd -")
    assert session.cwd == str(tmp_path.resolve())


def test_cd_to_missing_dir_is_ignored(session: ShellSession, tmp_path: Path) -> None:
    session.apply_side_effects("cd no_such_dir")
    assert session.cwd == str(tmp_path.resolve())


def test_cd_relative_resolves_against_current_cwd(
    session: ShellSession, tmp_path: Path
) -> None:
    session.apply_side_effects("cd sub\ncd ..")
    assert session.cwd == str(tmp_path.resolve())


def test_export_after_newline_is_tracked(session: ShellSession) -> None:
    """The export parser is separator-agnostic; a multi-line command still works."""
    session.apply_side_effects("echo hi\nexport FOO=bar")
    assert session.env["FOO"] == "bar"


def test_cleanup_removes_temp_dir(tmp_path: Path) -> None:
    """Cleanup removes the session's temporary directory from disk."""
    sess = ShellSession(str(tmp_path))
    temp_dir = sess.tmp_dir
    assert temp_dir.exists()
    assert temp_dir.is_dir()

    sess.cleanup()
    assert not temp_dir.exists()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    """Calling cleanup() multiple times does not raise any exception."""
    sess = ShellSession(str(tmp_path))
    temp_dir = sess.tmp_dir
    assert temp_dir.exists()

    sess.cleanup()
    assert not temp_dir.exists()

    # Second call when directory is already gone
    sess.cleanup()
    assert not temp_dir.exists()


def test_context_manager_cleans_up_on_exit(tmp_path: Path) -> None:
    """Using ShellSession as a context manager ensures cleanup on exit."""
    with ShellSession(str(tmp_path)) as sess:
        temp_dir = sess.tmp_dir
        assert temp_dir.exists()

    assert not temp_dir.exists()


def test_context_manager_cleans_up_on_exception(tmp_path: Path) -> None:
    """Using ShellSession as a context manager ensures cleanup even if an exception occurs."""
    temp_dir: Path | None = None
    with pytest.raises(RuntimeError, match="boom"), ShellSession(str(tmp_path)) as sess:
        temp_dir = sess.tmp_dir
        assert temp_dir.exists()
        raise RuntimeError("boom")

    assert temp_dir is not None
    assert not temp_dir.exists()


def test_large_output_cleaned_up_with_session(tmp_path: Path) -> None:
    """Large command output stored in temporary directory is removed upon session cleanup."""
    sess = ShellSession(str(tmp_path))

    # Create dummy output files in the session tmp_dir
    out_file1 = sess.next_output_path()
    out_file1.write_text("line\n" * 100, encoding="utf-8")
    out_file2 = sess.next_output_path()
    out_file2.write_text("another\n" * 60, encoding="utf-8")

    assert out_file1.exists()
    assert out_file2.exists()
    temp_dir = sess.tmp_dir
    assert temp_dir.exists()

    sess.cleanup()
    assert not out_file1.exists()
    assert not out_file2.exists()
    assert not temp_dir.exists()


def test_register_shell_tool_registers_atexit_cleanup(tmp_path: Path) -> None:
    """register_shell_tool registers an atexit handler for the session's cleanup."""
    registry = ToolRegistry()
    registered_handlers: list[tuple] = []

    def fake_register(func, *args, **kwargs):
        registered_handlers.append((func, args, kwargs))

    with patch("atexit.register", side_effect=fake_register):
        register_shell_tool(registry, project_root=str(tmp_path))

    # Assert that a cleanup function was registered
    assert len(registered_handlers) == 1
    func, _, _ = registered_handlers[0]
    assert callable(func)
    assert func.__name__ == "cleanup"
    # Execute the registered cleanup to ensure it works and doesn't leak the test dir
    func()

