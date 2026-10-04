"""Tests for WatcherScheduler progress lifecycle and concurrency."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.progress import Progress, TaskID

from devagent.core.config import DevAgentConfig
from devagent.core.models import IssueComplexity, WatchedRepo, WatcherAnalysis
from devagent.watcher.scheduler import WatcherScheduler


def _make_sample_analysis(owner: str, repo: str, issue_number: int) -> WatcherAnalysis:
    return WatcherAnalysis(
        owner=owner,
        repo=repo,
        issue_number=issue_number,
        issue_title=f"Sample issue {issue_number}",
        issue_url=f"https://github.com/{owner}/{repo}/issues/{issue_number}",
        analysed_at=datetime.now(UTC),
        requirements_count=1,
        conflicts_count=0,
        complexity=IssueComplexity.LOW,
        touched_files=["app.py"],
        conflicted_files=[],
        requirement_summaries=[],
        full_report_available=False,
    )


@pytest.mark.asyncio
async def test_single_repo_run_for_repo(tmp_path: Path):
    """Test that a single repository check completes cleanly without errors."""
    config = DevAgentConfig()
    scheduler = WatcherScheduler(config, tmp_path, interval_minutes=30)

    repo = WatchedRepo(
        owner="test-owner",
        repo="test-repo",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )

    analysis = _make_sample_analysis("test-owner", "test-repo", 1)

    async def fake_run_check(watched_repo, progress_callback=None):
        if progress_callback:
            progress_callback(1, "Sample issue 1", "analysing")
        return [analysis]

    scheduler.checker.run_check = AsyncMock(side_effect=fake_run_check)

    with (
        patch(
            "devagent.watcher.scheduler.get_all_analyses_for_repo",
            AsyncMock(return_value=[analysis]),
        ),
        patch("devagent.watcher.scheduler.save_cross_conflicts_for_repo", AsyncMock()),
        patch("devagent.watcher.scheduler.render_health_report", MagicMock()) as mock_render,
    ):
        await scheduler._run_for_repo(repo)
        assert mock_render.called
        report = mock_render.call_args[0][0]
        assert report.owner == "test-owner"
        assert report.repo == "test-repo"
        assert len(report.new_analyses) == 1


@pytest.mark.asyncio
async def test_concurrent_repo_checks_with_shared_progress(tmp_path: Path):
    """Test that multiple repository checks run concurrently using a shared Progress."""
    config = DevAgentConfig()
    scheduler = WatcherScheduler(config, tmp_path, interval_minutes=30)

    shared_progress = Progress()
    shared_progress.start = MagicMock()
    shared_progress.stop = MagicMock()
    shared_progress.add_task = MagicMock(side_effect=[TaskID(1), TaskID(2)])
    shared_progress.update = MagicMock()
    shared_progress.remove_task = MagicMock()

    repo1 = WatchedRepo(
        owner="owner1",
        repo="repo1",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )
    repo2 = WatchedRepo(
        owner="owner2",
        repo="repo2",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )

    entered = []

    async def fake_run_check(watched_repo, progress_callback=None):
        entered.append(watched_repo.repo)
        if progress_callback:
            progress_callback(1, f"Task for {watched_repo.repo}", "analysing")
        # Yield to allow concurrency
        await asyncio.sleep(0.01)
        return [_make_sample_analysis(watched_repo.owner, watched_repo.repo, 1)]

    scheduler.checker.run_check = AsyncMock(side_effect=fake_run_check)

    with (
        patch("devagent.watcher.scheduler.get_all_analyses_for_repo", AsyncMock(return_value=[])),
        patch("devagent.watcher.scheduler.save_cross_conflicts_for_repo", AsyncMock()),
        patch("devagent.watcher.scheduler.render_health_report", MagicMock()),
    ):
        await asyncio.gather(
            scheduler._run_for_repo(repo1, progress=shared_progress),
            scheduler._run_for_repo(repo2, progress=shared_progress),
        )

    assert set(entered) == {"repo1", "repo2"}
    assert shared_progress.add_task.call_count == 2
    assert shared_progress.update.call_count == 2
    assert shared_progress.remove_task.call_count == 2
    # When shared progress is passed, _run_for_repo should not stop the shared progress instance
    assert shared_progress.stop.call_count == 0


@pytest.mark.asyncio
async def test_concurrent_repo_checks_standalone_no_competing_threads(tmp_path: Path):
    """Test that concurrent _run_for_repo calls without explicit Progress handle tasks safely."""
    config = DevAgentConfig()
    scheduler = WatcherScheduler(config, tmp_path, interval_minutes=30)

    repo1 = WatchedRepo(
        owner="owner1",
        repo="repo1",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )
    repo2 = WatchedRepo(
        owner="owner2",
        repo="repo2",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )

    async def fake_run_check(watched_repo, progress_callback=None):
        if progress_callback:
            progress_callback(1, f"Task for {watched_repo.repo}", "analysing")
        await asyncio.sleep(0.01)
        return []

    scheduler.checker.run_check = AsyncMock(side_effect=fake_run_check)

    with patch("devagent.watcher.scheduler.console.print", MagicMock()):
        # Overlapping standalone executions should both complete without exceptions
        await asyncio.gather(
            scheduler._run_for_repo(repo1),
            scheduler._run_for_repo(repo2),
        )


@pytest.mark.asyncio
async def test_run_for_repo_no_new_issues(tmp_path: Path):
    """Test that when run_check returns empty list, no report is rendered."""
    config = DevAgentConfig()
    scheduler = WatcherScheduler(config, tmp_path, interval_minutes=30)

    repo = WatchedRepo(
        owner="test-owner",
        repo="test-repo",
        registered_at=datetime.now(UTC),
        check_interval_minutes=30,
        is_active=True,
    )

    scheduler.checker.run_check = AsyncMock(return_value=[])

    with (
        patch("devagent.watcher.scheduler.console.print", MagicMock()) as mock_print,
        patch("devagent.watcher.scheduler.render_health_report", MagicMock()) as mock_render,
    ):
        await scheduler._run_for_repo(repo)
        assert not mock_render.called
        assert any(
            "no new issues since last check" in str(call_arg)
            for call_arg in mock_print.call_args_list
        )
