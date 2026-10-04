"""APScheduler-based background scheduler for F3 Repo Health Monitor."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from devagent.core.config import DevAgentConfig
from devagent.core.models import WatchedRepo, WatchHealthReport
from devagent.output.watcher_renderer import render_health_report
from devagent.watcher.checker import WatcherChecker
from devagent.watcher.conflict_detector import CrossIssueConflictDetector
from devagent.watcher.storage import (
    get_all_analyses_for_repo,
    list_watched_repos,
    save_cross_conflicts_for_repo,
)

console = Console()


class WatcherScheduler:
    """Manages the background scheduling of repo health checks."""

    def __init__(
        self,
        config: DevAgentConfig,
        project_root: Path,
        interval_minutes: int,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.interval_minutes = interval_minutes
        self.checker = WatcherChecker(config, project_root)
        self.conflict_detector = CrossIssueConflictDetector()
        self._progress: Progress | None = None
        self._progress_lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Starts the scheduler. Runs a check immediately on start,
        then on the configured interval. Blocks until Ctrl+C.
        """
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = AsyncIOScheduler()

        # Run immediately on start
        await self._run_all_repos()

        scheduler.add_job(
            self._run_all_repos,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="repo_health_check",
            replace_existing=True,
        )
        scheduler.start()

        console.print(
            f"[dim]Watcher running. Checking every {self.interval_minutes} minutes. "
            f"Press Ctrl+C to stop.[/dim]"
        )

        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown(wait=False)
            console.print("\n[dim]Watcher stopped.[/dim]")

    async def run_once(self) -> None:
        """Runs a single check for all watched repos and exits."""
        await self._run_all_repos()

    async def _run_all_repos(self) -> None:
        """Runs the check for every active watched repo."""
        watched_repos = await list_watched_repos()

        if not watched_repos:
            console.print(
                "[dim]No repos being watched. Run 'devagent watch --repo owner/repo' to start.[/dim]"
            )
            return

        for watched_repo in watched_repos:
            await self._run_for_repo(watched_repo)

    async def _run_for_repo(
        self,
        watched_repo: WatchedRepo,
        progress: Progress | None = None,
    ) -> None:
        """Runs the full check cycle for one repo."""
        active_progress = progress or self._progress
        own_progress = False

        if active_progress is None:
            active_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console,
            )
            active_progress.start()
            own_progress = True

        async with self._progress_lock:
            task = active_progress.add_task(
                f"Checking {watched_repo.owner}/{watched_repo.repo}...",
                total=None,
            )

        def update_progress(issue_number: int, issue_title: str, status: str) -> None:
            active_progress.update(
                task,
                description=f"Analysing #{issue_number}: {issue_title[:40]}...",
            )

        try:
            new_analyses = await self.checker.run_check(
                watched_repo,
                progress_callback=update_progress,
            )
        finally:
            async with self._progress_lock:
                active_progress.remove_task(task)
            if own_progress:
                active_progress.stop()

        if not new_analyses:
            console.print(
                f"[dim]{watched_repo.owner}/{watched_repo.repo}: "
                f"no new issues since last check.[/dim]"
            )
            return

        # Detect cross-issue conflicts across all analyses for this repo
        all_analyses = await get_all_analyses_for_repo(watched_repo.owner, watched_repo.repo)
        conflicts = self.conflict_detector.detect(all_analyses)
        await save_cross_conflicts_for_repo(watched_repo.owner, watched_repo.repo, conflicts)

        health_report = WatchHealthReport(
            owner=watched_repo.owner,
            repo=watched_repo.repo,
            check_run_at=datetime.now(UTC),
            new_issues_count=len(new_analyses),
            new_analyses=new_analyses,
            cross_issue_conflicts=conflicts,
            total_watched_issues=len(all_analyses),
        )
        render_health_report(health_report)
