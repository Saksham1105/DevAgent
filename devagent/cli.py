"""DevAgent CLI - all command definitions using Typer + Rich."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Load .env from the project root before anything else touches env vars.
# This lets OLLAMA_HOST / OLLAMA_API_KEY wire Ollama Cloud without shell exports.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from devagent.core.config import (
    BraveConfig,
    DevAgentConfig,
    GitHubConfig,
    LLMConfig,
    OutputConfig,
    SearchXConfig,
    config_exists,
    load_config,
    save_config,
)
from devagent.core.storage import get_config_path
from devagent.core.url_parser import InvalidGitHubURLError, format_repo_string, parse_github_url

app = typer.Typer(
    name="devagent",
    help="DevAgent - AI development agent for specification-to-code implementation.",
    add_completion=False,
    rich_markup_mode="rich",
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()

# Global verbose flag
_verbose = False


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full tracebacks on error."),
) -> None:
    """DevAgent - AI development agent for specification-to-code implementation."""
    global _verbose
    _verbose = verbose


def _handle_error(exc: Exception) -> None:
    """Display a user-friendly error panel instead of a raw traceback."""
    import traceback

    from devagent.mcp.manager import NodeNotFoundError

    # Map known exceptions to friendly messages
    error_msg = str(exc)
    hint = ""

    if isinstance(exc, NodeNotFoundError):
        error_msg = "Node.js is required for DevAgent."
        hint = "Download it from: https://nodejs.org/"
    elif "ConnectError" in type(exc).__name__ or "Connection refused" in error_msg:
        error_msg = "Ollama is not reachable."
        hint = "Start it with: ollama serve"
    elif "401" in error_msg or "token" in error_msg.lower() and "invalid" in error_msg.lower():
        error_msg = "Your GitHub token appears to be invalid or expired."
        hint = "Run: devagent init to update it."
    elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
        error_msg = "API rate limit reached."
        hint = "Wait a moment and try again, or switch to a different LLM provider."
    elif "not indexed" in error_msg.lower():
        error_msg = "This project has not been indexed yet."
        hint = "Run: devagent index"

    body = f"[red bold]{error_msg}[/red bold]"
    if hint:
        body += f"\n\n[yellow][HINT] {hint}[/yellow]"

    if _verbose:
        body += f"\n\n[dim]{traceback.format_exc()}[/dim]"

    console.print(
        Panel(body, title="FAIL Error", border_style="red")
    )
    raise typer.Exit(1)

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {"model": "qwen2.5-coder:7b", "base_url": "http://localhost:11434"},
    "groq": {"model": "llama-3.3-70b-versatile"},
    "anthropic": {"model": "claude-3-5-haiku-20241022"},
    "openai": {"model": "gpt-4o-mini"},
    "gemini": {"model": "gemini-1.5-flash"},
}

PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    if not key or len(key) <= 4:
        return key
    return "*" * (len(key) - 4) + key[-4:]


_OLLAMA_CLOUD_HOST = "https://ollama.com"


def _fetch_ollama_models(base_url: str, api_key: str = "") -> list[str]:
    """Return model names from an Ollama endpoint; empty list on failure."""
    try:
        from devagent.core.config import LLMConfig as _LLMConfig
        from devagent.core.llm import LLMClient as _LLMClient
        cfg = _LLMConfig(provider="ollama", model="", base_url=base_url, api_key=api_key)
        return _LLMClient(cfg).list_ollama_models()
    except Exception:
        return []


def _validate_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Validate Ollama is running and the model is available.

    For Ollama Cloud (base_url == https://ollama.com) we skip the local
    reachability check — the API key is validated separately.
    """
    if base_url == _OLLAMA_CLOUD_HOST:
        return True, f"Ollama Cloud — model {model!r} will be verified on first use"

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        available = [m.get("name", "") for m in data.get("models", [])]
        # Check if the requested model is available (may include :latest suffix)
        for m in available:
            if m == model or m.startswith(f"{model}:") or model.startswith(m.split(":")[0]):
                return True, f"Model '{model}' available on Ollama"
        # Model not found but Ollama is running
        names = ", ".join(available[:5]) if available else "none"
        return False, (
            f"Ollama is running but model '{model}' not found. "
            f"Available: {names}. Pull it with: ollama pull {model}"
        )
    except httpx.ConnectError:
        return False, (
            f"Ollama is not reachable at {base_url}. "
            "Start it with: ollama serve"
        )
    except Exception as exc:
        return False, f"Ollama check failed: {exc}"


def _validate_github_token(token: str) -> tuple[bool, str]:
    """Validate a GitHub personal access token."""
    if not token:
        return False, "No token configured"
    try:
        resp = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            username = resp.json().get("login", "unknown")
            return True, f"Authenticated as {username}"
        return False, f"GitHub returned status {resp.status_code}"
    except Exception as exc:
        return False, f"GitHub check failed: {exc}"


def _validate_brave_key(api_key: str) -> tuple[bool, str]:
    """Validate a Brave Search API key."""
    if not api_key:
        return False, "Not configured (optional)"
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": "test", "count": 1},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, "API key valid"
        return False, f"Brave returned status {resp.status_code}"
    except Exception as exc:
        return False, f"Brave check failed: {exc}"


def _validate_searchx_key(api_key: str) -> tuple[bool, str]:
    """Validate a SearchX API key."""
    if not api_key:
        return False, "Not configured (optional)"
    try:
        resp = httpx.get(
            "https://searchx.dev/api/v1/search",
            params={"q": "test", "count": 1},
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, "API key valid"
        return False, f"SearchX returned status {resp.status_code}"
    except Exception as exc:
        return False, f"SearchX check failed: {exc}"


def _check_command(cmd: list[str]) -> tuple[bool, str]:
    """Check if a command is available and return its version."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, shell=True
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            return True, version
        return False, f"Command returned exit code {result.returncode}"
    except FileNotFoundError:
        return False, "Not found in PATH"
    except Exception as exc:
        return False, str(exc)


# ============================
# COMMANDS
# ============================


@app.command()
def init() -> None:
    """Interactive first-time setup - configure LLM provider, API keys, and preferences."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]Welcome to DevAgent![/bold cyan]\n\n"
            "This wizard will configure your LLM provider, API keys, and preferences.\n"
            "All settings are stored in your user config directory.",
            title="DevAgent Setup",
            border_style="cyan",
        )
    )
    console.print()

    # Load existing config if present
    existing = load_config() if config_exists() else None
    if existing:
        console.print(
            "[yellow]An existing configuration was found. "
            "Current values will be shown as defaults.[/yellow]\n"
        )

    # --- Step 1: LLM provider ---
    console.print("[bold]Step 1:[/bold] Choose your LLM provider\n")
    providers = ["ollama", "groq", "anthropic", "openai", "gemini"]
    for i, p in enumerate(providers, 1):
        default_tag = " [dim](default, local, free)[/dim]" if p == "ollama" else ""
        console.print(f"  {i}. {p}{default_tag}")
    console.print()

    default_choice = "1"
    if existing:
        try:
            default_choice = str(providers.index(existing.llm.provider) + 1)
        except ValueError:
            pass

    choice = Prompt.ask(
        "Select provider",
        choices=["1", "2", "3", "4", "5"],
        default=default_choice,
    )
    provider = providers[int(choice) - 1]

    # --- Step 2: Model name ---
    default_model = PROVIDER_DEFAULTS[provider]["model"]
    if existing and existing.llm.provider == provider:
        default_model = existing.llm.model or default_model

    model = Prompt.ask("Model name", default=default_model)

    # --- Step 3: Provider-specific config ---
    base_url = ""
    api_key = ""

    if provider == "ollama":
        # Detect whether the existing config is already cloud.
        _existing_is_cloud = (
            existing is not None
            and existing.llm.provider == "ollama"
            and (
                existing.llm.base_url == _OLLAMA_CLOUD_HOST
                or bool(existing.llm.api_key)
            )
        )
        console.print()
        console.print("  [dim]1. Local  — runs on your machine, fully private, free[/dim]")
        console.print("  [dim]2. Cloud  — Ollama Cloud (API key required, 15+ hosted models)[/dim]")
        console.print()
        ollama_mode = Prompt.ask(
            "Ollama mode",
            choices=["1", "2"],
            default="2" if _existing_is_cloud else "1",
        )
        if ollama_mode == "2":
            base_url = _OLLAMA_CLOUD_HOST
            existing_key = existing.llm.api_key if (existing and existing.llm.provider == "ollama") else ""
            if existing_key:
                console.print(f"  Current API key: {_mask_key(existing_key)}")
            api_key = Prompt.ask(
                "Ollama Cloud API key",
                password=True,
                default=existing_key if existing_key else "",
            )
            if not api_key:
                console.print("[yellow]Warning: no API key entered — requests will be rejected by Ollama Cloud.[/yellow]")

            # Fetch available cloud models and let the user pick by number.
            console.print()
            with console.status("[cyan]Fetching available Ollama Cloud models...[/cyan]"):
                cloud_models = _fetch_ollama_models(_OLLAMA_CLOUD_HOST, api_key) if api_key else []

            if cloud_models:
                console.print(f"[dim]Found {len(cloud_models)} available models:[/dim]\n")
                for i, m in enumerate(cloud_models, 1):
                    console.print(f"  [bold cyan]{i}[/bold cyan]. {m}")
                console.print()
                default_pick = "1"
                if existing and existing.llm.provider == "ollama" and existing.llm.model in cloud_models:
                    default_pick = str(cloud_models.index(existing.llm.model) + 1)
                pick = Prompt.ask(
                    "Pick a model (number) or type a model name",
                    default=default_pick,
                )
                if pick.isdigit() and 1 <= int(pick) <= len(cloud_models):
                    model = cloud_models[int(pick) - 1]
                else:
                    model = pick.strip() or cloud_models[0]
                console.print(f"[dim]Selected model: [bold]{model}[/bold][/dim]")
            else:
                # Fallback: manual entry when fetch fails or no key entered.
                fallback_default = "gpt-oss:20b"
                if existing and existing.llm.provider == "ollama" and existing.llm.model:
                    fallback_default = existing.llm.model
                console.print("[dim](Could not fetch model list — enter a model name manually)[/dim]")
                model = Prompt.ask("Model name", default=fallback_default)
        else:
            default_url = PROVIDER_DEFAULTS["ollama"]["base_url"]
            if existing and existing.llm.base_url and existing.llm.base_url != _OLLAMA_CLOUD_HOST:
                default_url = existing.llm.base_url
            base_url = Prompt.ask("Ollama base URL", default=default_url)
    else:
        existing_key = existing.llm.api_key if existing else ""
        if existing_key:
            console.print(f"  Current API key: {_mask_key(existing_key)}")
        api_key = Prompt.ask(
            f"{provider.capitalize()} API key",
            password=True,
            default=existing_key if existing_key else "",
        )

    # --- Step 4: GitHub token ---
    console.print()
    console.print("[bold]Step 2:[/bold] GitHub Personal Access Token")
    console.print(
        "  [dim]Create one at: https://github.com/settings/tokens[/dim]"
    )

    existing_gh_token = existing.github.token if existing else ""
    if existing_gh_token:
        console.print(f"  Current token: {_mask_key(existing_gh_token)}")

    gh_token = Prompt.ask(
        "GitHub token",
        password=True,
        default=existing_gh_token if existing_gh_token else "",
    )

    # --- Step 5: Search provider selection ---
    console.print()
    console.print("[bold]Step 3:[/bold] Choose your search provider")
    console.print("  [dim]SearchX is free (3K/day), Brave is paid[/dim]")
    
    search_providers = ["searchx", "brave"]
    for i, p in enumerate(search_providers, 1):
        default_tag = " [dim](default, free)[/dim]" if p == "searchx" else ""
        console.print(f"  {i}. {p}{default_tag}")
    console.print()

    default_search_choice = "1"
    if existing:
        try:
            default_search_choice = str(search_providers.index(existing.search_provider) + 1)
        except ValueError:
            pass

    search_choice = Prompt.ask(
        "Select search provider",
        choices=["1", "2"],
        default=default_search_choice,
    )
    search_provider = search_providers[int(search_choice) - 1]

    # --- Step 6: Search API key ---
    console.print()
    if search_provider == "brave":
        console.print("[bold]Step 4:[/bold] Brave Search API Key [dim](optional)[/dim]")
        console.print(
            "  [dim]Get one at: https://brave.com/search/api/[/dim]"
        )
        existing_key = existing.brave.api_key if existing else ""
        if existing_key:
            console.print(f"  Current key: {_mask_key(existing_key)}")
        
        search_key = ""
        if Confirm.ask("Configure Brave Search?", default=bool(existing_key)):
            search_key = Prompt.ask(
                "Brave API key",
                password=True,
                default=existing_key if existing_key else "",
            )
        brave_key = search_key
        searchx_key = existing.searchx.api_key if existing else ""
    else:
        console.print("[bold]Step 4:[/bold] SearchX API Key [dim](optional)[/dim]")
        console.print(
            "  [dim]Get one at: https://searchx.dev/ (3K/day free)[/dim]"
        )
        existing_key = existing.searchx.api_key if existing else ""
        if existing_key:
            console.print(f"  Current key: {_mask_key(existing_key)}")
        
        searchx_key = ""
        if Confirm.ask("Configure SearchX?", default=bool(existing_key)):
            searchx_key = Prompt.ask(
                "SearchX API key",
                password=True,
                default=existing_key if existing_key else "",
            )
        brave_key = existing.brave.api_key if existing else ""

    # --- Step 6: Validate LLM provider ---
    console.print()
    with console.status("[cyan]Validating LLM provider...[/cyan]"):
        if provider == "ollama":
            ok, msg = _validate_ollama(base_url, model)
            # Cloud mode: also require a key
            if ok and base_url == _OLLAMA_CLOUD_HOST and not api_key:
                ok = False
                msg = "Ollama Cloud requires an API key — get one at ollama.com/settings/api-keys"
        else:
            # For cloud providers, just check that the key is non-empty
            ok = bool(api_key)
            msg = "API key provided" if ok else "No API key provided"

    if ok:
        console.print(f"  OK LLM provider: {msg}")
    else:
        console.print(f"  FAIL LLM provider: {msg}")
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(1)

    # --- Step 7: Validate search provider ---
    if search_provider == "brave" and brave_key:
        with console.status("[cyan]Validating Brave Search...[/cyan]"):
            ok, msg = _validate_brave_key(brave_key)
        if ok:
            console.print(f"  OK Brave Search: {msg}")
        else:
            console.print(f"  FAIL Brave Search: {msg}")
    elif search_provider == "searchx" and searchx_key:
        with console.status("[cyan]Validating SearchX...[/cyan]"):
            ok, msg = _validate_searchx_key(searchx_key)
        if ok:
            console.print(f"  OK SearchX: {msg}")
        else:
            console.print(f"  FAIL SearchX: {msg}")
    else:
        console.print(f"  WARN {search_provider.capitalize()} API key not provided - search will not work")

    # --- Step 8: Validate GitHub token ---
    if gh_token:
        with console.status("[cyan]Validating GitHub token...[/cyan]"):
            ok, msg = _validate_github_token(gh_token)
        if ok:
            console.print(f"  OK GitHub: {msg}")
        else:
            console.print(f"  FAIL GitHub: {msg}")
    else:
        console.print("  WARN GitHub token not provided - issue fetching will not work")

    # --- Step 9: Build and save config ---
    config = DevAgentConfig(
        llm=LLMConfig(
            provider=provider,
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=0.1,
            api_key=api_key,
            fallback=existing.llm.fallback if existing else None,
        ),
        github=GitHubConfig(
            token=gh_token,
            default_repo=existing.github.default_repo if existing else "",
        ),
        brave=BraveConfig(api_key=brave_key),
        searchx=SearchXConfig(api_key=searchx_key),
        search_provider=search_provider,
        output=OutputConfig(
            verbosity=existing.output.verbosity if existing else "normal",
        ),
    )

    save_config(config)

    console.print()
    console.print(
        Panel(
            f"[green bold]Configuration saved![/green bold]\n\n"
            f"Config file: [dim]{get_config_path()}[/dim]\n"
            f"Provider: [cyan]{provider}[/cyan]\n"
            f"Model: [cyan]{model}[/cyan]",
            title="DevAgent Setup Complete",
            border_style="green",
        )
    )


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Display current configuration"),
    set_value: str | None = typer.Option(
        None, "--set", help="Update a config value (key=value, supports dot notation)"
    ),
) -> None:
    """Show or update DevAgent configuration."""
    if not show and set_value is None:
        console.print("[yellow]Use --show to display config or --set key=value to update.[/yellow]")
        raise typer.Exit()

    if show:
        if not config_exists():
            console.print(
                Panel(
                    "No configuration found. Run [bold cyan]devagent init[/bold cyan] first.",
                    title="FAIL No Config",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        cfg = load_config()
        table = Table(title="DevAgent Configuration", border_style="cyan")
        table.add_column("Setting", style="bold")
        table.add_column("Value")

        table.add_row("llm.provider", cfg.llm.provider)
        table.add_row("llm.model", cfg.llm.model)
        table.add_row("llm.base_url", cfg.llm.base_url)
        table.add_row("llm.temperature", str(cfg.llm.temperature))
        table.add_row("llm.api_key", _mask_key(cfg.llm.api_key) if cfg.llm.api_key else "")
        if cfg.llm.fallback:
            table.add_row("llm.fallback.provider", cfg.llm.fallback.provider)
            table.add_row("llm.fallback.model", cfg.llm.fallback.model)
        table.add_row("github.token", _mask_key(cfg.github.token) if cfg.github.token else "")
        table.add_row("github.default_repo", cfg.github.default_repo or "")
        table.add_row("search_provider", cfg.search_provider)
        table.add_row("brave.api_key", _mask_key(cfg.brave.api_key) if cfg.brave.api_key else "")
        table.add_row("searchx.api_key", _mask_key(cfg.searchx.api_key) if cfg.searchx.api_key else "")
        table.add_row("output.verbosity", cfg.output.verbosity)

        console.print()
        console.print(table)
        console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")
        return

    if set_value:
        if "=" not in set_value:
            console.print("[red]Invalid format. Use: --set key=value[/red]")
            raise typer.Exit(1)

        key, value = set_value.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not config_exists():
            console.print(
                "[red]No configuration found. Run [bold]devagent init[/bold] first.[/red]"
            )
            raise typer.Exit(1)

        cfg = load_config()

        # Support dot notation for setting nested values
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                console.print(f"[red]Unknown config key: {key}[/red]")
                raise typer.Exit(1)

        final_key = parts[-1]
        if not hasattr(obj, final_key):
            console.print(f"[red]Unknown config key: {key}[/red]")
            raise typer.Exit(1)

        # Validate specific keys before setting
        valid_providers = ["ollama", "groq", "anthropic", "openai", "gemini"]
        valid_search = ["brave", "searchx"]

        if key == "llm.provider" and value not in valid_providers:
            console.print(f"[red]Invalid provider '{value}'. Must be one of: {', '.join(valid_providers)}[/red]")
            raise typer.Exit(1)

        if key == "search_provider" and value not in valid_search:
            console.print(f"[red]Invalid search provider '{value}'. Must be one of: {', '.join(valid_search)}[/red]")
            raise typer.Exit(1)

        if key == "llm.temperature":
            try:
                temp = float(value)
                if not (0.0 <= temp <= 2.0):
                    console.print("[red]Temperature must be between 0.0 and 2.0[/red]")
                    raise typer.Exit(1)
            except ValueError:
                console.print("[red]Temperature must be a number[/red]")
                raise typer.Exit(1)

        # Type coercion
        current = getattr(obj, final_key)
        if isinstance(current, float):
            value = float(value)  # type: ignore[assignment]
        elif isinstance(current, int):
            value = int(value)  # type: ignore[assignment]

        setattr(obj, final_key, value)
        save_config(cfg)
        console.print(f"[green]Set {key} = {value}[/green]")


@app.command()
def index(
    full: bool = typer.Option(False, "--full", help="Force full re-index"),
    status: bool = typer.Option(False, "--status", help="Show index state for current project"),
    clear: bool = typer.Option(False, "--clear", help="Wipe index for current project"),
) -> None:
    """Index the current project's codebase for semantic search."""
    import asyncio

    from devagent.core.project import detect_project_root
    from devagent.core.storage import (
        clear_project_index,
        ensure_dirs,
        get_chroma_dir,
        get_sqlite_path,
    )

    # Detect project root
    project_root, found_marker = detect_project_root()
    if not found_marker:
        console.print(
            "[yellow]WARN No project marker found (.git, pyproject.toml, etc.). "
            f"Using current directory: {project_root}[/yellow]\n"
        )
    else:
        console.print(f"[dim]Project root: {project_root}[/dim]\n")

    ensure_dirs(project_root)
    sqlite_path = get_sqlite_path(project_root)

    # --- --status flag ---
    if status:
        from devagent.codeprism.client import CodePrismClient

        codeprism = CodePrismClient(str(project_root))
        if not codeprism.is_indexed:
            console.print(
                Panel(
                    "This project has not been indexed yet.\n"
                    "Run [bold cyan]devagent index[/bold cyan] to create the index.",
                    title="Index Status",
                    border_style="yellow",
                )
            )
            return
        console.print(
            Panel(
                "CodePrism index is available for this project.\n"
                "Graph-powered tools will be enabled in new sessions.",
                title="Index Status",
                border_style="green",
            )
        )
        return

    # --- --clear flag ---
    if clear:
        chroma_dir = get_chroma_dir(project_root)
        if not sqlite_path.exists() and not chroma_dir.exists():
            console.print("[yellow]No index found for this project.[/yellow]")
            return

        if not Confirm.ask(
            "[red]This will delete the entire index for this project. Continue?[/red]",
            default=False,
        ):
            console.print("[dim]Cancelled.[/dim]")
            return

        clear_project_index(project_root)
        console.print("[green]Index cleared successfully.[/green]")
        return

    # --- Default: run indexing ---
    console.print(
        Panel(
            f"[bold cyan]Indexing project...[/bold cyan]\n"
            f"Mode: [cyan]{'incremental' if not full else 'full rebuild'}[/cyan]\n"
            f"Project: [dim]{project_root}[/dim]",
            title="DevAgent Indexer",
            border_style="cyan",
        )
    )

    async def _run_index() -> dict:
        """Run the installed CodePrism indexer."""
        from codeprism import CodePrism, CodePrismConfig

        prism = CodePrism(
            str(project_root.resolve()),
            CodePrismConfig(parse_workers=1),
        )
        try:
            await prism.initialize()
            result = await prism.index()
            return {
                "files_indexed": result.file_count,
                "symbols_indexed": result.symbol_count,
                "edges_created": result.edge_count,
                "files_skipped": result.files_skipped,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
            }
        finally:
            await prism.close()

    try:
        with console.status("[cyan]Indexing codebase (this may take a moment on first run)...[/cyan]"):
            result = asyncio.run(_run_index())

        # Show summary
        table = Table(border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Files indexed", str(result.get("files_indexed", 0)))
        table.add_row("Symbols indexed", str(result.get("symbols_indexed", 0)))
        table.add_row("Edges created", str(result.get("edges_created", 0)))
        table.add_row("Files skipped (unchanged)", str(result.get("files_skipped", 0)))
        table.add_row("Duration", f"{result.get('duration_seconds', 0):.1f}s")

        console.print()
        console.print(table)

        if result.get("errors"):
            console.print("[yellow]Some files could not be indexed:[/yellow]")
            for error in result["errors"]:
                console.print(f"[yellow]- {error}[/yellow]")

        chroma_dir = get_chroma_dir(project_root)
        console.print(f"\n[dim]Index stored at: {chroma_dir.parent}[/dim]")
        console.print("[green]Indexing complete.[/green]")

    except Exception as exc:
        console.print(
            Panel(
                f"[red bold]Indexing failed[/red bold]\n\n{exc}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(1)


@app.command()
def analyze(
    issue: int | None = typer.Option(None, "--issue", "-i", help="GitHub issue number to analyze"),
    spec: str | None = typer.Option(None, "--spec", "-s", help="Path to a spec file (markdown/text)"),
    text: str | None = typer.Option(None, "--text", "-t", help="Inline spec text to analyze"),
    url: str | None = typer.Option(None, "--url", "-u", help="Full GitHub issue or PR URL"),
    output: str | None = typer.Option(
        "both", "--output", "-o",
        help="Output format: terminal, markdown, both, json",
    ),
    repo: str | None = typer.Option(
        None, "--repo", "-r", help="GitHub repo (owner/repo) - overrides default"
    ),
    chat: bool = typer.Option(False, "--chat", "-c", help="Drop into chat session after analysis"),
) -> None:
    """Run gap analysis / implement a spec against the current codebase."""
    console.print(
        "[yellow]devagent analyze[/yellow] is being rebuilt as part of the "
        "agent harness (Phase 1). Use [bold]devagent[/bold] (the interactive "
        "session) once Phase 1 is complete.\n\n"
        "[dim]Track progress: plans/DEVAGENT_ROADMAP.md[/dim]"
    )
    raise typer.Exit(0)

    if not config_exists():  # preserved for reference — unreachable after Exit above
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    config = load_config()
    project_root, _ = detect_project_root()
    project_name = project_root.name

    # Check if indexed
    sqlite_path = get_sqlite_path(project_root)
    idx_status = asyncio.run(get_index_status(sqlite_path))
    if not idx_status["exists"]:
        console.print("[yellow]WARN This project has not been indexed yet.[/yellow]")
        if Confirm.ask("Do you want to index it now?"):
            index(full=False, status=False, clear=False)
        else:
            console.print("[red]Analysis requires an indexed project. Exiting.[/red]")
            raise typer.Exit(1)

    # F4: Handle --url flag
    resource_type = "issue"
    if url is not None:
        # Cannot combine --url with --issue, --repo
        if issue is not None:
            console.print("[red]Cannot use --url together with --issue. Use one or the other.[/red]")
            raise typer.Exit(1)
        if repo is not None:
            console.print("[red]Cannot use --url together with --repo. Use one or the other.[/red]")
            raise typer.Exit(1)

        # Parse the URL
        try:
            parsed_url = parse_github_url(url)
        except InvalidGitHubURLError as e:
            console.print(f"[red]Invalid GitHub URL:[/red] {e.reason}")
            console.print("[dim]Expected format: https://github.com/owner/repo/issues/NUMBER[/dim]")
            raise typer.Exit(1)

        # Set issue and repo from parsed URL
        issue = parsed_url.number
        repo = format_repo_string(parsed_url)
        resource_type = parsed_url.resource_type

        # If it's a PR URL, show a note
        if resource_type == "pull_request":
            console.print(
                f"[dim]Note: Analyzing PR #{parsed_url.number} as a spec - "
                f"extracting intent from PR description and title.[/dim]"
            )

    # Determine spec source and text
    spec_source = ""
    spec_text = ""

    if issue:
        if not config.github.token:
            console.print("[red]GitHub token required to fetch issues/PRs. Run [bold]devagent init[/bold][/red]")
            raise typer.Exit(1)
        repo_name = repo or config.github.default_repo
        if not repo_name:
            console.print("[red]GitHub repo required. Pass --repo owner/repo or set default in config.[/red]")
            raise typer.Exit(1)
        
        prefix = "pull" if resource_type == "pull_request" else "issues"
        spec_source = f"https://github.com/{repo_name}/{prefix}/{issue}"
        
        # Quick MCP client just for github
        async def fetch_github_resource():
            async with MCPManager(config, project_root) as mcp:
                if not mcp.github:
                    raise RuntimeError("GitHub MCP not available")
                owner, r = repo_name.split("/")
                if resource_type == "pull_request":
                    return await mcp.github.get_pull_request(owner, r, issue)
                return await mcp.github.get_issue(owner, r, issue)
                
        with console.status(f"[cyan]Fetching {resource_type.replace('_', ' ')} #{issue}...[/cyan]"):
            try:
                resource_data = asyncio.run(fetch_github_resource())
                spec_text = f"Title: {resource_data.get('title', '')}\n\nBody:\n{resource_data.get('body', '')}"
            except Exception as e:
                console.print(f"[red]Failed to fetch {resource_type.replace('_', ' ')}: {e}[/red]")
                raise typer.Exit(1)
                
    elif spec:
        spec_path = Path(spec)
        if not spec_path.exists():
            console.print(f"[red]Spec file not found: {spec}[/red]")
            raise typer.Exit(1)
        spec_source = spec_path.name
        spec_text = spec_path.read_text(encoding="utf-8")
        
    elif text:
        spec_source = "Inline Text"
        spec_text = text
        
    else:
        console.print("[red]Must provide one of: --issue, --spec, --text, or --url[/red]")
        raise typer.Exit(1)

    # Run the pipeline
    async def run_analysis():
        async with MCPManager(config, project_root) as mcp:
            return await run_pipeline(config, mcp, spec_text, spec_source, str(project_root))

    try:
        report = asyncio.run(run_analysis())
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc)
        return  # unreachable but satisfies type checker

    # Handle output
    if output == "json":
        print(report.model_dump_json(indent=2))
        return

    md_path = None
    if output in ("markdown", "both"):
        md_path = generate_markdown_report(report, project_root, project_name)
        if output == "markdown":
            console.print(f"[green]Markdown report saved to: [cyan]{md_path}[/cyan][/green]")
            return

    if output in ("terminal", "both"):
        render_gap_report(report, project_name, md_path)

    if chat:
        from devagent.chat.session import ChatSession
        session = ChatSession(report, config, project_name)
        asyncio.run(session.run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic search query"),
) -> None:
    """Semantic search across the indexed codebase."""
    import asyncio

    from devagent.core.project import detect_project_root
    
    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    project_root, _ = detect_project_root()
    
    async def do_search():
        from devagent.mcp.manager import MCPManager
        
        async with MCPManager(load_config(), project_root) as mcp:
            if not mcp.code_search:
                raise RuntimeError("CodeSearch MCP not available")
            return await mcp.code_search.semantic_search(query, top_k=5)
            
    with console.status("[cyan]Searching codebase...[/cyan]"):
        try:
            results = asyncio.run(do_search())
        except typer.Exit:
            raise
        except Exception as exc:
            _handle_error(exc)
            return
            
    if not results:
        console.print("[yellow]No relevant matches found.[/yellow]")
        return
        
    table = Table(title=f"Search Results for: '{query}'", border_style="cyan")
    table.add_column("File", style="cyan")
    table.add_column("Type / Name", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Snippet")
    
    for r in results:
        lines = r.content.strip().splitlines()
        first_line = lines[0].strip() if lines else ""
        if len(first_line) > 50:
            first_line = first_line[:47] + "..."
            
        score_pct = f"{r.similarity_score * 100:.1f}%"
        table.add_row(r.file_path, f"{r.chunk_type}: {r.name}", score_pct, first_line)
        
    console.print()
    console.print(table)


@app.command()
def reports(
    show: str | None = typer.Option(
        None, "--show", help="Re-display a saved report by filename or partial match"
    ),
) -> None:
    """List or display saved analysis reports for the current project."""
    from rich.markdown import Markdown

    from devagent.core.project import detect_project_root
    from devagent.core.storage import get_reports_dir

    project_root, _ = detect_project_root()
    reports_dir = get_reports_dir(project_root)

    if not reports_dir.exists() or not any(reports_dir.iterdir()):
        console.print(
            Panel(
                "No reports found for this project.\n"
                "Run [bold cyan]devagent analyze[/bold cyan] to generate one.",
                title="Reports",
                border_style="yellow",
            )
        )
        return

    report_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not report_files:
        console.print("[yellow]No markdown reports found in the reports directory.[/yellow]")
        return

    # --- --show flag ---
    if show:
        # Find the report by exact filename or partial match
        match = None
        for rf in report_files:
            if rf.name == show or rf.stem == show or show in rf.name:
                match = rf
                break

        if not match:
            console.print(f"[red]No report matching '{show}' found.[/red]")
            console.print("[dim]Available reports:[/dim]")
            for rf in report_files:
                console.print(f"  - {rf.stem}")
            return

        content = match.read_text(encoding="utf-8")
        console.print()
        console.print(Panel(f"[dim]{match.name}[/dim]", title="Report", border_style="cyan"))
        console.print(Markdown(content))
        return

    # --- Default: list all reports ---
    table = Table(title="Saved Reports", border_style="cyan")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Report", style="cyan")
    table.add_column("Source")
    table.add_column("Date")
    table.add_column("Reqs", justify="right")
    table.add_column("Conflicts", justify="right", style="red")

    for i, rf in enumerate(report_files, 1):
        content = rf.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Parse header for source and date
        source = ""
        date = ""
        req_count = 0
        conflict_count = 0

        for line in lines[:10]:
            if line.startswith("**Source:**"):
                source = line.replace("**Source:**", "").strip().rstrip(" ")
            elif line.startswith("**Generated:**"):
                date = line.replace("**Generated:**", "").strip().rstrip(" ")

        # Count requirements (lines starting with ####)
        req_count = content.count("#### REQ-")
        # Count conflicts (lines containing CONFLICT)
        conflict_count = content.count("ALERT CONFLICT")

        # Truncate source for table display
        if len(source) > 40:
            source = source[:37] + "..."

        table.add_row(str(i), rf.stem, source, date, str(req_count), str(conflict_count))

    console.print()
    console.print(table)
    console.print(f"\n[dim]Reports directory: {reports_dir}[/dim]")
    console.print("[dim]Use [bold]devagent reports --show <name>[/bold] to view a report.[/dim]")


@app.command()
def chat(
    report_name: str | None = typer.Option(
        None, "--report", "-r", help="Name of the saved report to chat about"
    ),
) -> None:
    """Start an interactive chat session about a saved gap report."""
    from devagent.core.project import detect_project_root
    from devagent.core.storage import get_reports_dir

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    project_root, _ = detect_project_root()
    reports_dir = get_reports_dir(project_root)

    if not reports_dir.exists() or not any(reports_dir.iterdir()):
        console.print("[yellow]No reports found. Run devagent analyze first.[/yellow]")
        raise typer.Exit(1)

    report_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    target_file = None
    if report_name:
        for rf in report_files:
            if rf.name == report_name or rf.stem == report_name or report_name in rf.name:
                target_file = rf
                break
        if not target_file:
            console.print(f"[red]Report '{report_name}' not found.[/red]")
            raise typer.Exit(1)
    else:
        # Prompt user to choose
        console.print("\n[bold]Available Reports:[/bold]")
        for i, rf in enumerate(report_files, 1):
            console.print(f"  {i}. {rf.stem}")
        
        choice = Prompt.ask("\nSelect a report to chat about (number)", default="1")
        try:
            target_file = report_files[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection.[/red]")
            raise typer.Exit(1)

    report_content = target_file.read_text(encoding="utf-8")
    console.print(f"\n[dim]Loaded report: [bold]{target_file.stem}[/bold][/dim]")
    console.print("[dim]Starting chat session — ask anything about this report.[/dim]\n")

    from devagent.agent.flows import DevAgentSession
    from devagent.core.config import load_config

    cfg = load_config()
    session = DevAgentSession(cfg, project_root=str(project_root))
    seed = (
        f"I have loaded the following gap analysis report for your reference:\n\n"
        f"<report>\n{report_content}\n</report>\n\n"
        f"Please confirm you have read it and are ready to answer questions about it."
    )
    session.interactive_repl(first_message=seed)


@app.command()
def watch(
    repo: str | None = typer.Option(None, "--repo", "-r", help="GitHub repo as owner/repo"),
    status: bool = typer.Option(False, "--status", help="Run a check right now and show results"),
    start: bool = typer.Option(False, "--start", help="Start background scheduler (foreground process)"),
    stop: bool = typer.Option(False, "--stop", help="Stop watching a repo"),
    list_repos: bool = typer.Option(False, "--list", help="List all watched repos"),
    report: bool = typer.Option(False, "--report", help="Show all analysed issues for a repo"),
    show: int | None = typer.Option(None, "--show", help="Show full analysis for a specific issue number"),
    interval: str = typer.Option("30m", "--interval", help="Check interval: 30m, 1h, 6h, 12h, 24h"),
    labels: str | None = typer.Option(None, "--labels", help="Comma-separated label filters e.g. 'feature,enhancement'"),
) -> None:
    """Monitor a GitHub repo for new issues and detect cross-issue conflicts."""
    import asyncio

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    from devagent.core.project import detect_project_root
    project_root, _ = detect_project_root()

    interval_minutes = _parse_interval(interval)

    if list_repos:
        asyncio.run(_watch_list())
        return

    if stop:
        if not repo:
            console.print("[red]Specify --repo owner/repo to stop watching.[/red]")
            raise typer.Exit(1)
        asyncio.run(_watch_stop(repo))
        return

    if show is not None:
        owner, repo_name = _parse_repo_or_infer(repo, project_root)
        asyncio.run(_watch_show(owner, repo_name, show, cfg, project_root))
        return

    if report:
        owner, repo_name = _parse_repo_or_infer(repo, project_root)
        asyncio.run(_watch_report(owner, repo_name))
        return

    if status:
        owner, repo_name = _parse_repo_or_infer(repo, project_root)
        asyncio.run(_watch_run_once(owner, repo_name, cfg, project_root))
        return

    if start:
        asyncio.run(_watch_start(cfg, project_root, interval_minutes))
        return

    if repo:
        label_list = [lbl.strip() for lbl in labels.split(",")] if labels else []
        asyncio.run(_watch_register(repo, interval_minutes, label_list))
        return

    # No flags - show usage hint
    console.print(
        "[dim]Usage: devagent watch --repo owner/repo    (to start watching)\n"
        "       devagent watch --status              (to check now)\n"
        "       devagent watch --list                (to list watched repos)\n"
        "       devagent watch --help                (for all options)[/dim]"
    )


# ---------------------------------------------------------------------------
# Watch command helpers
# ---------------------------------------------------------------------------

def _parse_interval(interval_str: str) -> int:
    """Converts interval string to minutes. Supports: 30m, 1h, 6h, 12h, 24h."""
    s = interval_str.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    elif s.endswith("h"):
        return int(s[:-1]) * 60
    else:
        return int(s)


def _split_repo_string(repo_str: str) -> tuple[str, str]:
    """Splits 'owner/repo' into ('owner', 'repo')."""
    parts = repo_str.strip().split("/")
    if len(parts) != 2:
        console.print(f"[red]Invalid repo format: '{repo_str}'. Use 'owner/repo'.[/red]")
        raise typer.Exit(1)
    return parts[0], parts[1]


def _parse_repo_or_infer(repo: str | None, project_root) -> tuple[str, str]:
    """Returns (owner, repo_name). If repo not given, infers from git remote."""
    if repo:
        return _split_repo_string(repo)

    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        remote_url = result.stdout.strip()
        if "github.com" in remote_url:
            from devagent.core.url_parser import parse_github_url
            if remote_url.startswith("git@"):
                remote_url = remote_url.replace("git@github.com:", "https://github.com/")
            remote_url = remote_url.removesuffix(".git")
            parsed = parse_github_url(remote_url + "/issues/1")
            return parsed.owner, parsed.repo
    except Exception:
        pass

    console.print(
        "[red]Could not infer GitHub repo from git remote. "
        "Use --repo owner/repo explicitly.[/red]"
    )
    raise typer.Exit(1)


async def _watch_register(repo_str: str, interval_minutes: int, labels: list[str]) -> None:
    from devagent.watcher.storage import init_watcher_db, register_repo
    owner, repo_name = _split_repo_string(repo_str)
    await init_watcher_db()
    await register_repo(owner, repo_name, interval_minutes, labels)
    console.print(
        f"[green]OK[/green] Now watching [bold]{owner}/{repo_name}[/bold]  "
        f"-  checking every {interval_minutes} minutes"
    )
    console.print("[dim]Run 'devagent watch --status' to run the first check now.[/dim]")


async def _watch_run_once(owner: str, repo: str, cfg, project_root) -> None:
    from devagent.watcher.scheduler import WatcherScheduler
    from devagent.watcher.storage import get_watched_repo, init_watcher_db
    await init_watcher_db()
    watched_repo = await get_watched_repo(owner, repo)
    if not watched_repo:
        console.print(f"[red]{owner}/{repo} is not being watched.[/red]")
        console.print(f"[dim]Run 'devagent watch --repo {owner}/{repo}' to register it.[/dim]")
        return
    scheduler = WatcherScheduler(cfg, project_root, watched_repo.check_interval_minutes)
    await scheduler.run_once()


async def _watch_start(cfg, project_root, interval_minutes: int) -> None:
    from devagent.watcher.scheduler import WatcherScheduler
    from devagent.watcher.storage import init_watcher_db
    await init_watcher_db()
    scheduler = WatcherScheduler(cfg, project_root, interval_minutes)
    await scheduler.start()


async def _watch_list() -> None:
    from devagent.output.watcher_renderer import render_watched_repos
    from devagent.watcher.storage import init_watcher_db, list_watched_repos
    await init_watcher_db()
    repos = await list_watched_repos()
    render_watched_repos(repos)


async def _watch_stop(repo_str: str) -> None:
    from devagent.watcher.storage import deactivate_repo, init_watcher_db
    owner, repo_name = _split_repo_string(repo_str)
    await init_watcher_db()
    await deactivate_repo(owner, repo_name)
    console.print(f"[green]OK[/green] Stopped watching [bold]{owner}/{repo_name}[/bold]")
    console.print("[dim]Historical analysis data is preserved.[/dim]")


async def _watch_report(owner: str, repo: str) -> None:
    from devagent.output.watcher_renderer import render_all_analyses
    from devagent.watcher.storage import get_all_analyses_for_repo, init_watcher_db
    await init_watcher_db()
    analyses = await get_all_analyses_for_repo(owner, repo)
    render_all_analyses(analyses, owner, repo)


async def _watch_show(owner: str, repo: str, issue_number: int, cfg, project_root) -> None:
    from devagent.core.storage import get_watcher_reports_dir
    from devagent.watcher.storage import get_analysis, init_watcher_db

    await init_watcher_db()
    analysis = await get_analysis(owner, repo, issue_number)
    if not analysis:
        console.print(f"[red]Issue #{issue_number} has not been analysed by the watcher yet.[/red]")
        console.print("[dim]Run 'devagent watch --status' to check for new issues first.[/dim]")
        return

    reports_dir = get_watcher_reports_dir(owner, repo)
    report_json = reports_dir / f"issue-{issue_number}-watcher.json"

    if report_json.exists() and analysis.full_report_available:
        from devagent.core.models import GapReport
        from devagent.output.terminal import render_gap_report
        gap_report = GapReport.model_validate_json(report_json.read_text(encoding="utf-8"))
        render_gap_report(gap_report, f"{owner}/{repo}", report_json)
    else:
        console.print(
            f"[dim]Generating full analysis for #{issue_number}... "
            f"(this may take 30-60 seconds)[/dim]"
        )
        console.print(
            "[yellow]Full pipeline re-run is not available yet.[/yellow] "
            "It will be restored once the agent harness (Phase 1) is complete."
        )
        return
        generate_markdown_report(gap_report, project_root, f"{owner}/{repo}")

    console.print()
    console.print(
        f"[dim]To discuss this analysis: "
        f"devagent chat --report issue-{issue_number}-watcher[/dim]"
    )


# ---------------------------------------------------------------------------
# session sub-app
# ---------------------------------------------------------------------------

session_app = typer.Typer(name="session", help="Manage agent sessions.", add_completion=False)
app.add_typer(session_app, name="session")


@session_app.command("list")
def session_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
) -> None:
    """List recent agent sessions."""
    import datetime

    from devagent.session.manager import SessionManager

    mgr = SessionManager()
    sessions = mgr.list(limit=limit)
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title="Agent Sessions", border_style="cyan")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Title")
    table.add_column("Model")
    table.add_column("Updated", style="dim")

    for s in sessions:
        updated = datetime.datetime.fromtimestamp(s["updated_at"], tz=datetime.UTC).strftime("%Y-%m-%d %H:%M")
        table.add_row(s["id"][:8], s["title"] or "(untitled)", s["model"], updated)

    console.print()
    console.print(table)


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)"),
) -> None:
    """Show events and token usage for a session."""

    from devagent.session.manager import SessionManager

    mgr = SessionManager()
    sessions = mgr.list(limit=200)
    match = next((s for s in sessions if s["id"].startswith(session_id)), None)
    if not match:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    full_id = match["id"]
    events = mgr.get_events(full_id)
    totals = mgr.get_token_totals(full_id)

    console.print()
    console.print(Panel(
        f"[bold]{match['title']}[/bold]\n"
        f"ID: [dim]{full_id}[/dim]\n"
        f"Model: {match['model']} ({match['provider']})\n"
        f"Tokens: {totals['tokens_in']:,} in / {totals['tokens_out']:,} out",
        title="Session",
        border_style="cyan",
    ))

    for ev in events:
        role = ev["role"]
        content = ev["content"]
        tool_calls = ev.get("tool_calls") or []
        if role == "user":
            console.print(f"\n[bold cyan]You:[/bold cyan] {content[:200]}")
        elif role == "assistant":
            prefix = "[bold green]Agent:[/bold green] "
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                console.print(f"{prefix}[dim](called: {names})[/dim]")
            if content:
                console.print(f"{prefix}{content[:200]}")
        elif role == "tool_result":
            console.print(f"  [dim]result[{ev.get('tool_name')}]: {content[:100]}[/dim]")


@session_app.command("delete")
def session_delete(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)"),
) -> None:
    """Delete a session and all its events."""
    from devagent.session.manager import SessionManager

    mgr = SessionManager()
    sessions = mgr.list(limit=200)
    match = next((s for s in sessions if s["id"].startswith(session_id)), None)
    if not match:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    if not Confirm.ask(f"Delete session '{match['title']}' ({match['id'][:8]})?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return

    mgr.delete(match["id"])
    console.print("[green]Session deleted.[/green]")


@session_app.command("compress")
def session_compress(
    session_id: str = typer.Argument(..., help="Session ID (or unique prefix)"),
    keep: int = typer.Option(20, "--keep", "-k", help="Hot-window size: keep last N events verbatim"),
) -> None:
    """Compress a session's history into a compact summary to free context window space."""
    from devagent.core.llm import LLMClient
    from devagent.session.compressor import compress_session
    from devagent.session.manager import SessionManager

    mgr = SessionManager()
    sessions = mgr.list(limit=200)
    match = next((s for s in sessions if s["id"].startswith(session_id)), None)
    if not match:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    sid = match["id"]
    cfg = load_config() if config_exists() else None
    from devagent.core.config import DevAgentConfig
    if cfg is None:
        cfg = DevAgentConfig()

    llm = LLMClient(cfg.llm)

    console.print(f"[cyan]Compressing session {sid[:8]}...[/cyan]")
    result = compress_session(sid, llm, keep_last_n=keep)

    if result is None:
        console.print("[dim]Nothing to compress — session is short enough already.[/dim]")
        return

    console.print(f"[green]Compressed {result.events_compressed} events.[/green]")
    console.print(f"[dim]{result.events_remaining} hot-window events kept verbatim.[/dim]")
    console.print(f"[dim]~{result.tokens_saved} tokens freed.[/dim]")
    console.print(f"[dim]Summary length: {len(result.summary)} chars.[/dim]")


# ---------------------------------------------------------------------------
# Primary agent run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    resume: str | None = typer.Option(
        None, "--resume", "-r", help="Resume a previous session by ID prefix"
    ),
    project: str | None = typer.Option(
        None, "--project", "-p", help="Project path (defaults to current directory)"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override model (e.g. qwen2.5-coder:7b)"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Token budget for this session"
    ),
    no_limit: bool = typer.Option(
        False, "--no-limit", help="Remove the iteration cap (use with caution)"
    ),
    plan: bool = typer.Option(
        False, "--plan", help="Generate and approve a plan before each new task"
    ),
    allow: list[str] = typer.Option(  # noqa: B008
        [], "--allow", help="Auto-approve tool calls matching this rule (e.g. 'write_file:src/**')"
    ),
    deny: list[str] = typer.Option(  # noqa: B008
        [], "--deny", help="Auto-deny tool calls matching this rule (e.g. 'run_shell')"
    ),
    interactive_approval: bool = typer.Option(
        False, "--interactive-approval", help="Pause and ask before every unmatched tool call"
    ),
    effort: str | None = typer.Option(
        None, "--effort", help="Effort level: low|medium|high|xhigh|max"
    ),
    bare: bool = typer.Option(
        False, "--bare", help="Skip DEVAGENT.md, memory injection, CodePrism overlay, and permission gate"
    ),
    allow_tools: str | None = typer.Option(
        None, "--allow-tools",
        help="Comma-separated tool names to auto-approve (e.g. 'run_shell,write_file')"
    ),
    permission_mode: str = typer.Option(
        "default", "--permission-mode",
        help="Permission shorthand: default|accept-edits|read-only|auto|yolo",
    ),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Name this session (also used to resume by name)"
    ),
    add_dir: list[str] = typer.Option(  # noqa: B008
        [], "--add-dir", help="Extra directory the agent may read/write (repeatable)"
    ),
    diff_preview: bool = typer.Option(
        False, "--diff-preview", help="Show a diff and ask for confirmation before each file write/edit"
    ),
) -> None:
    """Start an interactive agent session (the main DevAgent command)."""
    from devagent.agent.flows import DevAgentSession
    from devagent.core.project import detect_project_root

    project_root, _ = detect_project_root(Path(project) if project else None)

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config(str(project_root))
    if model:
        cfg.llm.model = model
    tools_list = [t.strip() for t in allow_tools.split(",")] if allow_tools else None

    selected_resume = resume
    if selected_resume is None and sys.stdin.isatty() and sys.stdout.isatty():
        from devagent.session.manager import SessionManager

        sessions = SessionManager().list_by_project(str(project_root), limit=10)
        if sessions:
            console.print("[bold]Recent sessions[/bold]")
            for index, item in enumerate(sessions, 1):
                title = item.get("title") or "Untitled session"
                console.print(f"  [cyan]{index}[/cyan]. {title} ({item['id'][:8]})")
            new_index = len(sessions) + 1
            console.print(f"  [cyan]{new_index}[/cyan]. Start new session")
            choice = console.input("Select a session [default: new]: ").strip()
            if choice and choice.isdigit():
                selected = int(choice)
                if 1 <= selected <= len(sessions):
                    selected_resume = sessions[selected - 1]["id"]

    try:
        session = DevAgentSession(
            cfg,
            project_root,
            max_tokens=max_tokens,
            resume_id=selected_resume,
            no_limit=no_limit,
            plan_mode=plan,
            allow=list(allow),
            deny=list(deny),
            interactive_approval=interactive_approval,
            effort=effort,
            bare=bare,
            allow_tools=tools_list,
            permission_mode=permission_mode,
            name=name,
            extra_dirs=list(add_dir) or None,
            diff_preview=diff_preview,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        _handle_error(exc)
        return

    if selected_resume:
        console.print(f"[dim]Resuming session: {session.session_id[:8]}[/dim]")
        session.show_history()
    else:
        _name_suffix = f"  ({name})" if name else ""
        console.print(f"[dim]New session: {session.session_id[:8]}{_name_suffix}[/dim]")

    session.print_header("DevAgent")

    try:
        session.interactive_repl()
    except Exception as exc:
        _handle_error(exc)


# ---------------------------------------------------------------------------
# Phase 7.1 — Quick action mode
# ---------------------------------------------------------------------------

def _validate_against_schema(text: str, schema: dict) -> tuple[bool, str]:
    """Parse text as JSON and validate against schema. Returns (valid, error_message)."""
    try:
        import json as _json

        import jsonschema
    except ImportError:
        return True, ""

    try:
        data = _json.loads(text)
    except _json.JSONDecodeError as exc:
        return False, f"Response is not valid JSON: {exc}"

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, ""
    except jsonschema.ValidationError as exc:
        return False, exc.message
    except jsonschema.SchemaError as exc:
        return False, f"Invalid schema: {exc.message}"


@app.command()
def do(
    task: str = typer.Argument(..., help="Task for the agent to perform"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override model"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Token budget"),
    no_session: bool = typer.Option(False, "--no-session", help="Skip persisting session to SQLite"),
    no_limit: bool = typer.Option(False, "--no-limit", help="Remove the iteration cap"),
    effort: str | None = typer.Option(None, "--effort", help="Effort level: low|medium|high|xhigh|max"),
    bare: bool = typer.Option(False, "--bare", help="Skip DEVAGENT.md, memory, CodePrism, and permission gate"),
    allow_tools: str | None = typer.Option(
        None, "--allow-tools", help="Comma-separated tool names to auto-approve"
    ),
    output_format: str = typer.Option(
        "rich", "--output-format", help="Output format: rich (default), json, or stream-json"
    ),
    permission_mode: str = typer.Option(
        "default", "--permission-mode",
        help="Permission shorthand: default|accept-edits|read-only|auto|yolo",
    ),
    json_schema: str | None = typer.Option(
        None, "--json-schema",
        help="JSON Schema string. Validates final answer; retries once on mismatch (exit 2 if still invalid).",
    ),
    add_dir: list[str] = typer.Option(  # noqa: B008
        [], "--add-dir", help="Extra directory the agent may read/write (repeatable)"
    ),
) -> None:
    """Run a single task non-interactively and exit (exit code 0=success, 1=error, 2=schema mismatch)."""
    import json as _json

    from devagent.agent.flows import DevAgentSession
    from devagent.agent.loop import ErrorEvent, FinalAnswerEvent
    from devagent.core.project import detect_project_root
    from devagent.output.streaming import emit_json, render_events, stream_json_events

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    # Parse and validate the JSON Schema argument up-front
    schema_dict: dict | None = None
    if json_schema:
        try:
            schema_dict = _json.loads(json_schema)
        except _json.JSONDecodeError as exc:
            console.print(f"[red]--json-schema is not valid JSON: {exc}[/red]")
            raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model

    project_root, _ = detect_project_root(Path(project) if project else None)
    tools_list = [t.strip() for t in allow_tools.split(",")] if allow_tools else None

    # Phase 11 — stdin pipe: prepend piped data to the task
    if not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                task = f"<stdin>\n{stdin_data}\n</stdin>\n\n{task}"
        except Exception:
            pass

    def _make_session() -> DevAgentSession:
        return DevAgentSession(
            cfg,
            project_root,
            max_tokens=max_tokens,
            no_limit=no_limit,
            effort=effort,
            bare=bare,
            allow_tools=tools_list,
            permission_mode=permission_mode,
            extra_dirs=list(add_dir) or None,
        )

    def _extract_final_text(evs: list) -> str:
        for ev in evs:
            if isinstance(ev, FinalAnswerEvent):
                return ev.text
        return ""

    try:
        session = _make_session()
    except Exception as exc:
        _handle_error(exc)
        return

    # First attempt
    events = list(session._loop.run(task))

    def _render(evs: list) -> None:
        if output_format == "stream-json":
            stream_json_events(iter(evs))
        elif output_format == "json":
            emit_json(iter(evs))
        else:
            render_events(iter(evs))

    _render(events)

    # JSON Schema validation + one retry
    if schema_dict is not None:
        final_text = _extract_final_text(events)
        valid, err_msg = _validate_against_schema(final_text, schema_dict)
        if not valid:
            console.print(
                f"[yellow]Schema validation failed: {err_msg}. Retrying once…[/yellow]"
            )
            retry_task = (
                f"{task}\n\n"
                f"IMPORTANT: Your previous response did not match the required JSON Schema.\n"
                f"Error: {err_msg}\n"
                f"Schema: {_json.dumps(schema_dict)}\n"
                f"Respond with ONLY a JSON object that satisfies the schema. No prose, no markdown fences."
            )
            try:
                retry_session = _make_session()
                retry_events = list(retry_session._loop.run(retry_task))
            except Exception as exc:
                _handle_error(exc)
                raise typer.Exit(2)

            _render(retry_events)

            events = retry_events
            retry_text = _extract_final_text(retry_events)
            valid2, err_msg2 = _validate_against_schema(retry_text, schema_dict)
            if not valid2:
                console.print(f"[red]Schema validation failed after retry: {err_msg2}[/red]")
                raise typer.Exit(2)

    # Clean up session record if --no-session requested
    if no_session:
        try:
            session._mgr.delete(session.session_id)
        except Exception:
            pass

    # Exit 1 if any ErrorEvent was emitted
    for ev in events:
        if isinstance(ev, ErrorEvent):
            raise typer.Exit(1)
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# Phase 4 flow commands
# ---------------------------------------------------------------------------

@app.command()
def implement(
    url: str = typer.Argument(..., help="GitHub issue URL (https://github.com/owner/repo/issues/N)"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    model: str | None = typer.Option(None, "--model", "-m"),
    plan: bool = typer.Option(False, "--plan", help="Generate and approve a plan before executing"),
    no_limit: bool = typer.Option(False, "--no-limit", help="Remove the iteration cap"),
) -> None:
    """Fetch a GitHub issue and implement it end-to-end (branch → edit → test → PR)."""
    from devagent.agent.flows import run_implement
    from devagent.core.project import detect_project_root

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        run_implement(cfg, project_root, url, max_tokens=max_tokens, plan_mode=plan, no_limit=no_limit)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        _handle_error(exc)


@app.command()
def review(
    url: str = typer.Argument(..., help="GitHub PR URL (https://github.com/owner/repo/pull/N)"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Fetch a GitHub PR diff and post an AI code review with inline comments."""
    from devagent.agent.flows import run_review
    from devagent.core.project import detect_project_root

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        run_review(cfg, project_root, url, max_tokens=max_tokens)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        _handle_error(exc)


@app.command()
def triage(
    repo: str = typer.Argument(..., help="GitHub repo as owner/repo or full URL"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Classify open GitHub issues by effort and post triage comments."""
    from devagent.agent.flows import run_triage
    from devagent.core.project import detect_project_root

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        run_triage(cfg, project_root, repo, max_tokens=max_tokens)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        _handle_error(exc)


@app.command("fix-ci")
def fix_ci(
    url: str = typer.Argument(
        ..., help="GitHub Actions run URL (https://github.com/owner/repo/actions/runs/ID)"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Read failed CI logs and propose/apply a fix."""
    from devagent.agent.flows import run_fix_ci
    from devagent.core.project import detect_project_root

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        run_fix_ci(cfg, project_root, url, max_tokens=max_tokens)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        _handle_error(exc)


@app.command("autofix-pr")
def autofix_pr(
    pr_url: str = typer.Argument(
        ..., help="GitHub PR URL (https://github.com/owner/repo/pull/N)"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    model: str | None = typer.Option(None, "--model", "-m"),
    poll_interval: int = typer.Option(120, "--poll-interval", help="Seconds between polls (default 120)"),
    max_polls: int | None = typer.Option(None, "--max-polls", help="Stop after N polls (default: run forever)"),
) -> None:
    """Watch a PR for CI failures and review comments, auto-fix and push.

    Polls GitHub every --poll-interval seconds. Stops on Ctrl-C or --max-polls reached.
    Requires GITHUB_TOKEN or GH_TOKEN in environment or devagent config.
    """
    from devagent.agent.flows import run_autofix_pr
    from devagent.core.project import detect_project_root

    if not config_exists():
        console.print("[red]No config found. Run [bold]devagent init[/bold] first.[/red]")
        raise typer.Exit(1)

    cfg = load_config()
    if model:
        cfg.llm.model = model
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        run_autofix_pr(cfg, project_root, pr_url, poll_interval=poll_interval, max_polls=max_polls)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]autofix-pr stopped.[/dim]")
    except Exception as exc:
        _handle_error(exc)


@app.command()
def onboard(
    project: str | None = typer.Option(
        None, "--project", "-p", help="Project path (defaults to current directory)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON instead of rich display"),
) -> None:
    """Generate a CodePrism architecture overview for the current project.

    Shows: file map, most-coupled files, top public symbols, and test gap summary.
    Requires the project to be indexed: codeprism index <path>
    """
    from devagent.core.project import detect_project_root

    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        from devagent.codeprism.client import CodePrismClient
    except ImportError:
        console.print("[red]codeprism-ai package not installed.[/red]")
        raise typer.Exit(1)

    cp = CodePrismClient(str(project_root))
    if not cp.is_indexed:
        console.print(
            Panel(
                f"[yellow]This project has not been indexed by CodePrism.[/yellow]\n\n"
                f"Run:  [bold cyan]codeprism index {project_root}[/bold cyan]\n"
                "Then re-run [bold]devagent onboard[/bold].",
                title="CodePrism: Not Indexed",
                border_style="yellow",
            )
        )
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold cyan]DevAgent Onboarding[/bold cyan]\n"
        f"[dim]Project: {project_root}[/dim]",
        border_style="cyan",
    ))

    # ── 1. Stats ──────────────────────────────────────────────────────────
    with console.status("[cyan]Loading graph stats...[/cyan]"):
        stats = cp.get_stats()

    if "error" not in stats:
        t = Table(title="Knowledge Graph", border_style="dim")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        for k, v in stats.items():
            if k not in ("last_indexed_at",):
                t.add_row(str(k).replace("_", " ").title(), str(v))
        console.print(t)

    # ── 2. File map (top 20 files by symbol count) ────────────────────────
    with console.status("[cyan]Building file map...[/cyan]"):
        fm = cp.get_file_map()

    if "error" not in fm:
        entries = sorted(fm.get("entries", []), key=lambda e: e.get("symbols", 0), reverse=True)
        t = Table(title="Files by Symbol Count (top 20)", border_style="dim")
        t.add_column("File", style="cyan")
        t.add_column("Symbols", justify="right")
        t.add_column("Role")
        for e in entries[:20]:
            t.add_row(e["path"], str(e.get("symbols", 0)), e.get("role", ""))
        console.print()
        console.print(t)

    # ── 3. Most-coupled files (highest impact symbols) ────────────────────
    console.print()
    console.print("[bold]Most-coupled symbols[/bold] (HIGH/CRITICAL impact on change):")
    coupled = []
    if "error" not in fm:
        for e in entries[:15]:
            summary = cp.get_module_summary(e["path"])
            if "error" in summary:
                continue
            for sym in summary.get("public_api", [])[:3]:
                impact = cp.get_impact(e["path"], sym["name"])
                if "error" not in impact and impact.get("severity") in ("HIGH", "CRITICAL"):
                    coupled.append({
                        "file": e["path"],
                        "symbol": sym["name"],
                        "severity": impact["severity"],
                        "surface": impact.get("estimated_change_surface", 0),
                        "public": impact.get("public_api_affected", False),
                    })

    if coupled:
        coupled.sort(key=lambda x: x["surface"], reverse=True)
        t = Table(border_style="dim")
        t.add_column("Symbol", style="bold")
        t.add_column("File", style="cyan")
        t.add_column("Severity", style="red")
        t.add_column("Dependents", justify="right")
        t.add_column("Public API")
        for row in coupled[:12]:
            t.add_row(
                row["symbol"], row["file"], row["severity"],
                str(row["surface"]), "yes" if row["public"] else "no",
            )
        console.print(t)
    else:
        console.print("  [dim](none found — graph may be small or all symbols low-impact)[/dim]")

    # ── 4. Test gap summary ───────────────────────────────────────────────
    console.print()
    console.print("[bold]Test coverage gaps[/bold] (public files without a matching test file):")
    no_tests = []
    if "error" not in fm:
        for e in entries:
            if "test" in e["path"].lower():
                continue
            summary = cp.get_module_summary(e["path"])
            if "error" in summary:
                continue
            if not summary.get("test_coverage_file") and summary.get("public_api"):
                no_tests.append(e["path"])

    if no_tests:
        for p in no_tests[:10]:
            console.print(f"  [yellow]•[/yellow] {p}")
        if len(no_tests) > 10:
            console.print(f"  ... and {len(no_tests) - 10} more")
    else:
        console.print("  [green]No obvious test gaps found.[/green]")

    console.print()
    console.print(
        "[dim]Tip: Use [bold]devagent run[/bold] to start an agent session. "
        "The agent will automatically use the CodePrism graph to reduce token usage.[/dim]"
    )


@app.command()
def doctor() -> None:

    """Check that all dependencies and services are working."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]DevAgent Doctor[/bold cyan]\n"
            "Checking all dependencies and services...",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(border_style="dim")
    table.add_column("Service", style="bold", min_width=20)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Details")

    # 1. Config file
    if config_exists():
        try:
            cfg = load_config()
            table.add_row("Config File", "OK", str(get_config_path()))
        except Exception as exc:
            cfg = None
            table.add_row("Config File", "FAIL", f"Invalid: {exc}")
    else:
        cfg = None
        table.add_row("Config File", "FAIL", "Not found - run: devagent init")

    # 2. LLM provider
    if cfg:
        with console.status("[dim]Checking LLM provider...[/dim]"):
            if cfg.llm.provider == "ollama":
                ok, msg = _validate_ollama(cfg.llm.base_url, cfg.llm.model)
            else:
                ok = bool(cfg.llm.api_key)
                msg = f"API key configured for {cfg.llm.provider}" if ok else "No API key"
        table.add_row(
            f"LLM ({cfg.llm.provider})",
            "OK" if ok else "FAIL",
            msg,
        )
        # Offline capability row
        offline_capable = cfg.llm.provider == "ollama"
        table.add_row(
            "Offline Mode",
            "OK" if offline_capable else "CLOUD",
            "local (offline capable)" if offline_capable
            else f"cloud - requires internet ({cfg.llm.provider})",
        )
    else:
        table.add_row("LLM Provider", "FAIL", "No config")
        table.add_row("Offline Mode", "FAIL", "No config")

    # 3. GitHub token
    if cfg and cfg.github.token:
        with console.status("[dim]Checking GitHub token...[/dim]"):
            ok, msg = _validate_github_token(cfg.github.token)
        table.add_row("GitHub Token", "OK" if ok else "FAIL", msg)
    else:
        table.add_row("GitHub Token", "FAIL", "Not configured")

    # 4. Search provider
    if cfg:
        if cfg.search_provider == "brave" and cfg.brave.api_key:
            with console.status("[dim]Checking Brave Search...[/dim]"):
                ok, msg = _validate_brave_key(cfg.brave.api_key)
            table.add_row("Brave Search", "OK" if ok else "FAIL", msg)
        elif cfg.search_provider == "searchx" and cfg.searchx.api_key:
            with console.status("[dim]Checking SearchX...[/dim]"):
                ok, msg = _validate_searchx_key(cfg.searchx.api_key)
            table.add_row("SearchX", "OK" if ok else "FAIL", msg)
        else:
            table.add_row(f"{cfg.search_provider.capitalize()} Search", "WARN", "Not configured (optional)")
    else:
        table.add_row("Search Provider", "FAIL", "No config")

    # 5. Node.js
    ok, msg = _check_command(["node", "--version"])
    table.add_row("Node.js", "OK" if ok else "FAIL", msg if ok else "Not found - download from nodejs.org")

    # 6. npx
    ok, msg = _check_command(["npx", "--version"])
    table.add_row("npx", "OK" if ok else "FAIL", msg if ok else "Not found - install Node.js")

    console.print(table)
    console.print()



@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(7331, "--port", "-p", help="Port to listen on"),
    ui: bool = typer.Option(False, "--ui", help="Open graph visualisation in browser on start"),
    project: str | None = typer.Option(None, "--project", help="Project path"),
) -> None:
    """Start the DevAgent REST + WebSocket API server (port 7331 by default)."""
    from devagent.core.project import detect_project_root
    from devagent.server.fastapi_app import serve as _serve

    cfg = load_config() if config_exists() else None
    project_root, _ = detect_project_root(Path(project) if project else None)

    try:
        _serve(host=host, port=port, config=cfg, project_root=project_root, open_ui=ui)
    except OSError as exc:
        console.print(f"[red]Could not start server: {exc}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Phase 7.4 — Skills sub-app
# ---------------------------------------------------------------------------

skills_app = typer.Typer(name="skills", help="Manage DevAgent skills.", add_completion=False)
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list() -> None:
    """List all available skills (built-in + user-defined)."""
    from rich.table import Table

    from devagent.skills.loader import load_all_skills

    all_skills = load_all_skills()

    table = Table(title="Available Skills", border_style="cyan")
    table.add_column("Command", style="cyan bold")
    table.add_column("Description")
    table.add_column("Tools", style="dim")
    table.add_column("Model", style="dim")

    for name, skill in sorted(all_skills.items()):
        tools = ", ".join(skill.tools_only) if skill.tools_only else "all"
        model = skill.model or "default"
        table.add_row(f"/{name}", skill.description, tools, model)

    console.print()
    console.print(table)
    user_dir = Path.home() / ".config" / "devagent" / "skills"
    console.print(f"\n[dim]User skills directory: {user_dir}[/dim]")
    console.print("[dim]Run [bold]devagent skills new[/bold] to create a skill.[/dim]")


@skills_app.command("new")
def skills_new() -> None:
    """Interactive wizard to create a new user-defined skill."""
    from rich.prompt import Prompt

    console.print()
    console.print("[bold cyan]Create a new skill[/bold cyan]\n")

    name = Prompt.ask("Skill name (used as /command)").strip().lower().replace(" ", "-")
    if not name:
        console.print("[red]Name cannot be empty.[/red]")
        raise typer.Exit(1)

    description = Prompt.ask("Short description")
    console.print("[dim]Enter the prompt the agent will receive when this skill is invoked.[/dim]")
    console.print("[dim](Type END on a new line to finish)[/dim]")

    prompt_lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        prompt_lines.append(line)
    prompt_text = "\n".join(prompt_lines)

    tools_input = Prompt.ask(
        "Restrict to tools (comma-separated, leave empty for all)",
        default="",
    ).strip()
    tools_only = [t.strip() for t in tools_input.split(",") if t.strip()] if tools_input else []

    model = Prompt.ask(
        "Model tier (empty = session default, or: cheap / reviewing / planning)",
        default="",
    ).strip()

    max_iter = Prompt.ask("Max iterations (0 = session default)", default="0").strip()
    try:
        max_iter_int = int(max_iter)
    except ValueError:
        max_iter_int = 0

    # Build TOML content
    import tomli_w
    data: dict = {
        "name": name,
        "description": description,
        "prompt": prompt_text,
    }
    if tools_only:
        data["tools_only"] = tools_only
    if model:
        data["model"] = model
    if max_iter_int:
        data["max_iter"] = max_iter_int

    user_dir = Path.home() / ".config" / "devagent" / "skills"
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / f"{name}.toml"

    with open(out_path, "wb") as f:
        tomli_w.dump(data, f)

    console.print(f"\n[green]Skill saved: {out_path}[/green]")
    console.print(f"[dim]Use it with: /{name}[/dim]")


# ---------------------------------------------------------------------------
# Phase 7.6 — DevAgent as MCP server
# ---------------------------------------------------------------------------

@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
    port: int = typer.Option(7332, "--port", help="Port for SSE transport"),
    project: str | None = typer.Option(None, "--project", help="Project path"),
) -> None:
    """Start DevAgent as an MCP server (connect from Claude Desktop, Cursor, etc.)."""
    from devagent.core.project import detect_project_root
    from devagent.mcp.server import serve_mcp

    cfg = load_config() if config_exists() else None
    from devagent.core.config import DevAgentConfig
    if cfg is None:
        cfg = DevAgentConfig()

    project_root, _ = detect_project_root(Path(project) if project else None)

    if transport == "stdio":
        console.print("[dim]DevAgent MCP server starting (stdio)...[/dim]")
        console.print("[dim]Add to claude_desktop_config.json:[/dim]")
        console.print('[dim]  "devagent": {"command": "devagent", "args": ["mcp"]}[/dim]')
    else:
        console.print(f"[dim]DevAgent MCP server starting (SSE on port {port})...[/dim]")

    try:
        serve_mcp(str(project_root), cfg, transport=transport, port=port)
    except Exception as exc:
        _handle_error(exc)


# ---------------------------------------------------------------------------
# Phase 9 — Multi-agent orchestration
# ---------------------------------------------------------------------------

@app.command()
def orchestrate(
    task: str = typer.Argument(..., help="High-level task for the agent team to accomplish"),
    workers: int = typer.Option(4, "--workers", "-w", help="Maximum parallel workers"),
    plan: bool = typer.Option(False, "--plan", is_flag=True, help="Show decomposition plan before executing"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model"),
    max_iter: int | None = typer.Option(
        None, "--max-iter",
        help="Max iterations per worker agent (default: agent.max_iterations from config)",
    ),
    max_repair: int | None = typer.Option(
        None, "--max-repair",
        help="Max auto-repair attempts per worker (default: agent.max_repair_iterations from config)",
    ),
) -> None:
    """Decompose a task and run multiple worker agents in parallel."""
    from devagent.agent.loop import ErrorEvent, FinalAnswerEvent, StatusEvent, ThinkingEvent
    from devagent.agent.orchestrator import OrchestratorSession
    from devagent.core.project import detect_project_root

    cfg = load_config() if config_exists() else None
    from devagent.core.config import DevAgentConfig
    if cfg is None:
        cfg = DevAgentConfig()

    if model:
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"model": model})})

    project_root, _ = detect_project_root(Path(project) if project else None)

    console.print(
        f"[bold cyan]DevAgent Orchestrate[/bold cyan]  |  "
        f"{cfg.llm.provider}/{cfg.llm.model}  |  workers: {workers}"
    )

    session = OrchestratorSession(
        cfg=cfg,
        project_root=project_root,
        max_workers=workers,
        plan_mode=plan,
        worker_max_iterations=max_iter,
        worker_max_repair=max_repair,
    )

    exit_code = 0
    try:
        for event in session.run(task):
            if isinstance(event, ThinkingEvent):
                console.print(f"[dim]{event.text}[/dim]")
            elif isinstance(event, StatusEvent):
                console.print(f"[cyan]» {event.status_line}[/cyan]")
            elif isinstance(event, ErrorEvent):
                console.print(f"[red]✗ {event.message}[/red]")
                exit_code = 1
            elif isinstance(event, FinalAnswerEvent):
                console.print()
                console.rule("[cyan]Final Summary[/cyan]")
                console.print(event.text)
    except KeyboardInterrupt:
        console.print("\n[dim]Orchestration interrupted.[/dim]")
        exit_code = 1
    except Exception as exc:
        _handle_error(exc)
        exit_code = 1

    raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# Phase 12 — Project scaffolding
# ---------------------------------------------------------------------------

_DEVAGENT_MD_TEMPLATE = """\
# DEVAGENT.md

Project-specific instructions for DevAgent.
This file is automatically injected into the agent's system prompt at session start.

## Tech stack
<!-- e.g. Python 3.12, FastAPI, SQLite, Typer, pytest -->

## Test command
<!-- e.g. python -m pytest tests/ -q -->

## Code conventions
<!-- e.g. ruff format, no type: ignore, prefer edit_file over write_file -->

## Important paths
<!-- e.g. devagent/ — main source, tests/ — all tests -->

## Known constraints
<!-- e.g. No LangGraph; pure Python with official SDKs -->
"""


def _collect_project_context(root: Path, max_readme_lines: int = 200) -> str:
    """Gather a compact codebase summary for LLM-based DEVAGENT.md generation."""
    parts: list[str] = []

    # Top-level directory listing (dirs + key files)
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    listing = "\n".join(
        ("  " + e.name + ("/" if e.is_dir() else ""))
        for e in entries
        if not e.name.startswith(".")
    )
    parts.append(f"## Directory structure\n```\n{listing}\n```")

    # Key manifest files
    for fname in ("pyproject.toml", "setup.py", "package.json", "go.mod", "Cargo.toml"):
        p = root / fname
        if p.exists():
            try:
                parts.append(f"## {fname}\n```\n{p.read_text(encoding='utf-8')[:3000]}\n```")
            except OSError:
                pass

    # README excerpt
    for fname in ("README.md", "README.rst", "README"):
        p = root / fname
        if p.exists():
            try:
                lines = p.read_text(encoding="utf-8").splitlines()[:max_readme_lines]
                parts.append(f"## README excerpt\n{chr(10).join(lines)}")
            except OSError:
                pass
            break

    return "\n\n".join(parts)


_GENERATE_SYSTEM = (
    "You are a technical writer generating a DEVAGENT.md file. "
    "DEVAGENT.md is a concise Markdown file that tells a coding AI agent how to work "
    "effectively in this project. Write factual, specific content — no fluff. "
    "Use ## headings: Tech stack, Test command, Code conventions, Important paths, "
    "Known constraints. If information is not available, omit that section rather "
    "than writing placeholder comments. Output only the Markdown content."
)

_GENERATE_USER = """\
Analyse this project and write a DEVAGENT.md file for it.

{context}

Write a DEVAGENT.md with the sections: Tech stack, Test command, Code conventions, \
Important paths, Known constraints. Be specific and concise. Output only Markdown."""


@app.command("init-project")
def init_project(
    path: str = typer.Argument(".", help="Project root to initialise (default: current directory)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing DEVAGENT.md"),
    generate: bool = typer.Option(
        False, "--generate", "-g",
        help="Use the configured LLM to write DEVAGENT.md from codebase analysis",
    ),
) -> None:
    """Scaffold DEVAGENT.md (project instructions) and .devagent/ directory.

    By default writes a static template. With --generate, the configured LLM
    analyses key project files (pyproject.toml, README, directory layout) and
    writes a tailored DEVAGENT.md automatically.
    """
    root = Path(path).resolve()

    devagent_md = root / "DEVAGENT.md"
    memory_dir = root / ".devagent"
    memory_file = memory_dir / "memory.md"

    if devagent_md.exists() and not force:
        console.print(f"[yellow]DEVAGENT.md already exists at {devagent_md}[/yellow]")
        console.print("Use [bold]--force[/bold] to overwrite.")
        raise typer.Exit(1)

    if generate:
        from devagent.core.llm import LLMClient, Message
        cfg = load_config(root)
        console.print("[cyan]Analysing project…[/cyan]")
        context = _collect_project_context(root)
        llm = LLMClient(cfg.llm)
        try:
            resp = llm.complete([
                Message(role="system", content=_GENERATE_SYSTEM),
                Message(role="user", content=_GENERATE_USER.format(context=context)),
            ])
            content = resp.content.strip()
            if not content.startswith("#"):
                content = "# DEVAGENT.md\n\n" + content
        except Exception as exc:
            console.print(f"[red]LLM call failed:[/red] {exc}")
            console.print("[dim]Falling back to static template.[/dim]")
            content = _DEVAGENT_MD_TEMPLATE
    else:
        content = _DEVAGENT_MD_TEMPLATE

    devagent_md.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/green] Created {devagent_md}")

    memory_dir.mkdir(parents=True, exist_ok=True)
    if not memory_file.exists():
        memory_file.write_text(
            "# DevAgent Memory\n<!-- auto-managed — edit with care -->\n",
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] Created {memory_file}")

    gitignore = root / ".gitignore"
    if gitignore.exists():
        content_gi = gitignore.read_text(encoding="utf-8")
        if ".devagent/" not in content_gi:
            with gitignore.open("a", encoding="utf-8") as f:
                f.write("\n# DevAgent cross-session memory (local only)\n.devagent/\n")
            console.print(f"[green]✓[/green] Added .devagent/ to {gitignore}")

    console.print("\n[bold]Next steps:[/bold]")
    if generate:
        console.print("  1. Review [bold]DEVAGENT.md[/bold] — the LLM may have missed details.")
    else:
        console.print("  1. Edit [bold]DEVAGENT.md[/bold] to describe your project, tech stack, and conventions.")
    console.print("  2. Run [bold]devagent run[/bold] — the agent will read DEVAGENT.md on startup.")


# ---------------------------------------------------------------------------
# devagent bench — B3, B4, B5 benchmark commands
# ---------------------------------------------------------------------------

bench_app = typer.Typer(
    name="bench",
    help="Benchmark DevAgent performance — native task set, cost sweep, CI canary.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


@bench_app.command("native")
def bench_native(
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    difficulty: str | None = typer.Option(None, "--difficulty", "-d", help="easy | medium | hard"),
    task_id: list[str] | None = typer.Option(None, "--task-id", "-t", help="Run specific task(s) by ID (repeatable: -t id1 -t id2)"),  # noqa: B008
    live: bool = typer.Option(False, "--live/--dry", help="Use real LLM (--live) or dry-run (--dry)"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Override LLM provider (ollama, anthropic, openai)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model for live runs"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max number of tasks to run"),
) -> None:
    """Run the native task set benchmark (B3).

    By default runs in --dry mode (no LLM, oracle only).
    Pass --live to invoke the real agent against each task.
    """
    from devagent.bench.report import BenchReport
    from devagent.bench.runner import BenchRunner

    tasks = BenchRunner.load_tasks(category=category, difficulty=difficulty)
    if task_id:
        id_set = set(task_id)
        tasks = [t for t in tasks if t.id in id_set]
    if limit:
        tasks = tasks[:limit]

    if not tasks:
        console.print("[yellow]No tasks matched the filters.[/yellow]")
        raise typer.Exit(0)

    mode = "[yellow]live[/yellow]" if live else "[dim]dry[/dim]"
    console.print(
        f"\n[bold]devagent bench native[/bold] — {len(tasks)} tasks, mode={mode}"
    )

    # Always write a rolling partial-results file so a crash doesn't lose completed tasks.
    partial_path = BenchReport.partial_json_path(label="native")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_partial(result, all_so_far):
        BenchReport.write_partial_json(all_so_far, partial_path)

    runner = BenchRunner(tasks=tasks, dry_run=not live, provider=provider, model=model)
    results = runner.run_all(on_result=_save_partial)

    BenchReport.render_table(results)
    BenchReport.render_summary(results)

    # Always persist — results are saved after every task (partial) and again
    # as a timestamped final file so nothing is lost on crash or mid-run stop.
    path = BenchReport.save_json(results, label="native", model=model, provider=provider)
    console.print(f"\n[dim]Results saved -> {path}[/dim]")

    # Partial file is now redundant — clean it up.
    if partial_path.exists():
        partial_path.unlink()

    passed = sum(1 for r in results if r.passed)
    if passed < len(results):
        raise typer.Exit(1)


@bench_app.command("canary")
def bench_canary(
    threshold: float = typer.Option(0.80, "--threshold", help="Minimum pass rate (0.0–1.0)"),
    skip_legacy: bool = typer.Option(False, "--skip-legacy", help="Skip existing bench_*.py scripts"),
    save_json: bool = typer.Option(False, "--save-json", help="Persist canary results to benchmarks/results/"),
) -> None:
    """Run the CI canary benchmark (B5) — no LLM required.

    Runs framework checks from canary.json plus the existing bench_*.py scripts.
    Exits with code 1 if pass_rate < threshold.
    """
    import json
    from pathlib import Path

    canary_path = Path(__file__).parent.parent / "benchmarks" / "tasks" / "canary.json"
    canary = json.loads(canary_path.read_text(encoding="utf-8"))

    passed_checks = 0
    total_checks = 0
    failures: list[str] = []

    # 1. Framework checks (pure Python assertions)
    console.print("\n[bold]Framework checks[/bold]")
    for check in canary["framework_checks"]:
        total_checks += 1
        try:
            exec(check["check"])  # noqa: S102
            console.print(f"  [green]PASS[/green] {check['id']}: {check['description']}")
            passed_checks += 1
        except Exception as exc:
            console.print(f"  [red]FAIL[/red] {check['id']}: {exc}")
            failures.append(check["id"])

    # 2. Legacy benchmark scripts (mock-LLM, no real agent needed)
    if not skip_legacy:
        console.print("\n[bold]Legacy benchmark scripts[/bold]")
        bench_dir = Path(__file__).parent.parent / "benchmarks"
        for script_rel in canary.get("legacy_scripts", []):
            script_path = Path(__file__).parent.parent / script_rel
            if not script_path.exists():
                console.print(f"  [dim]skip[/dim] {script_rel} (not found)")
                continue
            total_checks += 1
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(bench_dir),
            )
            if result.returncode == 0:
                console.print(f"  [green]PASS[/green] {script_rel}")
                passed_checks += 1
            else:
                short = (result.stdout + result.stderr).strip()[:200]
                console.print(f"  [red]FAIL[/red] {script_rel}\n    {short}")
                failures.append(script_rel)

    # Summary
    rate = passed_checks / total_checks if total_checks else 0.0
    color = "green" if rate >= threshold else "red"
    console.print(
        f"\n[bold]Canary result:[/bold] "
        f"[{color}]{passed_checks}/{total_checks} passed ({rate * 100:.0f}%)[/{color}]"
        f" | threshold={threshold * 100:.0f}%"
    )

    if save_json:
        import datetime

        from devagent.bench.report import _RESULTS_DIR

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        path = _RESULTS_DIR / f"canary_{ts}.json"
        rows: list[dict] = [
            {
                "task_id": chk["id"],
                "passed": chk["id"] not in failures,
                "duration_sec": 0.0,
                "iterations_used": 0,
                "cost_usd": 0.0,
                "files_touched": [],
                "files_missed": [],
                "oracle_output": "",
            }
            for chk in canary["framework_checks"]
        ]
        if not skip_legacy:
            for script_rel in canary.get("legacy_scripts", []):
                script_path = Path(__file__).parent.parent / script_rel
                if script_path.exists():
                    rows.append(
                        {
                            "task_id": script_rel,
                            "passed": script_rel not in failures,
                            "duration_sec": 0.0,
                            "iterations_used": 0,
                            "cost_usd": 0.0,
                            "files_touched": [],
                            "files_missed": [],
                            "oracle_output": "",
                        }
                    )
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        console.print(f"[dim]Canary results saved -> {path}[/dim]")

    if rate < threshold:
        console.print(f"[red]Canary FAILED — {len(failures)} check(s) failed: {failures}[/red]")
        raise typer.Exit(1)

    console.print("[green]Canary PASSED[/green]")


@bench_app.command("sweep")
def bench_sweep(
    params: str = typer.Option("model,max_iterations", "--params", help="Comma-separated param names to sweep"),
    limit: int = typer.Option(5, "--tasks", "-n", help="Number of tasks per combination"),
    category: str | None = typer.Option(None, "--category", "-c"),
    difficulty: str | None = typer.Option(None, "--difficulty", "-d"),
    live: bool = typer.Option(False, "--live/--dry", help="Use real LLM (--live) or dry-run (--dry)"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Override LLM provider (ollama, anthropic, openai)"),
    output_json: bool = typer.Option(False, "--output-json"),
) -> None:
    """Sweep parameter combinations for cost-vs-correctness analysis (B4).

    Runs the native task set with different model/iteration/effort combinations
    and renders a comparison table.  Pass --live to invoke the real agent.
    """
    import json

    from devagent.bench.sweep import SweepRunner

    param_keys = [p.strip() for p in params.split(",")]

    # Default grid values for known params
    _defaults: dict[str, list] = {
        "model": ["gpt-oss:20b", "glm-5.3-flash"],
        "max_iterations": [10, 30],
        "effort": ["low", "high", "max"],
    }
    param_grid = {k: _defaults.get(k, []) for k in param_keys if k in _defaults}

    if not param_grid:
        console.print(f"[red]Unknown params: {param_keys}. Choose from: {list(_defaults)}[/red]")
        raise typer.Exit(1)

    mode = "[yellow]live[/yellow]" if live else "[dim]dry[/dim]"
    console.print(f"\n[bold]devagent bench sweep[/bold] — grid: {param_grid}, tasks per combo: {limit}, mode={mode}")

    runner = SweepRunner(
        param_grid=param_grid,
        task_limit=limit,
        category=category,
        difficulty=difficulty,
        dry_run=not live,
        provider=provider,
    )
    sweep_results = runner.run()
    SweepRunner.render_table(sweep_results, param_keys)

    if output_json:
        import datetime

        from devagent.bench.report import _RESULTS_DIR

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        path = _RESULTS_DIR / f"sweep_{ts}.json"
        data = [
            {
                "params": sr.params,
                "pass_rate": round(sr.pass_rate, 4),
                "avg_cost_usd": round(sr.avg_cost_usd, 6),
                "avg_duration_sec": round(sr.avg_duration_sec, 2),
            }
            for sr in sweep_results
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"\n[dim]Sweep results saved -> {path}[/dim]")


@bench_app.command("history")
def bench_history(
    limit: int = typer.Option(10, "--limit", "-n", help="Show last N runs (0 = all)"),
    remote: bool = typer.Option(False, "--remote", help="Fetch results from the bench-results git branch"),
) -> None:
    """Show a trend table comparing saved benchmark runs.

    Reads native_*.json, sweep_*.json, and canary_*.json from benchmarks/results/.
    Use --remote to pull canary results persisted to the bench-results branch by CI.
    """
    import re
    import tempfile

    from devagent.bench.report import BenchReport

    src_dir = None
    if remote:
        fetch_result = subprocess.run(
            ["git", "fetch", "origin", "bench-results"],
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            console.print("[yellow]bench-results branch not found on origin — no remote results yet.[/yellow]")
            return

        ls_result = subprocess.run(
            ["git", "ls-tree", "--name-only", "origin/bench-results"],
            capture_output=True,
            text=True,
        )
        filenames = [
            f.strip()
            for f in ls_result.stdout.splitlines()
            if re.match(r"^(native|sweep|canary)_\d{8}_\d{6}\.json$", f.strip())
        ]
        if not filenames:
            console.print("[yellow]No result files found on bench-results branch.[/yellow]")
            return

        tmp_dir = tempfile.mkdtemp(prefix="devagent_history_")
        src_dir = Path(tmp_dir)
        for fname in filenames:
            show = subprocess.run(
                ["git", "show", f"origin/bench-results:{fname}"],
                capture_output=True,
                text=True,
            )
            if show.returncode == 0:
                (src_dir / fname).write_text(show.stdout, encoding="utf-8")

        console.print(f"[dim]Loaded {len(filenames)} remote result file(s) from bench-results[/dim]")

    BenchReport.render_history(limit=limit if limit > 0 else None, src_dir=src_dir)


@bench_app.command("leaderboard")
def bench_leaderboard(
    remote: bool = typer.Option(False, "--remote", help="Fetch results from the bench-results git branch"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write markdown leaderboard to this file"),
) -> None:
    """Show a leaderboard grouped by model — best and latest score per model.

    Use --output LEADERBOARD.md to write a markdown file (used by CI to keep
    the repo leaderboard up to date after every push to main).
    """
    import re
    import tempfile

    from devagent.bench.report import BenchReport

    src_dir = None
    if remote:
        fetch_result = subprocess.run(
            ["git", "fetch", "origin", "bench-results"],
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            console.print("[yellow]bench-results branch not found on origin — no remote results yet.[/yellow]")
            return

        ls_result = subprocess.run(
            ["git", "ls-tree", "--name-only", "origin/bench-results"],
            capture_output=True,
            text=True,
        )
        filenames = [
            f.strip()
            for f in ls_result.stdout.splitlines()
            if re.match(r"^native_\d{8}_\d{6}\.json$", f.strip())
        ]
        if filenames:
            tmp_dir = tempfile.mkdtemp(prefix="devagent_leaderboard_")
            src_dir = Path(tmp_dir)
            for fname in filenames:
                show = subprocess.run(
                    ["git", "show", f"origin/bench-results:{fname}"],
                    capture_output=True,
                    text=True,
                )
                if show.returncode == 0:
                    (src_dir / fname).write_text(show.stdout, encoding="utf-8")
            console.print(f"[dim]Loaded {len(filenames)} remote result file(s) from bench-results[/dim]")

    if output:
        import datetime
        date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        md = BenchReport.generate_leaderboard_md(src_dir=src_dir, date=date)
        Path(output).write_text(md, encoding="utf-8")
        console.print(f"[green]Leaderboard written to {output}[/green]")
    else:
        BenchReport.render_leaderboard(src_dir=src_dir)


# ---------------------------------------------------------------------------
# Phase 10 — hooks subcommand group
# ---------------------------------------------------------------------------

hooks_app = typer.Typer(name="hooks", help="Manage DevAgent tool-use hooks.", add_completion=False)
app.add_typer(hooks_app, name="hooks")


@hooks_app.command("list")
def hooks_list(
    project: str = typer.Option(".", "--project", "-p", help="Project root"),
) -> None:
    """List all hooks configured in .devagent/hooks.toml."""
    from devagent.hooks.config import load_hooks
    hooks = load_hooks(project)
    if not hooks:
        console.print("[dim]No hooks configured.[/dim]")
        console.print("\nCreate [bold].devagent/hooks.toml[/bold]:")
        console.print('[[hooks]]')
        console.print('event   = "pre_tool_use"')
        console.print('tool    = "run_shell"          # omit to match all tools')
        console.print('type    = "command"')
        console.print('command = "echo $DEVAGENT_TOOL_INPUT"')
        return
    for h in hooks:
        tool_str = f" tool={h.tool}" if h.tool else " (all tools)"
        detail = h.command or h.url
        console.print(f"  [cyan]{h.event}[/cyan]{tool_str}  [{h.type}] {detail}")


@hooks_app.command("test")
def hooks_test(
    event: str = typer.Argument("pre_tool_use", help="Event to test"),
    tool: str = typer.Option("run_shell", "--tool", "-t", help="Tool name"),
    args_json: str = typer.Option("{}", "--args", help="Tool args as JSON string"),
    project: str = typer.Option(".", "--project", "-p", help="Project root"),
) -> None:
    """Dry-run hooks for a given event + tool without executing the real tool."""
    import json as _json

    from devagent.hooks.runner import HookRunner
    runner = HookRunner(project)
    if not runner.hooks:
        console.print("[dim]No hooks loaded from .devagent/hooks.toml[/dim]")
        return
    try:
        args = _json.loads(args_json)
    except Exception:
        console.print(f"[red]Invalid JSON: {args_json}[/red]")
        raise typer.Exit(1)

    lifecycle_events = {"session_start", "session_end"}
    if event in lifecycle_events:
        runner._fire_lifecycle(event, {"session_id": "test-dry-run"})
        console.print(f"[green]lifecycle hook fired[/green] event={event}")
        return

    if event == "post_tool_use":
        runner.post_tool_use(tool, args, result="dry-run")
        console.print(f"[green]post hook fired[/green] tool={tool}")
        return

    result = runner.pre_tool_use(tool, args)
    if result.allowed:
        console.print("[green]allowed[/green]")
        if result.rewritten_args is not None:
            console.print(f"  rewritten args: {result.rewritten_args}")
    else:
        console.print(f"[red]blocked[/red]: {result.feedback}")
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# Phase 14 — agent definition subcommand group
# ---------------------------------------------------------------------------

agent_app = typer.Typer(name="agent", help="Manage DevAgent agent definitions.", add_completion=False)
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list(
    project: str = typer.Option(".", "--project", "-p", help="Project root"),
) -> None:
    """List agent definitions from .devagent/agents/*.toml."""
    from devagent.agents.loader import load_agent_defs
    defs = load_agent_defs(project)
    if not defs:
        console.print("[dim]No agent definitions found in .devagent/agents/[/dim]")
        console.print("Run [bold cyan]devagent agent new <name>[/bold cyan] to create one.")
        return
    for name, ag in defs.items():
        console.print(
            f"  [cyan]{name}[/cyan]  {ag.description or '(no description)'}  "
            f"[dim]model={ag.model}  max_iter={ag.max_iter}  perm={ag.permission}[/dim]"
        )


@agent_app.command("new")
def agent_new(
    name: str = typer.Argument(..., help="Agent slug (used as filename and invocation name)"),
    project: str = typer.Option(".", "--project", "-p", help="Project root"),
) -> None:
    """Scaffold a new agent definition in .devagent/agents/<name>.toml."""
    target = Path(project) / ".devagent" / "agents" / f"{name}.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        console.print(f"[yellow]Already exists: {target}[/yellow]")
        raise typer.Exit(1)
    template = (
        f'name         = "{name}"\n'
        f'description  = "Describe what this agent does"\n'
        f'worker_type  = "implementer"\n'
        f'model        = "coding"\n'
        f'max_iter     = 20\n'
        f'tools        = []  # empty = all tools\n'
        f'permission   = "default"\n'
        f'prompt       = """\n'
        f'You are a specialised coding agent. Describe your role here.\n'
        f'"""\n'
    )
    target.write_text(template, encoding="utf-8")
    console.print(f"[green]Created:[/green] {target}")
    console.print(f"Invoke in a session with: [bold cyan]/{name}[/bold cyan]")


@agent_app.command("run")
def agent_run(
    name: str = typer.Argument(..., help="Agent name to run"),
    task: str = typer.Option("", "--task", "-t", help="Task (uses agent's prompt if empty)"),
    project: str = typer.Option(".", "--project", "-p", help="Project root"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background thread"),
) -> None:
    """Run a named agent definition from .devagent/agents/."""
    from devagent.agents.loader import load_agent_defs
    defs = load_agent_defs(project)
    if name not in defs:
        console.print(f"[red]Agent not found: {name}[/red]")
        console.print("Run [bold]devagent agent list[/bold] to see available agents.")
        raise typer.Exit(1)
    ag = defs[name]
    cfg = load_config(project)
    task_msg = task or ag.prompt or f"Run the {name} agent on the current project."
    use_worktree = ag.isolation == "worktree"
    agent_name = name if ag.memory == "project" else None
    if background:
        import threading
        import uuid

        from devagent.session.task_store import get_store as _get_store
        task_id = uuid.uuid4().hex[:8]
        _get_store().add(task_id, f"{name}: {task_msg[:40]}")
        def _bg(tid=task_id, t=task_msg, c=cfg, r=project, wt=use_worktree, an=agent_name):
            try:
                from devagent.agent.flows import DevAgentSession
                if wt:
                    from devagent.agent.worktree import WorktreeError, isolated_worktree
                    try:
                        with isolated_worktree(r, branch_prefix=name) as (wt_path, _branch):
                            s = DevAgentSession(c, str(wt_path), bare=True,
                                                permission_mode=ag.permission, agent_name=an)
                            result = s.run_message(t, quiet=True)
                    except WorktreeError as exc:
                        _get_store().complete(tid, f"[worktree error] {exc}", failed=True)
                        return
                else:
                    s = DevAgentSession(c, r, bare=True, permission_mode=ag.permission, agent_name=an)
                    result = s.run_message(t, quiet=True)
                _get_store().complete(tid, result or "done")
            except Exception as exc:
                _get_store().complete(tid, str(exc), failed=True)
        threading.Thread(target=_bg, daemon=True).start()
        console.print(f"[dim]Background task {task_id} started for agent: {name}[/dim]")
        console.print("Check with: [bold]devagent tasks[/bold]")
    else:
        from devagent.agent.flows import DevAgentSession
        if use_worktree:
            from devagent.agent.worktree import WorktreeError, isolated_worktree
            try:
                with isolated_worktree(project, branch_prefix=name) as (wt_path, branch):
                    console.print(f"[dim]Worktree: {wt_path} (branch {branch})[/dim]")
                    session = DevAgentSession(cfg, str(wt_path), permission_mode=ag.permission,
                                             agent_name=agent_name)
                    session.run_message(task_msg)
            except WorktreeError as exc:
                console.print(f"[red]Worktree error:[/red] {exc}")
                raise typer.Exit(1)
        else:
            session = DevAgentSession(cfg, project, permission_mode=ag.permission, agent_name=agent_name)
            session.run_message(task_msg)


@app.command("tasks")
def list_tasks() -> None:
    """List background agent tasks for this process."""
    from devagent.session.task_store import get_store as _get_store
    tasks = _get_store().list_tasks()
    if not tasks:
        console.print("[dim]No background tasks.[/dim]")
        return
    for t in tasks:
        color = {"running": "yellow", "done": "green", "failed": "red"}.get(t.status, "white")
        console.print(f"  [{color}]{t.status}[/{color}]  {t.task_id}  {t.label}  ({t.elapsed:.0f}s)")
        if t.result and t.status != "running":
            console.print(f"    [dim]{t.result[:120]}[/dim]")


# ---------------------------------------------------------------------------
# Phase 26 — plugin bundle commands
# ---------------------------------------------------------------------------

plugins_app = typer.Typer(name="plugins", help="Manage DevAgent plugin bundles.")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list() -> None:
    """List all installed DevAgent plugin bundles."""
    from devagent.plugins import load_plugins

    bundles = load_plugins()
    if not bundles:
        console.print("[dim]No plugin bundles installed.[/dim]")
        console.print("[dim]Install one with: devagent plugins install <package>[/dim]")
        return

    table = Table(title="Installed DevAgent Plugins", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Version", style="green")
    table.add_column("Description")
    table.add_column("Skills", justify="right")
    table.add_column("Hooks", justify="right")
    table.add_column("MCP Servers", justify="right")
    table.add_column("Tools", justify="right")

    for b in bundles:
        table.add_row(
            b.name,
            b.version,
            b.description or "—",
            str(len(b.skills)),
            str(len(b.hooks)),
            str(len(b.mcp_servers)),
            str(len(b.tools)),
        )

    console.print(table)


@plugins_app.command("install")
def plugins_install(
    package: str = typer.Argument(..., help="PyPI package name or local path to install."),
) -> None:
    """Install a DevAgent plugin bundle via pip."""
    from devagent.plugins import install_plugin

    console.print(f"[dim]Installing plugin: {package}[/dim]")
    success, output = install_plugin(package)
    if output:
        console.print(output)
    if success:
        console.print(f"[green]Installed {package}.[/green] Run [bold]devagent plugins list[/bold] to verify.")
    else:
        console.print(f"[red]Failed to install {package}.[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Phase 35 — devagent mcp: manage .mcp.json project server config
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(name="mcp", help="Manage MCP server configuration.", add_completion=False)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("ls")
def mcp_ls(
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
) -> None:
    """List MCP servers declared in .mcp.json for the current project."""
    from devagent.core.project import detect_project_root
    from devagent.mcp.project_config import find_mcp_json, load_mcp_json

    project_root, _ = detect_project_root(Path(project) if project else None)
    entries = load_mcp_json(project_root)

    cfg_path = find_mcp_json(project_root)
    if cfg_path:
        console.print(f"[dim]Config: {cfg_path}[/dim]\n")
    else:
        console.print(f"[dim]No .mcp.json found in {project_root}[/dim]")
        console.print("[dim]Create one with: devagent mcp add <name> <command> [args...][/dim]")
        return

    if not entries:
        console.print("[dim]No servers declared in .mcp.json[/dim]")
        return

    table = Table(title="Project MCP Servers", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Args")
    table.add_column("Env keys")

    for e in entries:
        args_str = " ".join(e.args) if e.args else "—"
        env_str = ", ".join(e.env.keys()) if e.env else "—"
        table.add_row(e.name, e.command, args_str, env_str)

    console.print(table)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Server name (unique key in .mcp.json)"),
    command: str = typer.Argument(..., help="Command to launch the server (e.g. python, npx)"),
    args: list[str] = typer.Argument(default=None, help="Arguments to pass to the command"),  # noqa: B008
    env: list[str] = typer.Option(  # noqa: B008
        [], "--env", "-e", help="Env var to pass: KEY=VALUE (repeatable)"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
) -> None:
    """Add or update a server entry in .mcp.json.

    Example:
      devagent mcp add my-tool python -m my_tool.server
      devagent mcp add brave npx -y @modelcontextprotocol/server-brave-search --env BRAVE_API_KEY=xxx
    """
    from devagent.core.project import detect_project_root
    from devagent.mcp.project_config import MCPServerEntry, save_mcp_json

    project_root, _ = detect_project_root(Path(project) if project else None)

    # Parse --env KEY=VALUE pairs
    env_dict: dict[str, str] = {}
    for pair in env:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_dict[k.strip()] = v.strip()
        else:
            console.print(f"[yellow]Ignored malformed env var (expected KEY=VALUE): {pair}[/yellow]")

    entry = MCPServerEntry(
        name=name,
        command=command,
        args=list(args or []),
        env=env_dict,
    )
    path = save_mcp_json(project_root, [entry])
    console.print(f"[green]Added server [bold]{name}[/bold] to {path}[/green]")
    console.print(f"[dim]  command: {command} {' '.join(entry.args)}[/dim]")
    if env_dict:
        console.print(f"[dim]  env: {', '.join(env_dict.keys())}[/dim]")


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project path"),
) -> None:
    """Remove a server entry from .mcp.json."""
    import json as _json

    from devagent.core.project import detect_project_root
    from devagent.mcp.project_config import _CANDIDATES

    project_root, _ = detect_project_root(Path(project) if project else None)
    root = Path(project_root)

    for candidate in _CANDIDATES:
        path = root / candidate
        if path.exists():
            try:
                data = _json.loads(path.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError):
                console.print(f"[red]Could not parse {path}[/red]")
                raise typer.Exit(1)
            servers = data.get("mcpServers") or {}
            if name not in servers:
                console.print(f"[yellow]Server '{name}' not found in {path}[/yellow]")
                raise typer.Exit(1)
            del servers[name]
            data["mcpServers"] = servers
            path.write_text(_json.dumps(data, indent=2) + "\n", encoding="utf-8")
            console.print(f"[green]Removed server [bold]{name}[/bold] from {path}[/green]")
            return

    console.print(f"[yellow]No .mcp.json found in {project_root}[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
