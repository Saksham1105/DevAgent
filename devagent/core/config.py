"""Configuration loading and saving using platformdirs + tomllib.

The config file is a TOML file stored at get_config_path().
All config models are Pydantic models with sensible defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ValidationError

from devagent.core.storage import get_config_path


class ConfigError(RuntimeError):
    """Raised when a configuration file exists but is unreadable, malformed or invalid.

    A missing config file is not an error — defaults are used. A file that *is*
    present and cannot be turned into a :class:`DevAgentConfig` is, because the
    alternative is running with values the user never chose.
    """


class LLMFallbackConfig(BaseModel):
    """Optional fallback LLM provider configuration."""
    provider: str = ""
    model: str = ""
    api_key: str = ""


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    api_key: str = ""
    fallback: LLMFallbackConfig | None = None
    # Phase 15 — effort levels and extended thinking
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    extended_thinking: bool = False
    thinking_budget_tokens: int = 10_000


class GitHubConfig(BaseModel):
    """GitHub integration configuration."""
    token: str = ""
    default_repo: str = ""


class BraveConfig(BaseModel):
    """Brave Search API configuration."""
    api_key: str = ""


class SearchXConfig(BaseModel):
    """SearchX API configuration (SearXNG-compatible self-hosted metasearch)."""
    api_key: str = ""
    base_url: str = "http://localhost:8888"


class OutputConfig(BaseModel):
    """Output formatting preferences."""
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"


class WatcherConfig(BaseModel):
    """Repo Health Monitor configuration."""
    default_interval_minutes: int = 30
    max_issues_per_check: int = 20
    default_labels: list[str] = []
    notify_on_cross_conflict: bool = True
    skip_closed_issues: bool = True


# ---------------------------------------------------------------------------
# Phase 1+ — Agent harness config sections
# ---------------------------------------------------------------------------

class RouterConfig(BaseModel):
    """Multi-model routing: maps task type → (provider, model) pair."""
    planning: dict[str, str] = {"provider": "ollama", "model": "qwen2.5-coder:14b"}
    coding: dict[str, str] = {"provider": "ollama", "model": "qwen2.5-coder:7b"}
    reviewing: dict[str, str] = {"provider": "ollama", "model": "qwen2.5-coder:7b"}
    cheap: dict[str, str] = {"provider": "ollama", "model": "qwen2.5-coder:3b"}
    fallback: dict[str, str] = {"provider": "ollama", "model": "qwen2.5-coder:7b"}


class AgentConfig(BaseModel):
    """Core agent loop configuration."""
    max_iterations: int = 50
    max_repair_iterations: int = 3
    stream_thoughts: bool = True
    confirmation_required: bool = True
    auto_run_tests: bool = True
    loop_detection: bool = True       # detect repeated identical tool calls and bail early
    shell_output_cap_kb: int = 0      # 0 = unlimited; stream large output to temp file
    shell_timeout_sec: int = 300      # per-command timeout in seconds (0 = no timeout)


class SessionConfig(BaseModel):
    """Session persistence configuration."""
    auto_resume: bool = True
    max_sessions: int = 20
    # Phase 8 — context auto-compression
    auto_compress: bool = True
    compression_threshold: float = 0.45   # compress before history dominates the request
    compression_window_size: int = 20     # keep this many most-recent events verbatim
    compression_model: str = "cheap"      # router tier used for summarisation LLM calls


class SecurityConfig(BaseModel):
    """Security Gate configuration."""
    gate_enabled: bool = True
    block_on_secrets: bool = True
    warn_on_weak_crypto: bool = True
    check_new_dependencies: bool = True


class TokenBudgetConfig(BaseModel):
    """Token budget and cost tracking."""
    session_cap_usd: float = 0.0
    warn_at_percent: int = 80
    track_by_model: bool = True


class CodePrismConfig(BaseModel):
    """CodePrism knowledge graph integration."""
    auto_index: bool = True
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_port: int = 8765


class UIConfig(BaseModel):
    """Terminal UI preferences (Phase 36)."""
    theme: Literal["default", "dracula", "monokai", "solarized"] = "default"


class DevAgentConfig(BaseModel):
    """Root configuration model for DevAgent."""
    llm: LLMConfig = LLMConfig()
    github: GitHubConfig = GitHubConfig()
    brave: BraveConfig = BraveConfig()
    searchx: SearchXConfig = SearchXConfig()
    search_provider: Literal["brave", "searchx"] = "searchx"
    output: OutputConfig = OutputConfig()
    watcher: WatcherConfig = WatcherConfig()
    # Phase 1+ agent harness config
    router: RouterConfig = RouterConfig()
    agent: AgentConfig = AgentConfig()
    session: SessionConfig = SessionConfig()
    security: SecurityConfig = SecurityConfig()
    budget: TokenBudgetConfig = TokenBudgetConfig()
    codeprism: CodePrismConfig = CodePrismConfig()
    ui: UIConfig = UIConfig()


def config_exists() -> bool:
    """Check if the user-level config file exists."""
    return get_config_path().is_file()


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (override wins on conflicts)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config(project_root: str | None = None) -> DevAgentConfig:
    """Load config merging four levels (lowest to highest priority):

      1. Built-in defaults (DevAgentConfig field defaults)
      2. User config      (~/.config/devagent/settings.toml)
      3. Project local    (<root>/.devagent/settings.local.toml)  [gitignored]
      4. Project config   (<root>/.devagent/settings.toml)        [committed]

    Returns defaults when no files exist.

    Raises :class:`ConfigError` when a file *is* present but cannot be parsed or
    does not validate — a rejected config file must not silently become defaults.
    """
    merged: dict = {}
    sources: list[Path] = []

    def _merge_file(path: Path) -> None:
        try:
            with open(path, "rb") as f:
                _deep_merge(merged, tomllib.load(f))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
        sources.append(path)

    # Level 2: user config
    user_path = get_config_path()
    if user_path.is_file():
        _merge_file(user_path)

    if project_root:
        root = Path(project_root)

        # Level 3: project local (gitignored secrets / overrides)
        local_path = root / ".devagent" / "settings.local.toml"
        if local_path.is_file():
            _merge_file(local_path)

        # Level 4 (highest): committed project settings
        project_path = root / ".devagent" / "settings.toml"
        if project_path.is_file():
            _merge_file(project_path)

    try:
        return DevAgentConfig(**merged)
    except ValidationError as exc:
        where = ", ".join(str(p) for p in sources) if sources else "the built-in defaults"
        raise ConfigError(f"invalid configuration in {where}:\n{exc}") from exc


def save_config(
    config: DevAgentConfig,
    project_root: str | None = None,
    scope: str = "user",
) -> None:
    """Write DevAgentConfig to the appropriate config file.

    scope:
      "user"    — ~/.config/devagent/settings.toml  (default)
      "project" — <project_root>/.devagent/settings.toml
      "local"   — <project_root>/.devagent/settings.local.toml
    """
    from pathlib import Path as _Path

    if scope == "project" and project_root:
        config_path = _Path(project_root) / ".devagent" / "settings.toml"
    elif scope == "local" and project_root:
        config_path = _Path(project_root) / ".devagent" / "settings.local.toml"
    else:
        config_path = get_config_path()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(exclude_none=True)

    # Remove empty fallback section to keep config clean
    if "llm" in data and "fallback" in data["llm"]:
        fallback = data["llm"]["fallback"]
        if not fallback.get("provider"):
            del data["llm"]["fallback"]

    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)
