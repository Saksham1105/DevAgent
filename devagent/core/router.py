"""Multi-model router — selects the right LLM for each agent iteration.

Router logic (matches roadmap Phase 5 spec, wired in Phase 1 fix):
  - Iteration 1          → planning model  (needs the full picture)
  - Last calls were writes → coding model   (precision matters)
  - Last calls were reads  → cheap model    (just exploring)
  - Last calls were impact/review → reviewing model
  - Otherwise            → fallback model
"""

from __future__ import annotations

from devagent.core.config import DevAgentConfig, LLMConfig
from devagent.core.llm import LLMClient

# Tool name sets for task detection
_WRITE_TOOLS   = {"write_file", "edit_file"}
_READ_TOOLS    = {
    "read_file", "grep", "find_files", "list_files",
    "cp_get_context", "cp_search_symbol", "cp_get_module_summary",
    "cp_get_callers", "cp_get_callees", "cp_get_file_map", "cp_get_dependencies",
}
_REVIEW_TOOLS  = {"cp_get_impact", "cp_get_data_flow", "git_diff", "git_show"}


class MultiModelRouter:
    """Routes each LLM call to the model best suited for that task type."""

    def __init__(self, config: DevAgentConfig) -> None:
        self._config = config
        self._cache: dict[str, LLMClient] = {}
        self._override: LLMConfig | None = None

    def set_override(self, llm_config: LLMConfig) -> None:
        """Use one live model for every routed iteration.

        Interactive model changes must take precedence over static router tiers.
        Copy the config so later mutations cannot leave a cached client stale.
        """
        self._override = llm_config.model_copy(deep=True)
        self._cache.clear()

    def clear_override(self) -> None:
        """Return to the configured per-task router tiers."""
        self._override = None
        self._cache.clear()

    def detect_task(self, last_tool_names: list[str], iteration: int) -> str:
        """Return a task key: planning | coding | reviewing | cheap | fallback."""
        if iteration == 1:
            return "planning"
        names = set(last_tool_names)
        if names & _WRITE_TOOLS:
            return "coding"
        if names & _REVIEW_TOOLS:
            return "reviewing"
        if names & _READ_TOOLS:
            return "cheap"
        return "fallback"

    def get_llm(self, task: str = "fallback") -> LLMClient:
        if self._override is not None:
            if "__override__" not in self._cache:
                self._cache["__override__"] = LLMClient(self._override)
            return self._cache["__override__"]
        if task not in self._cache:
            from devagent.core.llm import get_llm_for_task
            self._cache[task] = get_llm_for_task(self._config, task)
        return self._cache[task]

    def get_llm_for_iteration(
        self,
        last_tool_names: list[str],
        iteration: int,
    ) -> tuple[LLMClient, str]:
        """Return (LLMClient, task_name) for the current loop iteration."""
        task = self.detect_task(last_tool_names, iteration)
        return self.get_llm(task), task
