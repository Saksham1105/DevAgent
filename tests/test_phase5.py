"""Phase 5 tests: router caching, memory tools, repair loop, offline doctor."""

from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. MultiModelRouter — LLM client caching
# ---------------------------------------------------------------------------

def _make_config(provider="ollama", model="qwen2.5-coder:7b", api_key=""):
    from devagent.core.config import DevAgentConfig, LLMConfig, RouterConfig
    return DevAgentConfig(
        llm=LLMConfig(provider=provider, model=model, api_key=api_key),
        router=RouterConfig(
            planning={"provider": "ollama", "model": "qwen2.5-coder:14b"},
            coding={"provider": "ollama", "model": "qwen2.5-coder:7b"},
            reviewing={"provider": "ollama", "model": "qwen2.5-coder:7b"},
            cheap={"provider": "ollama", "model": "qwen2.5-coder:3b"},
            fallback={"provider": "ollama", "model": "qwen2.5-coder:7b"},
        ),
    )


def test_router_caches_llm_clients():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    client1 = router.get_llm("planning")
    client2 = router.get_llm("planning")
    assert client1 is client2, "Router must return the same LLMClient instance on repeated calls"


def test_router_different_tasks_different_clients():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    planning_client = router.get_llm("planning")
    cheap_client = router.get_llm("cheap")
    # Different tasks → different configs → different clients
    assert planning_client is not cheap_client
    assert planning_client.cfg.model == "qwen2.5-coder:14b"
    assert cheap_client.cfg.model == "qwen2.5-coder:3b"


def test_router_runtime_override_replaces_cached_task_clients():
    from devagent.core.config import LLMConfig
    from devagent.core.router import MultiModelRouter

    router = MultiModelRouter(_make_config())
    router.get_llm("planning")
    router.set_override(LLMConfig(
        provider="openai",
        model="gpt-test",
        api_key="new-key",
        base_url="https://example.test/v1",
    ))

    planning, _ = router.get_llm_for_iteration([], 1)
    coding, _ = router.get_llm_for_iteration(["write_file"], 2)
    assert planning is coding
    assert planning.cfg.provider == "openai"
    assert planning.cfg.model == "gpt-test"
    assert planning.cfg.api_key == "new-key"
    assert planning.cfg.base_url == "https://example.test/v1"


def test_router_detect_task_iteration_1():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    assert router.detect_task([], 1) == "planning"


def test_router_detect_task_write_tools():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    assert router.detect_task(["write_file"], 2) == "coding"
    assert router.detect_task(["edit_file"], 3) == "coding"


def test_router_detect_task_read_tools():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    assert router.detect_task(["read_file", "grep"], 2) == "cheap"


def test_router_detect_task_review_tools():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    assert router.detect_task(["cp_get_impact"], 2) == "reviewing"


def test_router_detect_task_fallback():
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    assert router.detect_task(["run_shell"], 5) == "fallback"


# ---------------------------------------------------------------------------
# 2. Memory tools — remember_fact, recall_facts, forget_fact
# ---------------------------------------------------------------------------

def _make_memory_block():
    """Create an in-memory MemoryBlock without DB backing."""
    from devagent.session.memory import MemoryBlock
    mb = MemoryBlock.__new__(MemoryBlock)
    mb.session_id = "test-session"
    mb.db_path = None
    mb._cache = {}
    mb._loaded = True

    # Stub out the DB calls
    mb.set = lambda key, value, item_type="fact": mb._cache.update({key: value})
    mb.get = lambda key, default=None: mb._cache.get(key, default)
    mb.delete = lambda key: mb._cache.pop(key, None)
    mb.all = lambda: dict(mb._cache)
    return mb


def _build_registry_with_memory():
    from devagent.tools.memory_tools import register_memory_tools
    from devagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    memory = _make_memory_block()
    register_memory_tools(registry, memory)
    return registry, memory


def test_remember_fact_stores_value():
    registry, memory = _build_registry_with_memory()
    result = registry.call("remember_fact", {"key": "auth_file", "value": "devagent/core/auth.py"})
    assert "remembered" in result
    assert memory.all().get("auth_file") == "devagent/core/auth.py"


def test_remember_fact_missing_key():
    registry, _ = _build_registry_with_memory()
    result = registry.call("remember_fact", {"value": "some value"})
    assert "[error]" in result


def test_remember_fact_missing_value():
    registry, _ = _build_registry_with_memory()
    result = registry.call("remember_fact", {"key": "mykey"})
    assert "[error]" in result


def test_recall_facts_empty():
    registry, _ = _build_registry_with_memory()
    result = registry.call("recall_facts", {})
    assert "No facts" in result


def test_recall_facts_returns_all():
    registry, _ = _build_registry_with_memory()
    registry.call("remember_fact", {"key": "file_a", "value": "path/to/a.py"})
    registry.call("remember_fact", {"key": "file_b", "value": "path/to/b.py"})
    result = registry.call("recall_facts", {})
    assert "file_a" in result
    assert "file_b" in result
    assert "path/to/a.py" in result


def test_forget_fact_removes_key():
    registry, memory = _build_registry_with_memory()
    registry.call("remember_fact", {"key": "temp", "value": "remove me"})
    assert "temp" in memory.all()
    result = registry.call("forget_fact", {"key": "temp"})
    assert "forgotten" in result
    assert "temp" not in memory.all()


def test_forget_fact_missing_key():
    registry, _ = _build_registry_with_memory()
    result = registry.call("forget_fact", {})
    assert "[error]" in result


def test_memory_tools_registered_in_registry():
    registry, _ = _build_registry_with_memory()
    names = registry.names()
    assert "remember_fact" in names
    assert "recall_facts" in names
    assert "forget_fact" in names


# ---------------------------------------------------------------------------
# 3. Auto-test repair loop in AgentLoop
# ---------------------------------------------------------------------------

def _make_loop_with_mocked_cp(module_summary=None, test_output="1 passed"):
    """Build an AgentLoop stub for testing the repair method."""
    from devagent.agent.loop import MAX_REPAIR, AgentLoop
    from devagent.session.budget import TokenBudget
    from devagent.tools.registry import ToolRegistry

    cp = MagicMock()
    cp.get_module_summary.return_value = module_summary or {"test_coverage_file": "tests/test_foo.py"}

    registry = ToolRegistry()
    registry.register(
        "run_shell",
        "Run a shell command",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        lambda args: test_output,
    )

    loop = AgentLoop.__new__(AgentLoop)
    loop.llm = MagicMock()
    loop.registry = registry
    loop.session_mgr = MagicMock()
    loop.session_id = "test"
    loop.memory = MagicMock()
    loop.budget = TokenBudget()
    loop.system_prompt = ""
    loop._cp_client = cp
    loop._router = None
    loop._repair_attempts = {}
    loop._max_repair = MAX_REPAIR
    return loop


def test_auto_test_passes_returns_success_note():
    loop = _make_loop_with_mocked_cp(test_output="1 passed in 0.5s")
    note = loop._auto_test_after_write("src/foo.py")
    assert "all tests pass" in note
    assert loop._repair_attempts.get("src/foo.py") == 0


def test_auto_test_fails_increments_counter():
    loop = _make_loop_with_mocked_cp(test_output="1 failed in 0.5s\nAssertionError: ...")
    note = loop._auto_test_after_write("src/foo.py")
    assert "Tests failed" in note
    assert "attempt 1/3" in note
    assert loop._repair_attempts.get("src/foo.py") == 1


def test_auto_test_max_repair_stops():
    from devagent.agent.loop import MAX_REPAIR
    loop = _make_loop_with_mocked_cp(test_output="1 failed")
    loop._repair_attempts["src/foo.py"] = MAX_REPAIR
    note = loop._auto_test_after_write("src/foo.py")
    assert note == ""  # silenced at limit


def test_auto_test_no_test_file_returns_empty():
    loop = _make_loop_with_mocked_cp(module_summary={"test_coverage_file": ""})
    note = loop._auto_test_after_write("src/foo.py")
    assert note == ""


def test_auto_test_no_cp_client_returns_empty():
    loop = _make_loop_with_mocked_cp()
    loop._cp_client = None
    note = loop._auto_test_after_write("src/foo.py")
    assert note == ""


def test_auto_test_empty_path_returns_empty():
    loop = _make_loop_with_mocked_cp()
    note = loop._auto_test_after_write("")
    assert note == ""


def test_auto_test_resets_counter_on_success():
    loop = _make_loop_with_mocked_cp(test_output="1 passed")
    loop._repair_attempts["src/foo.py"] = 2  # simulate prior failures
    note = loop._auto_test_after_write("src/foo.py")
    assert "all tests pass" in note
    assert loop._repair_attempts.get("src/foo.py") == 0


def test_auto_test_max_repair_warning():
    loop = _make_loop_with_mocked_cp(test_output="2 failed")
    loop._repair_attempts["src/foo.py"] = 2  # one away from limit
    note = loop._auto_test_after_write("src/foo.py")
    assert "WARNING" in note or "Max repair" in note
    assert loop._repair_attempts.get("src/foo.py") == 3


def test_agent_loop_stores_configured_max_repair():
    """agent.max_repair_iterations must reach the loop instead of the constant."""
    from devagent.agent.loop import MAX_REPAIR, AgentLoop
    from devagent.core.config import DevAgentConfig
    from devagent.session.budget import TokenBudget

    cfg = DevAgentConfig()

    def build(max_repair=None):
        kwargs = {}
        if max_repair is not None:
            kwargs["max_repair"] = max_repair
        return AgentLoop(
            llm=MagicMock(),
            registry=MagicMock(),
            session_mgr=MagicMock(),
            session_id="test",
            memory=MagicMock(),
            budget=TokenBudget(),
            system_prompt="",
            **kwargs,
        )

    assert build()._max_repair == MAX_REPAIR  # unchanged default
    assert build(cfg.agent.max_repair_iterations)._max_repair == cfg.agent.max_repair_iterations
    assert build(5)._max_repair == 5
    assert build(0)._max_repair == 0  # explicit 0 disables auto-repair


def test_auto_test_honours_configured_max_repair():
    """The repair loop must stop at the configured cap, not at 3."""
    loop = _make_loop_with_mocked_cp(test_output="1 failed")
    loop._max_repair = 5
    for _ in range(5):
        note = loop._auto_test_after_write("src/foo.py")
    assert loop._repair_attempts.get("src/foo.py") == 5
    assert "attempt 5/5" in note
    assert loop._auto_test_after_write("src/foo.py") == ""  # silenced at the configured cap


def test_auto_test_per_file_repair_budget_isolated():
    """Repair budget is tracked independently per file, not session-wide."""
    loop = _make_loop_with_mocked_cp(test_output="1 failed")
    # File A exhausts its repair budget (3 attempts)
    for i in range(3):
        note_a = loop._auto_test_after_write("src/file_a.py")
        assert f"attempt {i+1}/3" in note_a

    # File A is now capped
    assert loop._auto_test_after_write("src/file_a.py") == ""
    assert loop._repair_attempts["src/file_a.py"] == 3

    # File B should have its own full budget starting at attempt 1
    note_b = loop._auto_test_after_write("src/file_b.py")
    assert "attempt 1/3" in note_b
    assert loop._repair_attempts["src/file_b.py"] == 1
    assert loop._repair_attempts["src/file_a.py"] == 3

    # File C passes tests on first try -> counter is 0, does not affect A or B
    loop.registry.register(
        "run_shell",
        "Run a shell command",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        lambda args: "1 passed in 0.2s",
    )
    note_c = loop._auto_test_after_write("src/file_c.py")
    assert "all tests pass" in note_c
    assert loop._repair_attempts["src/file_c.py"] == 0
    assert loop._repair_attempts["src/file_a.py"] == 3
    assert loop._repair_attempts["src/file_b.py"] == 1


# ---------------------------------------------------------------------------
# 4. Offline capability in doctor output (CLI)
# ---------------------------------------------------------------------------

def test_router_config_inherits_api_key():
    """get_llm_for_task passes api_key from main llm config to routed config."""
    from devagent.core.llm import get_llm_for_task
    cfg = _make_config(provider="anthropic", model="claude-3-haiku", api_key="sk-test-key")
    client = get_llm_for_task(cfg, "planning")
    assert client.cfg.api_key == "sk-test-key"


def test_router_config_inherits_effort_and_thinking():
    """get_llm_for_task keeps effort / extended-thinking settings from the main llm config."""
    from devagent.core.llm import get_llm_for_task
    cfg = _make_config(provider="anthropic", model="claude-3-haiku", api_key="sk-test-key")
    cfg.llm.effort = "max"
    cfg.llm.extended_thinking = True
    cfg.llm.thinking_budget_tokens = 4_000
    client = get_llm_for_task(cfg, "coding")
    assert client.cfg.effort == "max"
    assert client.cfg.extended_thinking is True
    assert client.cfg.thinking_budget_tokens == 4_000


def test_router_config_fallback_on_unknown_task():
    """Unknown task falls back to RouterConfig.fallback."""
    from devagent.core.router import MultiModelRouter
    cfg = _make_config()
    router = MultiModelRouter(cfg)
    client = router.get_llm("nonexistent_task")
    # Should fall back to fallback config (qwen2.5-coder:7b)
    assert client.cfg.model == cfg.router.fallback.get("model")
