"""Tests for TokenBudget with USD cost tracking."""

import pytest

from devagent.session.budget import BudgetExceeded, TokenBudget


def test_record_accumulates_tokens():
    b = TokenBudget()
    b.record(100, 50, provider="ollama", model="qwen2.5-coder:7b")
    assert b.input_tokens == 100
    assert b.output_tokens == 50
    assert b.total_used == 150
    assert b.call_count == 1


def test_record_multiple_calls_same_model():
    b = TokenBudget()
    b.record(100, 50, "openai", "gpt-4o")
    b.record(200, 80, "openai", "gpt-4o")
    assert b.input_tokens == 300
    assert b.output_tokens == 130
    assert b.call_count == 2


def test_record_multiple_providers():
    b = TokenBudget()
    b.record(500, 100, "anthropic", "claude-sonnet-4-6")
    b.record(300, 80, "openai", "gpt-4o")
    assert b.total_used == 980
    assert len(b.per_model_summary()) == 2


def test_cost_zero_for_ollama():
    b = TokenBudget()
    b.record(10000, 5000, "ollama", "qwen2.5-coder:7b")
    assert b.total_cost_usd == 0.0


def test_cost_nonzero_for_anthropic():
    b = TokenBudget()
    b.record(1_000_000, 1_000_000, "anthropic", "claude-sonnet-4-6")
    # $3.0 / 1M input + $15.0 / 1M output = $18.0
    assert abs(b.total_cost_usd - 18.0) < 0.01


def test_cost_nonzero_for_openai():
    b = TokenBudget()
    b.record(1_000_000, 1_000_000, "openai", "gpt-4o")
    # $2.5 / 1M input + $10.0 / 1M output = $12.5
    assert abs(b.total_cost_usd - 12.5) < 0.01


def test_unknown_model_cost_zero():
    b = TokenBudget()
    b.record(500000, 200000, "somevendor", "mystery-model")
    assert b.total_cost_usd == 0.0


def test_status_line_no_budget():
    b = TokenBudget()
    b.record(1000, 500, "ollama", "llama3")
    line = b.status_line()
    assert "tokens: 1,500" in line
    assert "calls: 1" in line
    # Ollama is free — cost not shown
    assert "cost" not in line


def test_status_line_with_cost():
    b = TokenBudget()
    b.record(500000, 100000, "anthropic", "claude-sonnet-4-6")
    line = b.status_line()
    assert "cost: $" in line
    assert "calls: 1" in line


def test_status_line_with_budget():
    b = TokenBudget(max_tokens=10000)
    b.record(3000, 2000, "ollama", "x")
    line = b.status_line()
    assert "/ 10,000" in line
    assert "50%" in line


def test_warn_threshold():
    b = TokenBudget(max_tokens=10000, warn_at_percent=80)
    assert b.warn_threshold == 0.80


def test_warn_threshold_none_when_no_limit():
    b = TokenBudget()
    assert b.warn_threshold is None


def test_remaining_with_budget():
    b = TokenBudget(max_tokens=1000)
    b.record(300, 200, "ollama", "x")
    assert b.remaining == 500


def test_remaining_none_without_limit():
    b = TokenBudget()
    assert b.remaining is None


def test_is_exhausted_false():
    b = TokenBudget(max_tokens=1000)
    b.record(100, 100, "ollama", "x")
    assert not b.is_exhausted


def test_is_exhausted_true():
    b = TokenBudget(max_tokens=100)
    b.record(60, 60, "ollama", "x")
    assert b.is_exhausted


def test_check_raises_when_exhausted():
    b = TokenBudget(max_tokens=100)
    b.record(60, 60, "ollama", "x")
    with pytest.raises(BudgetExceeded):
        b.check()


def test_check_passes_when_under_budget():
    b = TokenBudget(max_tokens=1000)
    b.record(100, 50, "ollama", "x")
    b.check()  # should not raise


def test_summary_keys():
    b = TokenBudget(max_tokens=5000)
    b.record(200, 100, "ollama", "x")
    s = b.summary()
    assert s["total_tokens"] == 300
    assert s["input_tokens"] == 200
    assert s["output_tokens"] == 100
    assert s["llm_calls"] == 1
    assert s["max_tokens"] == 5000
    assert s["remaining"] == 4700
    assert "total_cost_usd" in s


def test_per_model_summary_structure():
    b = TokenBudget()
    b.record(100, 50, "anthropic", "claude-sonnet-4-6")
    b.record(200, 80, "openai", "gpt-4o")
    rows = b.per_model_summary()
    assert len(rows) == 2
    keys = {"provider", "model", "calls", "input_tokens", "output_tokens", "cost_usd"}
    for row in rows:
        assert keys.issubset(row.keys())


def test_default_provider_model_args():
    b = TokenBudget()
    b.record(100, 50)  # use defaults: provider="ollama", model="unknown"
    assert b.total_used == 150
    assert b.total_cost_usd == 0.0  # ollama default


def test_total_tokens_alias_matches_total_used():
    budget = TokenBudget()
    budget.record(120, 30)
    assert budget.total_tokens == budget.total_used == 150
