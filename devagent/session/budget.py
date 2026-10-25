"""Token budget tracker with USD cost estimation.

Tracks per-model token usage and estimates cost using a built-in price
table. Ollama (local) is free — cost is always 0.0 for local models.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Price table  (USD per 1M tokens: input / output)
# Approximate as of mid-2025.
# ---------------------------------------------------------------------------
_PRICES: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-4-8":          (15.0,  75.0),
    "anthropic:claude-opus-4-7":          (15.0,  75.0),
    "anthropic:claude-sonnet-4-6":        (3.0,   15.0),
    "anthropic:claude-sonnet-4-5":        (3.0,   15.0),
    "anthropic:claude-haiku-4-5-20251001":(0.25,  1.25),
    "anthropic:claude-3-5-haiku-20241022":(0.8,   4.0),
    "openai:gpt-4o":                      (2.5,   10.0),
    "openai:gpt-4o-mini":                 (0.15,  0.60),
    "openai:o1":                          (15.0,  60.0),
    "openai:o3-mini":                     (1.1,   4.4),
    "groq:llama-3.3-70b-versatile":       (0.59,  0.79),
    "groq:llama-3.1-8b-instant":          (0.05,  0.08),
    "gemini:gemini-1.5-flash":            (0.075, 0.30),
    "gemini:gemini-1.5-pro":              (3.5,   10.5),
    "gemini:gemini-2.0-flash":            (0.10,  0.40),
    # Ollama is local/free — not in table → cost 0.0
}


def _cost_usd(provider: str, model: str, in_tok: int, out_tok: int) -> float:
    prices = _PRICES.get(f"{provider}:{model}")
    if not prices:
        return 0.0
    return (in_tok / 1_000_000) * prices[0] + (out_tok / 1_000_000) * prices[1]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when an LLM call would exceed the session token budget."""


# ---------------------------------------------------------------------------
# Per-model accumulator
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return _cost_usd(self.provider, self.model, self.input_tokens, self.output_tokens)


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Tracks token and USD usage across all LLM calls in a session."""
    max_tokens: int | None = None
    warn_at_percent: int = 80

    def __post_init__(self) -> None:
        self._per_model: dict[str, ModelUsage] = {}
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Accumulate
    # ------------------------------------------------------------------

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str = "ollama",
        model: str = "unknown",
    ) -> None:
        key = f"{provider}:{model}"
        if key not in self._per_model:
            self._per_model[key] = ModelUsage(provider=provider, model=model)
        rec = self._per_model[key]
        rec.input_tokens += input_tokens
        rec.output_tokens += output_tokens
        rec.calls += 1
        self._call_count += 1

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._per_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._per_model.values())

    @property
    def total_used(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Backward-compatible alias for the aggregate token count."""
        return self.total_used

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._per_model.values())

    @property
    def remaining(self) -> int | None:
        return max(0, self.max_tokens - self.total_used) if self.max_tokens else None

    @property
    def is_exhausted(self) -> bool:
        return bool(self.max_tokens and self.total_used >= self.max_tokens)

    @property
    def warn_threshold(self) -> float | None:
        return self.warn_at_percent / 100.0 if self.max_tokens else None

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def check(self) -> None:
        if self.max_tokens and self.total_used >= self.max_tokens:
            raise BudgetExceeded(
                f"Token budget exceeded: {self.total_used:,} used of {self.max_tokens:,}"
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_used,
            "llm_calls": self._call_count,
            "max_tokens": self.max_tokens,
            "remaining": self.remaining,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    def per_model_summary(self) -> list[dict]:
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "calls": r.calls,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd, 6),
            }
            for r in self._per_model.values()
        ]

    def status_line(self) -> str:
        """Compact one-liner for the terminal status bar."""
        tok = self.total_used
        cost = self.total_cost_usd
        parts = [f"tokens: {tok:,}"]
        if self.max_tokens:
            pct = tok / self.max_tokens * 100
            parts[0] += f" / {self.max_tokens:,} ({pct:.0f}%)"
        if cost > 0:
            parts.append(f"cost: ${cost:.4f}")
        parts.append(f"calls: {self._call_count}")
        return "  |  ".join(parts)
