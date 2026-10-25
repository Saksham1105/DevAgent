# DevAgent Leaderboard

_Last updated: 2026-10-25_

Scores are from `devagent bench native --live` runs against the full task set.
Pass = oracle exits 0 after the agent loop. Higher is better.

| Rank | Model | Provider | Best Score | Latest Score | Latest Run | Runs | Avg Time |
|------|-------|----------|------------|--------------|------------|------|----------|
| 1 | `unknown` | unknown | **4/24 (17%)** | 4/24 (17%) | 2026-10-25 | 68 | 0.5s |

---

To add your model to this leaderboard:

```bash
devagent bench native --live --model <your-model> --provider <ollama|anthropic|openai>
```

Results are saved automatically to `benchmarks/results/` and picked up next time
`devagent bench leaderboard` runs.
