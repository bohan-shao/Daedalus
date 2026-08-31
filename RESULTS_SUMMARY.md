# KuaiRand-Pure — Run & Results Summary

**Agent framework**: ML-Master (MCTS tree search + code interpreter + LLM feedback)
**LLM**: DeepSeek v4-pro (code) / DeepSeek v4-pro (feedback)
**Run**: `kuairand_afi` (final converged run)

---

## 1. Final results (validation-best, and delta over the official baseline)

Validation-best (the metric the agent optimizes during development):

| metric | official baseline (val) | agent (val) | delta |
|---|---|---|---|
| GAUC | 0.6674 | **0.6899** | **+0.0225** |
| nDCG@5 | 0.5357 | **0.5474** | **+0.0117** |
| **primary** | 0.6016 | **0.6186** | **+0.0170** |

Independent test-split check (local, labels visible — for reference; the official
hidden-test score is computed once by the organizers):

| metric | official baseline (test) | agent (test) | delta |
|---|---|---|---|
| GAUC | 0.6610 | **0.6817** | +0.0207 |
| nDCG@5 | 0.5282 | **0.5423** | +0.0141 |
| **primary** | 0.5946 | **0.6120** | **+0.0174** |

The validation-best **primary 0.6186** is the converged score (see §3).

## 2. Resource usage (for Feasibility & Practicality scoring)

| resource | value |
|---|---|
| **Iterations used** | **50 / 50** (hit the iteration cap; the ε=0.002 / N=3 convergence rule would have triggered earlier — see §3) |
| **Agent wall-clock** | **≈ 2 h 10 min** (start 20:30:49 → last node 22:40:27). The best score was found at 20:35:35 (≈ 5 min in); the remaining time was spent exploring 0.61-range variants that did not beat 0.6186. |
| **GPU-hours** | **0** (ran on CPU / Apple MPS; no GPU used — this benchmark is CPU-bound by design) |
| **LLM tokens** | **≈ 0.96 M total** — input ≈ 0.78 M + output ≈ 0.19 M (from the DeepSeek usage dashboard) |

> **Token note**: taken from the DeepSeek usage dashboard (platform.deepseek.com) for
> 2026-08-30, `deepseek-v4-pro`. The 20:00–21:00 hour also contained a short earlier run
> (`kuairand_din3`, 37 code calls), so that hour was prorated by request count (33/107) to
> isolate this run. Estimated LLM cost ≈ 4.56 CNY.

## 3. Convergence

- The best solution (LightGBM LambdaRank + past-only historical features, val primary 0.6186)
  was first reached at **20:35:35**, ~5 minutes / ~3 iterations after the run started.
- It was **never exceeded** across the remaining ~47 iterations (scores stayed in the
  0.60–0.615 band), so the ε = 0.002 / N = 3 convergence rule is satisfied well before the
  50-iteration cap. The framework ran to the cap because it does not implement an automatic
  early-stop on this rule; the *validation-best* checkpoint is what is submitted.

## 4. Autonomy / manual interventions

- **Manual interventions during the run: 0.** The agent ran all 50 iterations autonomously —
  it proposed hypotheses, wrote code, executed, evaluated, and revised on its own.
- Robustness events handled autonomously: **21 nodes were marked buggy** (code errors / failed
  submissions), and the agent recovered via **19 debug steps**, plus 5 fresh drafts and
  26 improve steps, without external help.

## 5. The winning solution (what the agent converged on)

- **Model**: LightGBM, `objective=lambdarank` (listwise ranking loss), `num_leaves=63`,
  `lr=0.05`, ~200 boost rounds with early-stopping on valid nDCG@5.
- **Features** (past-only, no leakage): per-entity (`user_id`, `video_id`, `author_id`, `tab`)
  cumulative *before-current-row* statistics — exposure count, historical `long_view` rate,
  historical `is_click` rate — plus `duration_ms`, `hour` (from `hourmin`), and `is_weekend`.
- Runtime ≈ 22 s per full train+valid+test pass.

## 6. Reproduce

```bash
cd <repo>
python3 final_submission.py   # reads input/, writes submission/submission.csv
```
The final submission file follows the required schema: `row_id,user_id,video_id,score`,
one row per test-set row, aligned to `data.load()['test']` order.
