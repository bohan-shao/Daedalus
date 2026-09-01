# KuaiRand-1k — Results Summary (Bonus Benchmark)

> Autonomous ML Research Agent for Recommender Systems — bonus benchmark KuaiRand-1k.
> Same task & metrics as KuaiRand-Pure: predict `long_view`, rank within each user's logged
> impressions, scored by **GAUC / nDCG@5**, primary = mean(GAUC, nDCG@5).

## 1. Final result

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation | 0.7103 | 0.8235 | 0.7669 |
| **test** | **0.7148** | **0.8462** | **0.7805** |

## 2. Delta over the official baseline

| metric | baseline (test) | agent (test) | delta |
|---|---|---|---|
| GAUC | 0.6730 | 0.7148 | **+0.0418** |
| nDCG@5 | 0.6049 | 0.8462 | **+0.2413** |
| primary | 0.6390 | 0.7805 | **+0.1415** |

**Bonus score** (per the Judging Criteria formula) =
`mean over m of delta(m)` = `(0.0418 + 0.2413) / 2` = **+0.1416**.

For reference, the validation-side deltas are GAUC +0.0354 / nDCG@5 +0.2082 / primary +0.1218.

## 3. Approach (what the agent discovered autonomously)

The agent started from the LightGBM LambdaRank reference (the Pure-validated recipe) and, on the
larger 1k dataset, kept a LambdaRank objective and enriched the feature set with:

- **Time-decayed historical statistics** — per `user_id` / `video_id` / `author_id` / `tab` and
  their pairs, an exponentially decayed running rate & count of past `long_view` (half-life 2 days),
  so recent impressions carry more weight.
- **Click-history statistics** — past `is_click` rate & count (used strictly as *past-only*
  auxiliary history, never the current impression's feedback).
- **Recent-window features** — the last ~20 impressions' positive rate, click rate, and activity gap.
- **Rank / ratio features** — within-user percentile ranks of the historical rates, per-video
  duration-vs-user-average ratio, and time-of-day / weekend features.

Label-leakage guard: `is_click` and `long_view` are only used via the "read history → then append
the current row" pattern; the current impression's own feedback is never used as a feature.

Training: LightGBM `lambdarank`, capped to the most recent ≤10,000 rows per user (1k has ~983 users
with thousands of impressions each, hitting LightGBM's 10k-rows-per-query limit), early stopping on
the full validation split.

## 4. Resource usage (to reach the converged result)

| item | value |
|---|---|
| iterations | 26 search nodes (5 draft + 11 improve + 11 debug); the run hit a child-process deadlock before reaching the 50-iteration cap — see note below |
| agent wall-clock | ≈ 2 h 45 m (00:17 → 03:02) |
| LLM tokens | ≈ 0.5 M (estimate; authoritative figure on the DeepSeek usage dashboard) |
| GPU-hours | 0 (CPU/MPS only) |
| manual interventions | 0 during the run |

Note on robustness: the run reached its best (0.7669) at ~01:59 and then stalled at 03:02 when a
low-efficiency feature-loop (pure-Python `sum` over 5M rows) timed out and its child process could
not be reaped. This deadlock was later fixed in the interpreter (bounded EOF wait), so subsequent
runs terminate cleanly on overtime.

## 5. Reproduce

```bash
cd kuairand-1k-mlmaster
# make the input/ dir the script expects (1k logs + video features + official helpers)
mkdir -p input
cp data/log_standard_4_08_to_4_21_1k.csv data/log_standard_4_22_to_5_08_1k.csv input/
cp data/video_features_basic_1k.csv input/
cp data/evaluate.py data/data.py input/

python final_submission_1k.py
# → prints validation primary ≈ 0.7669, writes submission/submission.csv with row order aligned
#   to the official data.load()["test"] split
```

Validate the submission (row alignment verified against the official loader):

```bash
python submit.py --check submission/submission.csv --data_dir <path>/KuaiRand-1K/data --split test
```

## 6. Team cross-check

A parallel GPU run by a teammate (LightGBM LambdaRank with simpler cumsum history features) reached
test GAUC 0.6282 / nDCG@5 0.5881 / primary **0.6082** — below the 1k baseline (0.6390). The final
bonus submission therefore uses this (higher) result, test primary **0.7805**.
