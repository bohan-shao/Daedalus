# Run & Iteration Log — KuaiRand-Pure

**Experiment:** `kuairand_afi` · **Framework:** ML-Master (MCTS-driven autonomous ML agent)
**Code model:** DeepSeek `deepseek-v4-pro` · **Feedback model:** DeepSeek `deepseek-v4-pro`

---

## 1. Run summary

| Field | Value |
|---|---|
| Benchmark | KuaiRand-Pure (required) |
| Task / label | within-user ranking of `long_view` (0/1) |
| Metrics | GAUC, nDCG@5; primary = mean(GAUC, nDCG@5) |
| Iterations used | **50 / 50** (5 draft + 26 improve + 19 debug) |
| Validation best (primary) | **0.6186** (GAUC 0.6899 / nDCG@5 0.5474) |
| Official baseline (val) | 0.6016 → **delta +0.0170** |
| Hidden-test primary | **0.6120** (delta vs baseline +0.0174) |
| Agent wall-clock | ≈ 2 h 10 m (20:30:49 → 22:40:27) |
| LLM tokens | ≈ 0.96 M total (0.78 M input + 0.19 M output) |
| GPU-hours | 0 (CPU/MPS only; no GPU used) |
| Manual interventions during the run | **0** (fully autonomous) |
| Convergence | best 0.6186 found at iteration ~4 (~5 min in), never exceeded in the remaining ~46 iterations |

**Final solution:** LightGBM `LambdaRank` + past-only historical aggregate features
(user/video/author/tab exposure counts, long-view rates, click rates, time-of-day & weekend features).
Selected as the validation-best checkpoint and evaluated once on the test split.

---

## 2. How to read the trajectory

ML-Master searches a tree. Each **node** below is one agent iteration: the agent (a) states a
hypothesis, (b) writes/modifies a full training script, (c) executes it in a sandbox, (d) a
feedback model reads the execution log and returns `{is_bug, metric, summary}`, and (e) the tree
search decides the next move (draft a new idea / improve the best node / debug a buggy node).
The 50 nodes therefore map 1:1 onto the 50-iteration cap.

```
# 0  draft  → BUG   (TypeError: dim is not a dict)
# 1  debug  → BUG   (PicklingError: DataLoader worker)
# 2  debug  → 0.6022  ✅ first working solution
# 3  draft  → BUG   (KeyError: author_id not in raw CSV)
# 4  debug  → 0.6186  🏆 BEST (LightGBM LambdaRank + historical features)
# 5  draft  → BUG   (KeyError: author_id)
# 6  debug  → 0.4849 (LightGBM but near-random → wrong label/group handling)
# 7  draft  → BUG   (TypeError: field_dims = list(dim))
# 8–10 debug  → BUG  (PicklingError / RuntimeError torch.cat / RuntimeError MLP dims)
#11  draft  → BUG   (TypeError: dim is int)
#12–13 debug  → BUG  (PicklingError / IndexError embedding out of range)
#14  debug  → 0.6011 (BPR+BCE ranking model)
#15–20 improve/debug → BUG (timeout in listwise / ValueError unpack / IndexError / RuntimeError)
#21  debug  → 0.5837 (AutoInt + historical features)
#22  improve → 0.6018 (DIN+AutoInt, BPR loss)
#23  improve → 0.5994 (DIN+field-attention, listwise loss)
#24–25 improve → 0.6000 (DIN+field-attention multi-task + click head)
#26  improve → BUG   (ValueError: hourmin parse out of range)
#27  debug  → 0.5999 (AutoInt + continuous features)
#28  improve → 0.6108 (LightGBM + time-decayed user long-view rate)
#29  improve → 0.6141 (LightGBM + historical & cross features)
#30  improve → 0.6144 (LightGBM)
#31  improve → 0.6146 (LightGBM; GAUC 0.6846)
#32–34 improve → 0.5816 / 0.5733 / 0.5810 (AutoInt+sequence — regression vs #31)
#35  improve → BUG   (timeout, 2nd epoch)
#36  debug  → 0.5757 (AutoInt + sequence attention, listwise)
#37  improve → 0.6011 (DIN+AutoInt, BPR)
#38  improve → BUG   (timeout)
#39  debug  → 0.6029 (AutoInt + historical video/user lv-rate)
#40  improve → 0.6114 (LightGBM)
#41  improve → 0.6107 (LightGBM + static video popularity)
#42  improve → 0.6085 (LightGBM)
#43  improve → 0.6087 (LightGBM)
#44  improve → BUG   (TypeError: dim is int)
#45  debug  → 0.5934 (AutoInt + historical continuous features)
#46  improve → 0.5999 (AutoInt+DIN)
#47  improve → 0.5930 (AutoInt+DIN)
#48  improve → 0.5915 (AutoInt+DIN, combined BCE+listwise+BPR)
#49  improve → 0.6036 (AutoInt+DIN multi-task)
```

---

## 3. Key iterations (hypothesis → action → result)

### Iterations #0–#2 — draft: AutoInt + DIN + BPR (first working pipeline)
- **Hypothesis:** combine AutoInt's multi-head self-attention over the 5 base fields with a
  DIN-style target-attention over the user's recent video history, trained with a pairwise BPR
  ranking loss on top of BCE.
- **Action:** wrote a full torch training script (load → encode → history build → train → predict).
- **Errors → recovery:** #0 crashed (`TypeError`, treated `dim` from `encode()` as a dict — it is
  an int); #1 crashed (`PicklingError`, DataLoader worker couldn't pickle the in-`__main__`
  dataset class). The agent fixed both and #2 ran clean.
- **Result:** val primary **0.6022** (GAUC 0.6682 / nDCG@5 0.5363) — already above the FM baseline.

### Iterations #3–#4 — draft: LightGBM LambdaRank + historical features (BEST)
- **Hypothesis:** a gradient-boosted ranking model with carefully constructed *past-only*
  historical statistics (per-user and per-video long-view/click rates and exposure counts) would
  directly optimize the within-user ranking objective without any deep-network overhead.
- **Action:** wrote a LightGBM `lambdarank` script with time-ordered `cumsum`-based historical
  features (subtracting the current row to avoid leakage).
- **Errors → recovery:** #3 crashed (`KeyError`: raw CSV has no `author_id` column — it lives in
  the video-features table). The agent joined the video-features table and #4 ran clean.
- **Result:** val primary **0.6186** — the best of the whole run, and never beaten again.

### Iterations #15–#21 — the deep-model detour (mostly bugs, then a modest recovery)
- **Hypothesis:** push listwise ranking losses and deeper AutoInt/DIN variants further.
- **Action:** tried listwise softmax over users, pairwise BPR loops, AutoInt+historical features.
- **Errors → recovery:** the agent hit a cluster of real failures and recovered from each —
  three 10-min timeouts (slow Python loops in the loss), `ValueError` from `torch.unique`
  unpacking, `IndexError` from out-of-range embedding lookups, `RuntimeError` from MLP dimension
  mismatch. The agent repeatedly simplified/routed around each and finally got #21 → 0.5837.
- **Result:** the deep-model branch never matched the LightGBM branch.

### Iterations #28–#31 — LightGBM resurgence to the plateau
- **Hypothesis:** enrich the winning LightGBM recipe — add a *time-decayed* user long-view rate,
  then historical + cross features.
- **Action:** iteratively added recency-weighted historical features and cross statistics.
- **Result:** steady climb **0.6108 → 0.6141 → 0.6144 → 0.6146**, but still below the 0.6186 peak.

### Iterations #32–#49 — search saturates (convergence)
- The agent alternated between the LightGBM branch (0.60–0.61) and the AutoInt+DIN branch
  (0.57–0.60), confirming the plateau. **No iteration exceeded 0.6186** — the run satisfied the
  convergence rule (ε = 0.002, N = 3) long before the 50-iteration cap.

---

## 4. Error & recovery statistics (robustness evidence)

21 of 50 iterations produced a bug; every one was either debugged to success or routed around by
the tree search. Categories:

| Error type | Count | How the agent recovered |
|---|---|---|
| `TypeError` (dim/field_dims misuse) | 4 | read `encode()` contract, switched to correct integer handling |
| `PicklingError` (DataLoader workers) | 3 | set `num_workers=0` / moved dataset class out of `__main__` |
| `KeyError` (`author_id` missing) | 2 | joined the video-features table |
| `RuntimeError` (MLP dim mismatch / torch.cat) | 3 | recomputed input dims, simplified the head |
| `IndexError` (embedding out of range) | 2 | aligned field_dims between train & inference |
| `ValueError` (hourmin parse / unpack) | 2 | fixed the parsing logic |
| Timeout (>10 min training) | 5 | simplified the loss loop / reduced epochs, or the search aborted the branch |

No crash, stall, or divergence killed the run: the agent always recovered or the tree search
moved to a healthier branch.

---

## 5. Manual-intervention summary

| Phase | Human interventions |
|---|---|
| Environment setup (macOS patch, DeepSeek config, task description) | run-before only |
| During the 50-iteration autonomous run | **0** |

All 50 iterations were proposed, implemented, evaluated, and revised by the agent itself. The only
human-authored inputs were the task description (data schema, metrics, leakage rule, and public
headroom directions from the Starter Kit README) and the reference starting point.
