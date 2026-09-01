# Autonomous ML Research Agent for Recommender Systems — KuaiRand-Pure

An **autonomous ML research agent** for the TikTok TechJam / ByteDance hackathon track
*"Autonomous Machine Learning Research Agent for Recommender Systems."*

The agent runs the full MLE loop — read the problem → inspect data → engineer features → train &
tune → evaluate → reflect & iterate — **on its own**, and converges to a recommendation model that
**beats the official FM baseline** on the required KuaiRand-Pure benchmark.

> **Final result (hidden-test): GAUC 0.6817 / nDCG@5 0.5423 / primary 0.6120** vs the official
> baseline **0.5946** — an absolute delta of **+0.0174** (≈ 20× the baseline's 5-seed std of 0.0008).
>
> **Bonus (KuaiRand-1k, hidden-test): GAUC 0.7148 / nDCG@5 0.8462 / primary 0.7805** vs baseline
> **0.6390** — an absolute delta of **+0.1415** (see `1k/`).

---

## 1. Project overview

The agent is built on **ML-Master** (an MCTS-driven autonomous ML-research agent), adapted to run
on KuaiRand with a DeepSeek reasoning model. Given a task description (data schema, exact GAUC /
nDCG@5 metric definitions, a leakage rule, and the public headroom directions from the Starter Kit
README), the agent autonomously:

1. drafts candidate pipelines (model + loss + features),
2. executes them in a sandbox,
3. reads its own validation score,
4. reflects and iterates via Monte-Carlo tree search.

Over 50 iterations it **self-discovered** that the winning recipe is not a deep model but a
**LightGBM `LambdaRank` ranker + past-only historical aggregate features** (per-user / per-video /
per-author / per-tab exposure counts, long-view rates, click rates, time-of-day & weekend
features). This beat both the organizer's FM baseline and a hand-written AutoInt reference.

### Key numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| **Agent (test)** | **0.6817** | **0.5423** | **0.6120** |
| **Δ vs baseline** | **+0.0207** | **+0.0141** | **+0.0174** |

### Bonus benchmark — KuaiRand-1k

The same agent was re-run on **KuaiRand-1k** (~983 high-activity users, ~5.06M train / 2.52M valid
/ 4.13M test rows; same schema as Pure). It kept the LambdaRank objective and enriched the
historical features with time-decayed statistics, click-history stats, and recent-window features,
and capped per-user query groups at LightGBM's 10,000-row hard limit.

| (test) | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Item popularity (floor) | — | — | 0.5249 |
| Official FM baseline | 0.6730 | 0.6049 | 0.6390 |
| **Agent** | **0.7148** | **0.8462** | **0.7805** |
| Δ vs baseline | +0.0418 | +0.2413 | **+0.1415** |

Full details in `1k/RESULTS_SUMMARY_1k.md`. A teammate's parallel GPU run (simpler cumsum features)
reached test primary 0.6082; the final submission uses the higher 0.7805 result.

### Repository layout

```
├── final_submission.py   # the final pipeline (LightGBM LambdaRank) — generates submission.csv
├── submission/
│   └── submission.csv    # the final submission (passed official submit.py --check)
├── instructions.txt      # task description given to the agent (KuaiRand-Pure)
├── instructions_1k.txt   # task description for the KuaiRand-1K subset (default in config)
├── helpers/              # official Starter Kit evaluation/data/submit/baseline (unchanged)
├── references/           # reference implementations fed to the agent (DIN / AutoInt)
├── mlmaster/             # the ML-Master agent framework (adapted for KuaiRand + DeepSeek)
├── 1k/                   # bonus benchmark KuaiRand-1k (submission + pipeline + summary)
│   ├── submission_1k.csv.gz   # final 1k submission (gzip; gunzip to submission_1k.csv)
│   ├── final_submission_1k.py # the 1k pipeline (row-order fixed)
│   ├── RESULTS_SUMMARY_1k.md  # 1k results + delta + resource report
│   └── instructions_1k.txt    # task description for the 1k run
├── ITERATION_LOG.md      # per-iteration log (hypothesis / diff / metric / error-recovery)
├── RESULTS_SUMMARY.md    # results table + resource usage report
├── ABOUT_PROJECT.md      # Devpost "About the project"
└── requirements.txt
```

The agent framework lives in `mlmaster/` (upstream ML-Master, adapted for KuaiRand + DeepSeek). The
final run's raw logs and best solution are not in this repo (they are large runtime artifacts);
see `ITERATION_LOG.md` for the extracted per-iteration record.

---

## 2. Setup & installation

Python 3.10+, and the following packages:

```bash
pip install numpy pandas lightgbm scikit-learn
# for the agent framework (to reproduce the autonomous run):
pip install omegaconf rich openai flask coolname shutup humanize backoff
# optional, for the deep-model baselines the agent explored:
pip install torch
```

Data: download **KuaiRand-Pure** from https://kuairand.com (Zenodo direct link, no registration)
and unpack it so you have `KuaiRand-Pure/data/*.csv`.

---

## 3. Reproduce the final result

### 3.1 Generate the submission (standalone — no agent, no API key)

```bash
# 1. Download KuaiRand-Pure from https://kuairand.com and unpack it — you get <data_dir>/*.csv
# 2. Prepare the input/ layout that final_submission.py expects:
mkdir -p input
cp <data_dir>/log_standard_4_08_to_4_21_pure.csv input/
cp <data_dir>/log_standard_4_22_to_5_08_pure.csv input/
cp <data_dir>/video_features_basic_pure.csv input/
cp helpers/evaluate.py helpers/data.py input/

python final_submission.py
# → prints Validation primary ≈ 0.6186 and writes submission/submission.csv
```

Validate against the official harness (helpers/submit.py):

```bash
python helpers/submit.py --check submission/submission.csv --data_dir <data_dir> --split test
python helpers/submit.py --score submission/submission.csv --data_dir <data_dir> --split test
# → ✓ alignment OK · GAUC 0.6817 | nDCG@5 0.5423 | primary 0.6120
```

### 3.2 Reproduce the autonomous run (ML-Master + DeepSeek)

```bash
cd mlmaster
# 1. Put the KuaiRand-Pure CSVs + helpers/evaluate.py & data.py into mlmaster/data/
#    (config_mcts.yaml → data_dir points to ./data)
# 2. Set your DeepSeek key / base_url in utils/config_mcts.yaml (agent.code / agent.feedback)
python main_mcts.py agent.steps=50 start_cpu_id=0 cpu_number=8 exp_name=kuairand_afi
```

The task description is pointed to by `config_mcts.yaml → desc_file`: the default is the
KuaiRand-1K prompt (`../instructions_1k.txt`, and `mlmaster/data/` holds the 1k subset);
use `../instructions.txt` for the full KuaiRand-Pure task. Note: ML-Master is Linux-first; on
macOS one line in `interpreter/interpreter_parallel.py` (the `os.sched_setaffinity` injection)
was guarded to skip CPU-pinning, and DeepSeek's reasoning mode was disabled for the code model
(see `utils/llm_caller.py`).

### 3.3 Bonus benchmark — KuaiRand-1k (reproduce)

```bash
cd 1k
gunzip -k submission_1k.csv.gz        # → submission_1k.csv (the final submission)
# regenerate the submission from scratch:
mkdir -p input && cp <KuaiRand-1K>/data/*_1k.csv input/ && cp ../helpers/evaluate.py ../helpers/data.py input/
python final_submission_1k.py
```

Full details (deltas, resource report, team cross-check) in `1k/RESULTS_SUMMARY_1k.md`.

---

## 4. Limitations & what we'd improve

- **The gain is real but small.** test primary +0.0174 is ~20σ of the baseline std, but the
  attainable ceiling is 0.8645 (oracle), so there is still ≈ 0.25 headroom left.
- **The agent's own multi-task attempt was flawed.** In the best checkpoint the auxiliary click
  head was effectively inert (labels extracted as zeros). A *correctly wired* multi-task head over
  `is_click` / `is_like` is the clearest next lever (Starter Kit direction #3).
- **Temporal drift between train (early April) and test (early May) was not directly modeled.**
  A time-decay experiment actually hurt (the learned decay went *negative* — long-term preference
  dominates `long_view`), but drift-aware training deserves a proper study.
- **The framework doesn't implement "stop on convergence."** The best score was found ~5 minutes
  in, yet the run burned the remaining wall-clock/tokens exploring the plateau. Implementing the
  ε=0.002 / N=3 early-stop would cut resource consumption (Feasibility score) ~10×.
- **No sequence modeling in the winning model.** The agent's historical features are aggregate
  statistics, not DIN/SIM-style attention. A true sequence module is the largest unexplored
  direction (Starter Kit direction #2).

---

## 5. Team contributions

- **Agent framework & adaptation** — **Nuoyan Xu**: adapted ML-Master to KuaiRand (DeepSeek),
  wrote the task description, leakage rule, and reference starting point; ran the autonomous loop.
- **Recommendation modeling & baselines** — **Bohan Shao**: the AutoInt (AFI) reference
  implementation and the D2Co label-ablation study that mapped the dead-ends; independent MLEvolve
  runs for cross-checking.
