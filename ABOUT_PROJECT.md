# About the project

## Inspiration

Recommender engineers spend most of their time in one loop: read the data, engineer features,
train, evaluate, repeat. This challenge asks us to automate that loop — build an agent that
iterates on its own and beats the official FM baseline on KuaiRand-Pure. We started from
[ML-Master](https://github.com/sjtu-sai-agents/ML-Master), an MCTS-driven agent that is
state-of-the-art on MLE-Bench, and adapted it to recommender ranking.

## What it does

The agent runs the full loop autonomously. The task is **within-user ranking**: for each user,
rank only that user's already-logged impressions by the probability of `long_view == 1`. The
scored metrics are per-user **GAUC** (grouped AUC, weighted by each user's positive count) and
**nDCG@5** (gain `2^rel - 1`, position discount `1/log2(rank + 1)`), with

```
primary = (GAUC + nDCG@5) / 2
```

Starting from a hand-written AutoInt/DIN reference, the agent **self-discovered** that the winning
recipe is not a deep model but a **LightGBM `LambdaRank` ranker + past-only historical aggregate
features** — per-user / per-video / per-author / per-tab exposure counts, long-view rates and
click rates, plus time-of-day and weekend features.

**Final score: val primary 0.6186 / test primary 0.6120**, against the official FM baseline of
0.6016 / 0.5946 — an absolute test delta of **+0.0174** (about 20x the baseline's 5-seed std of
0.0008). The agent found and built this stack itself over 50 autonomous iterations.

## How we built it

Three layers on top of ML-Master:

1. **Set up the task** — wrote the date-based splits, the GAUC/nDCG@5 evaluation, and the
   submission format, configured for the DeepSeek v4 API.
2. **Gave the agent hints, not answers** — the organizer's headroom directions as a map, plus
   validated reference starting points (DIN, AutoInt), leaving every improvement to the agent.
3. **Validated independently** — every hypothesis was scored on the held-out validation split,
   and the final submission was checked against the official `submit.py` harness.

## What we learned

- **The ranking loss matters most.** With pointwise BCE the agent got primary ≈ 0.478 (near
  random); switching to a listwise ranking objective (LambdaRank, which directly optimizes NDCG
  by weighting each pair's gradient by the NDCG change it would cause) jumped the same signal to
  0.6186.
- **Feature engineering beat model depth here.** AutoInt (self-attention field interaction) and
  DIN (sequence attention) both plateaued below a gradient-boosted ranker fed historical
  statistics, matching the organizers' hint that "the bottleneck is not model capacity".
- **Negative results count.** A learnable recency-decay was learned *negative* — the model
  preferred older history, because `long_view` reflects long-term taste, not recent spikes.

## Challenges we faced

- **Reasoning-model LLM** — DeepSeek v4's default "thinking" mode burned the whole token budget;
  disabling it cut per-call latency from ~287 s to ~21 s.
- **Label leakage** — the agent learned to feed `is_click`/`play_time_ratio` as features and
  "scored" 0.84 (past the 0.8645 oracle ceiling). We had to forbid it at the source in the task
  description.
- **Autonomy vs. results** — too much guidance kills the Autonomy score, too little wastes the
  50-iteration budget; "hints, not answers" was the balance.

## Built with

**Development tools**: VS Code, macOS Terminal, Git.

**APIs used**: DeepSeek v4 API (`deepseek-v4-pro`, via the OpenAI-compatible endpoint).

**Libraries & frameworks**: ML-Master (MCTS agent framework), LightGBM, PyTorch, NumPy, pandas,
scikit-learn, OmegaConf, OpenAI SDK.

**Datasets & assets**: KuaiRand-Pure (https://kuairand.com), the official KuaiRand starter kit
(`evaluate.py` / `data.py` / `submit.py`).
