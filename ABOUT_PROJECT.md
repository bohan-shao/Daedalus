# About the project

## Inspiration

Recommender engineers spend most of their time in one loop: read the data, engineer features,
train, evaluate, repeat. This challenge asks us to automate that loop — build an agent that
iterates on its own and beats the official FM baseline on KuaiRand-Pure. We started from
[ML-Master](https://github.com/sjtu-sai-agents/ML-Master), an MCTS-driven agent that is
state-of-the-art on MLE-Bench, and adapted it to recommender ranking.

## What it does

The agent runs the full loop autonomously. The task is within-user ranking: score each user's
logged impressions by the probability of `long_view == 1`, measured by GAUC and nDCG@5,
with `primary = (GAUC + nDCG@5) / 2`.

Left to itself, the agent converged on a stack that combines a **DIN-style sequence model**
(target attention over each user's time-ordered history), a **ranking loss** (listwise softmax
+ pairwise BPR instead of pointwise BCE), and **multi-task auxiliary heads** (`is_click`).

Final score: primary **0.6017** (val) / **0.5948** (test), against the official FM baseline of
0.6016 / 0.5946. The margin is small, but the agent found and built this stack itself.

## How we built it

Three layers on top of ML-Master:

1. **Ported it to KuaiRand** — wrote the date-based splits, the GAUC/nDCG@5 evaluation, and the
   submission format, and patched it to run on an Apple-Silicon Mac with the DeepSeek v4 API.
2. **Gave the agent hints, not answers** — the organizer's seven headroom directions as a map,
   plus one validated DIN starting point, leaving every improvement to the agent.
3. **Validated independently** — every hypothesis was scored on the held-out validation split.

## What we learned

- **The loss function matters most.** With pointwise BCE the agent got primary ≈ 0.478 (near
  random); switching to a ranking loss jumped the same model to 0.60.
- **Sequence modeling was the biggest untouched headroom.** A generic LLM alone produces only
  average-pooled "sequence features", not real target attention.
- **Negative results count.** A learnable recency-decay was learned *negative* — the model
  preferred older history, because `long_view` reflects long-term taste, not recent spikes.

## Challenges we faced

- **Environment** — ML-Master is Linux/CUDA-only; running on a fanless Mac M4 required patching
  and trimming 470 dependencies to ~15.
- **Reasoning-model LLM** — DeepSeek v4's default "thinking" mode burned the whole token budget;
  disabling it cut per-call latency from ~287 s to ~21 s.
- **Label leakage** — the agent learned to feed `is_click`/`play_time_ratio` as features and
  "scored" 0.84 (past the 0.8645 oracle ceiling). We had to forbid it at the source.
- **Autonomy vs. results** — too much guidance kills the Autonomy score, too little wastes the
  50-iteration budget; "hints, not answers" was the balance.

Built on ML-Master (MCTS + code interpreter), PyTorch, DeepSeek v4 API, and the KuaiRand
starter kit, developed on a single Apple M4 laptop.
