# PPT Presentation Script — SHORT version (≈3 min)

> Matches the 11-slide deck. Normal pace ≈ 3 minutes. Each section starts with 【Slide N】.

---

## 【Slide 1 · Cover】

Hi, we're Nuoyan Xu and Bohan Shao. Our project is an **autonomous ML research agent for recommender systems**.

The result in one line: on KuaiRand-Pure, our agent iterated on its own and reached **test primary 0.6120**, beating the baseline **0.5946** by **0.0174** — about 20× its standard deviation.

---

## 【Slide 2 · Background】

Recommender engineers spend every day in one loop: read data, engineer features, train, evaluate, reflect, and repeat. It's mechanical, repetitive, and every step writes code — exactly what a code-generating LLM can automate. So our task is to hand that whole loop to an AI agent.

---

## 【Slide 3 · Solution】

We built it on **ML-Master**: the LLM writes code → a sandbox runs it and scores it → a second model judges bugs → **MCTS** picks the next move. The whole loop runs with **no human in it**.

---

## 【Slide 4 · Method】

Three layers of work: **set up the task** (splits, GAUC/nDCG@5 evaluation, submission format); **give hints, not answers**; and **validate independently** on the held-out split.

---

## 【Slide 5 · The winning stack】

The best part: the agent **discovered its own recipe**. Not a deep model — **LightGBM LambdaRank + "past-only" historical features**, as shown in this pipeline.

---

## 【Slide 6 · Results】

It beat the baseline across the board: primary **+0.0174** on the test set, about 20× the baseline's std — statistically significant.

---

## 【Slide 7 · Full autonomy】

Fifty iterations, **zero human interventions**. This trajectory shows the agent climbing from near-random past the baseline all on its own.

---

## 【Slide 8 · Leakage guard】

It once cheated by feeding `is_click` as a feature, scoring 0.84 — past the oracle ceiling. We **forbade it at the source**, so the final score is genuine.

---

## 【Slide 9 · A negative result】

It learned recency-decay **negative** — meaning long-term taste beats recent behavior. A negative result worth keeping.

---

## 【Slide 10 · Challenges】

DeepSeek's "thinking" mode (287s → 21s after disabling), and the label-leakage issue.

---

## 【Slide 11 · Summary + code structure】

Everything is open-sourced on GitHub: four documents at the root, `final_submission.py` + `submission.csv`, and three folders — `helpers`, `references`, `mlmaster`. **Framework is the body, solution is the product, docs are the support.** Thank you.
