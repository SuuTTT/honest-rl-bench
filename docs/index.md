---
layout: home
title: Benchmark RL like you mean it
---

# honest-rl-bench

**Most RL benchmarking misleads.** Not usually through fraud — through a thousand
small, defensible-looking choices: a mean over three seeds, the best checkpoint,
the friendly opponent, the final-task number. Each is locally reasonable. Together
they manufacture wins that don't reproduce.

This site is the tutorial half of [**honest-rl-bench**](https://github.com/SuuTTT/honest-rl-bench),
a small toolkit that makes the honest path the easy path: point it at a folder of
run CSVs (`step, return, seed`) and it returns **IQM with stratified bootstrap
confidence intervals**, **peak *and* final** reporting, and **paired-difference**
A-vs-B comparison — instead of fragile point estimates.

The thesis in one line:

> **Never report a number you didn't read back from disk. Report peak *and* final.
> Report uncertainty, not point estimates. Pre-register your gate before you look.
> And check the mechanism before you spend the compute.**

## The case study is real

The methodology here isn't aspirational. It comes out of
[**tdmpc-glass**](https://github.com/SuuTTT/tdmpc-glass), a multi-week, ~10–12-GPU
model-based-RL campaign that tried to beat TD-MPC2 with "abstraction." Under a strict
fair protocol, **eight apparent wins dissolved to null** (thirteen levers in total
across the full campaign), the *one* method that genuinely beat the baseline turned
out to be **published prior art**, and a single cheap **mechanism-check** killed a
multi-week idea in an afternoon. The most embarrassing wins were the most instructive,
so we use them as worked examples throughout.

## The five tutorials

Read them in order, or jump to your pain point. Each ends with a **Takeaways** box.

1. **[Report uncertainty, not point estimates]({{ '/statistics/' | relative_url }})**
   — Why mean±std over 3 seeds lies. IQM, stratified bootstrap CIs, performance
   profiles, probability of improvement, and how many seeds you actually need.
2. **[The ways RL results mislead — and how to stop]({{ '/cherry-picking/' | relative_url }})**
   — Peak-vs-final (report both), best-of-N seed selection, checkpoint selection,
   tuning on the test tasks, pre-registered gates, and paired comparison. A do/don't checklist.
3. **[MARL-specific pitfalls]({{ '/marl/' | relative_url }})**
   — Non-transitivity (A beats B beats C beats A), why a single opponent's win-rate
   misleads, population/tournament evaluation (Elo, Nash averaging), self-play vs
   fixed-pool, and non-stationarity.
4. **[Continual & multi-task RL evaluation]({{ '/catastrophic-forgetting/' | relative_url }})**
   — What forgetting is, the evaluation matrix over task sequences, and the metrics
   that expose it: average performance, backward transfer (forgetting), forward
   transfer, and plasticity.
5. **[Mechanism-check: spend a day, save a month]({{ '/mechanism-check/' | relative_url }})**
   — The cheap-kill-test-before-fan-out discipline, and the anti-fabrication rule
   that this very project had to learn the hard way.

## Quickstart (the toolkit)

```bash
pip install -e .
# Aggregate a runs directory (CSVs with columns: step,return,seed)
python -m honest_rl_bench.report examples/runs --out report.html
# Live learning-curve dashboard
python -m honest_rl_bench.dashboard examples/runs   # http://localhost:5055
```

The code snippets on every page call this same toolkit, so the docs and the tool
can't drift apart. If a snippet here disagrees with the API, the snippet is the bug.
