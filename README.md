# honest-rl-bench

**An honest RL / MARL benchmarking toolkit — and a tutorial on how to benchmark RL the right way.**

Most RL papers report a single mean curve over 3 seeds, pick the best checkpoint, and call it a win.
Then it doesn't reproduce. This repo is two things:

1. **A drop-in toolkit** — point it at a folder of run CSVs (`step, return, seed`) and get:
   - **IQM** (interquartile mean) + **stratified bootstrap confidence intervals** (rliable-style, Agarwal et al. 2021) instead of fragile means,
   - **peak *and* final** reporting (the single most common way RL results mislead),
   - **paired-difference bootstrap** for honest head-to-head A-vs-B comparison,
   - a **live learning-curve dashboard** and a static **report card**.
2. **A tutorial** ([honest-rl-bench docs site](https://suuttt.github.io/honest-rl-bench/)) on doing benchmarking right:
   how to avoid cherry-picking, how to report uncertainty, MARL-specific pitfalls (non-transitivity,
   evaluation-opponent choice), catastrophic forgetting in continual RL, and the
   **mechanism-check-before-fan-out** discipline that saves compute and credibility.

## Why
This grew out of a real model-based-RL research campaign ([tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass))
where **eight apparent "wins" dissolved to null** under a strict protocol — best-of-N seed luck,
peak-vs-final confusion, basin-lottery from restarts. The tooling and the rules here are what survived.
The campaign is used as the running case study in the tutorial.

## Quickstart
```bash
pip install -e .
# aggregate a runs directory (CSVs with columns: step,return,seed)
python -m honest_rl_bench.report examples/runs --out report.html
# live dashboard
python -m honest_rl_bench.dashboard examples/runs   # http://localhost:5055
```

## The one rule
**Never report a number you didn't read back from disk.** Peak *and* final. CIs, not point estimates.
Pre-register your gate before you look. See the tutorial.

## License
MIT.
