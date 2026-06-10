"""Generate synthetic sample runs for honest-rl-bench demos.

Writes per-(algorithm, seed) CSVs (columns: step,return,seed) under
examples/runs/<algo>/seed_<n>.csv. Three "algorithms" x 5 seeds x ~20 eval
points each. Deliberately includes the failure modes the toolkit is built to
expose:

  - ``stable_baseline``  : monotone-ish learner that holds its peak (final≈peak).
  - ``peaky_overfit``    : peaks mid-training then DEGRADES — peak >> final.
                           This is the classic "report the best checkpoint" trap.
  - ``lucky_cherry``     : high variance; one seed gets lucky and spikes, which a
                           best-of-N / max-over-seeds report would cherry-pick,
                           but the IQM and final tell the honest story.

Run:  python examples/make_sample_runs.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "runs"
N_SEEDS = 5
N_EVALS = 20
MAX_STEP = 1_000_000
RNG = np.random.default_rng(20210813)  # rliable arXiv date, for fun


def _steps():
    return np.linspace(MAX_STEP / N_EVALS, MAX_STEP, N_EVALS).astype(int)


def stable_baseline(seed: int) -> np.ndarray:
    """Saturating learning curve that holds near its ceiling (final ≈ peak)."""
    steps = _steps()
    ceil = 560 + RNG.normal(0, 25)
    frac = steps / MAX_STEP
    curve = ceil * (1 - np.exp(-3.2 * frac))
    curve += RNG.normal(0, 12, size=curve.size)
    return np.clip(curve, 0, 1000)


def peaky_overfit(seed: int) -> np.ndarray:
    """Rises, peaks ~60% through, then degrades — large peak−final gap."""
    steps = _steps()
    frac = steps / MAX_STEP
    peak = 600 + RNG.normal(0, 20)
    # inverted-U: rise to a peak near frac=0.6 then fall back ~30%
    curve = peak * np.exp(-((frac - 0.6) ** 2) / (2 * 0.18 ** 2))
    curve *= (1 - 0.25 * np.clip(frac - 0.6, 0, 1) / 0.4)  # extra late decay
    curve += RNG.normal(0, 14, size=curve.size)
    return np.clip(curve, 0, 1000)


def lucky_cherry(seed: int) -> np.ndarray:
    """High-variance learner; ONE seed gets a lucky spike (best-of-N trap)."""
    steps = _steps()
    frac = steps / MAX_STEP
    base = 380 + RNG.normal(0, 40)
    curve = base * (1 - np.exp(-2.5 * frac))
    curve += RNG.normal(0, 45, size=curve.size)
    if seed == 0:  # the cherry-picked lucky seed
        spike_i = int(0.75 * N_EVALS)
        curve[spike_i:] += 220  # a single seed shoots up
    return np.clip(curve, 0, 1000)


ALGOS = {
    "stable_baseline": stable_baseline,
    "peaky_overfit": peaky_overfit,
    "lucky_cherry": lucky_cherry,
}


def main():
    steps = _steps()
    for algo, fn in ALGOS.items():
        d = ROOT / algo
        d.mkdir(parents=True, exist_ok=True)
        for seed in range(N_SEEDS):
            returns = fn(seed)
            with open(d / f"seed_{seed}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "return", "seed"])
                for s, r in zip(steps, returns):
                    w.writerow([int(s), round(float(r), 2), seed])
        print(f"  wrote {algo}: {N_SEEDS} seeds x {N_EVALS} evals")
    print(f"done → {ROOT}")


if __name__ == "__main__":
    main()
