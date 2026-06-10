"""Statistics core for honest RL benchmarking.

Pure-numpy implementations of the robust aggregate statistics recommended by
rliable (Agarwal, Schwarzer, Castro, Courville & Bellemare, "Deep Reinforcement
Learning at the Edge of the Statistical Precipice", NeurIPS 2021,
https://arxiv.org/abs/2108.13264). The headline recommendations we implement:

  - report the **interquartile mean (IQM)** instead of the mean (robust to the
    few-seed outliers that dominate RL) -- :func:`iqm`;
  - report **stratified bootstrap confidence intervals** rather than point
    estimates -- :func:`bootstrap_ci`, :func:`mean_ci`;
  - compare methods with a **paired** test on the per-run difference rather than
    comparing two independent CIs -- :func:`paired_diff_bootstrap`;
  - characterise the whole distribution with **performance profiles** (the
    fraction of runs scoring above a swept threshold tau) -- :func:`performance_profile`.

No external dependencies beyond numpy. These are deliberately self-contained
re-implementations (rliable proper pulls in pandas/arch, which clash in a
Flask-only venv); cross-check against rliable-proper at write-up time.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "iqm",
    "mean",
    "median",
    "coef_of_variation",
    "bootstrap_ci",
    "bootstrap_ci_from_scores",
    "mean_ci",
    "paired_diff_bootstrap",
    "paired_bootstrap",
    "prob_improvement",
    "performance_profile",
    "linear_decode_r2",
    "load_runs",
    "peak_scores",
    "final_scores",
]


def iqm(xs: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% of the scores.

    Drops the bottom and top 25% of values before averaging, so a single lucky
    or unlucky seed cannot dominate the aggregate. This is the primary point
    estimate recommended by rliable (Agarwal et al. 2021). For fewer than 4
    values the trimming is ill-defined and we fall back to the plain mean.
    """
    xs = np.sort(np.asarray(xs, dtype=float))
    if xs.size == 0:
        return float("nan")
    if xs.size < 4:
        return float(np.mean(xs))
    lo = int(0.25 * xs.size)
    hi = int(np.ceil(0.75 * xs.size))
    return float(np.mean(xs[lo:hi]))


def mean(xs: Sequence[float]) -> float:
    """Plain arithmetic mean (thin wrapper over ``numpy.mean``).

    Provided so callers can stay inside this module's vocabulary; it is the
    fragile aggregate rliable warns against (Agarwal et al. 2021) -- shown for
    contrast, never as the sole headline. Returns ``nan`` for empty input.
    """
    xs = np.asarray(xs, dtype=float)
    return float(np.mean(xs)) if xs.size else float("nan")


def median(xs: Sequence[float]) -> float:
    """Median (thin wrapper over ``numpy.median``).

    Robust to outliers but statistically inefficient (it uses a single order
    statistic), which is why rliable prefers the IQM. Returns ``nan`` for empty
    input.
    """
    xs = np.asarray(xs, dtype=float)
    return float(np.median(xs)) if xs.size else float("nan")


def coef_of_variation(xs: Sequence[float]) -> float:
    """Coefficient of variation: ``std / mean`` (population std, ddof=0).

    A scale-free measure of spread. Used in mechanism-checks as a cheap test of
    whether a quantity *varies enough across states* to be worth adapting to: a
    near-uniform quantity (low CV) has nothing to gate on. Returns ``nan`` for
    empty input or a zero mean (the ratio is undefined).
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return float("nan")
    m = float(np.mean(xs))
    if m == 0.0:
        return float("nan")
    return float(np.std(xs) / m)


def _grouped_resample(rng: np.random.Generator, xs: np.ndarray,
                      groups: np.ndarray | None) -> np.ndarray:
    """One stratified-bootstrap draw.

    If ``groups`` is given, resample WITHIN each group (e.g. within each seed or
    each task) and concatenate, preserving the stratified structure rliable
    recommends. Otherwise do an ordinary i.i.d. resample with replacement.
    """
    if groups is None:
        return rng.choice(xs, size=xs.size, replace=True)
    parts = []
    for g in np.unique(groups):
        members = xs[groups == g]
        if members.size:
            parts.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(parts) if parts else xs


def bootstrap_ci(xs: Sequence[float], fn: Callable[[Sequence[float]], float] = iqm,
                 n: int = 20000, alpha: float = 0.05, seed: int = 0,
                 groups: Sequence | None = None) -> tuple[float, float]:
    """Stratified bootstrap confidence interval for a statistic ``fn``.

    Resamples ``xs`` with replacement ``n`` times and returns the
    ``[alpha/2, 1 - alpha/2]`` percentiles of the resampled statistic (the
    percentile bootstrap CI used by rliable). When ``groups`` is supplied the
    resampling is *stratified*: each draw resamples within each group and
    concatenates, which is the correct procedure when runs are nested under
    seeds or tasks (Agarwal et al. 2021).

    Returns ``(lo, hi)``; ``(nan, nan)`` for an empty input.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return (float("nan"), float("nan"))
    grp = np.asarray(groups) if groups is not None else None
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=float)
    for i in range(n):
        boots[i] = fn(_grouped_resample(rng, xs, grp))
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


_AGGREGATES = {"iqm": iqm, "mean": np.mean, "median": np.median}


def bootstrap_ci_from_scores(scores: Sequence[float], aggregate: str | Callable = "iqm",
                             n: int = 20000, ci: float = 0.95, seed: int = 0,
                             groups: Sequence | None = None) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI on an aggregate of a flat array of per-run scores.

    A clear, named wrapper around :func:`bootstrap_ci` for the common case
    "I already have one score per run/seed and want ``(estimate, (lo, hi))``".
    ``aggregate`` may be ``"iqm"`` / ``"mean"`` / ``"median"`` or any callable.
    ``ci`` is the coverage (e.g. 0.95 for a 95% interval). Returns
    ``(point_estimate, (lo, hi))`` so callers can unpack it directly, matching
    the rliable convention of reporting an estimate *and* an interval together
    (Agarwal et al. 2021).
    """
    fn = _AGGREGATES.get(aggregate, aggregate) if not callable(aggregate) else aggregate
    if not callable(fn):
        raise ValueError(f"unknown aggregate {aggregate!r}; use 'iqm'/'mean'/'median' or a callable")
    alpha = 1.0 - ci
    xs = np.asarray(scores, dtype=float)
    est = float(fn(xs)) if xs.size else float("nan")
    lo, hi = bootstrap_ci(xs, fn=fn, n=n, alpha=alpha, seed=seed, groups=groups)
    return est, (lo, hi)


def mean_ci(xs: Sequence[float], n: int = 20000, alpha: float = 0.05,
            seed: int = 0, groups: Sequence | None = None) -> tuple[float, float, float]:
    """Point estimate (mean) plus a bootstrap CI for the mean.

    Returns ``(mean, lo, hi)``. Convenience wrapper around :func:`bootstrap_ci`
    with ``fn=np.mean`` for callers that want the mean rather than the IQM.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = bootstrap_ci(xs, fn=np.mean, n=n, alpha=alpha, seed=seed, groups=groups)
    return (float(np.mean(xs)), lo, hi)


def paired_diff_bootstrap(a: Sequence[float], b: Sequence[float],
                          fn: Callable[[Sequence[float]], float] = iqm,
                          n: int = 20000, alpha: float = 0.05,
                          seed: int = 0) -> dict:
    """Paired head-to-head test: bootstrap CI on the per-pair difference a - b.

    This is the statistically correct way to compare two methods that were run
    on the *same* set of seeds/tasks: form the per-pair difference ``d = a - b``
    and bootstrap a CI on ``fn(d)``. Comparing two independent CIs (does method
    A's CI overlap method B's?) is *not* equivalent and is systematically less
    powerful, because it ignores the seed-level pairing (Agarwal et al. 2021,
    on probability of improvement and paired comparisons).

    ``a`` and ``b`` must be the same length and aligned pair-for-pair (e.g.
    a[i] and b[i] are the same seed). Returns a dict with the point estimate of
    the difference, its CI, ``prob_a_better`` (fraction of bootstrap draws with
    positive difference), and ``significant`` (CI excludes 0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError(f"paired_diff_bootstrap needs aligned arrays; got {a.size} vs {b.size}")
    if a.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "prob_a_better": float("nan"), "significant": False, "n_pairs": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=float)
    idx_pool = np.arange(d.size)
    for i in range(n):
        idx = rng.choice(idx_pool, size=d.size, replace=True)
        boots[i] = fn(d[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "diff": float(fn(d)),
        "lo": lo,
        "hi": hi,
        "prob_a_better": float(np.mean(boots > 0)),
        "significant": lo > 0 or hi < 0,
        "n_pairs": int(d.size),
    }


# paired_bootstrap is the name used in the docs; keep it as a public alias of
# the canonical paired_diff_bootstrap so both resolve to the same code.
paired_bootstrap = paired_diff_bootstrap


def prob_improvement(a: Sequence[float], b: Sequence[float],
                     n: int = 20000, ci: float = 0.95,
                     seed: int = 0) -> tuple[float, tuple[float, float]]:
    """Probability of improvement P(X > Y) with a bootstrap CI (rliable).

    Estimates the Mann-Whitney-style probability that a randomly drawn run of
    method ``a`` scores above a randomly drawn run of method ``b`` (ties count
    as half), the metric rliable recommends for an *interpretable*,
    distribution-free head-to-head: "if I switch, how often am I better off?"
    (Agarwal et al. 2021). Unlike :func:`paired_diff_bootstrap` this does NOT
    assume the runs are paired -- it compares the two score *distributions* over
    all i*j cross-pairs.

    The CI is obtained by resampling ``a`` and ``b`` independently with
    replacement ``n`` times and taking the ``[ (1-ci)/2, 1-(1-ci)/2 ]``
    percentiles. Identical distributions give exactly 0.5. Returns
    ``(p_improvement, (lo, hi))``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"), (float("nan"), float("nan")))

    def _p(xa: np.ndarray, xb: np.ndarray) -> float:
        # mean over all cross-pairs of [x>y] + 0.5*[x==y]
        diff = xa[:, None] - xb[None, :]
        return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)

    point = _p(a, b)
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=float)
    for i in range(n):
        ra = rng.choice(a, size=a.size, replace=True)
        rb = rng.choice(b, size=b.size, replace=True)
        boots[i] = _p(ra, rb)
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, (lo, hi)


def linear_decode_r2(X, y) -> float:
    r"""Coefficient of determination R^2 of a least-squares linear probe y ~ X.

    Fits ``y ≈ X @ w + b`` by ordinary least squares (a bias column is added
    automatically) and returns the in-sample R^2 = ``1 - SS_res / SS_tot``. This
    is the generic "is the target already linearly decodable from the
    representation?" mechanism-check: an R^2 near 1 means the quantity is already
    present in the features, so an objective that tries to *add* it has no
    headroom. ``X`` may be 1-D (treated as a single feature) or 2-D
    ``(n_samples, n_features)``; ``y`` is 1-D ``(n_samples,)``.

    Returns ``nan`` if ``y`` has zero variance (R^2 undefined) or there are no
    samples. Generic linear-algebra only -- no domain assumptions -- which is
    why it ships as a real function (unlike the task-specific probes that
    *produce* X and y).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0] or y.size == 0:
        return float("nan")
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def performance_profile(xs: Sequence[float], taus: Sequence[float] | None = None,
                        num_taus: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """rliable-style performance profile: fraction of runs scoring above tau.

    Returns ``(taus, fractions)`` where ``fractions[i]`` is the fraction of runs
    in ``xs`` with score >= ``taus[i]``. As tau increases the surviving fraction
    can only decrease, so the curve is monotonically non-increasing -- a single
    plot that shows the entire score distribution rather than one summary number
    (Agarwal et al. 2021, "performance profiles"). If ``taus`` is None a linear
    sweep of ``num_taus`` points spanning the observed score range is used.
    """
    xs = np.asarray(xs, dtype=float)
    if taus is None:
        if xs.size == 0:
            taus = np.linspace(0.0, 1.0, num_taus)
        else:
            lo, hi = float(np.min(xs)), float(np.max(xs))
            if lo == hi:
                hi = lo + 1.0
            taus = np.linspace(lo, hi, num_taus)
    taus = np.asarray(taus, dtype=float)
    if xs.size == 0:
        return taus, np.zeros_like(taus)
    fractions = np.array([float(np.mean(xs >= t)) for t in taus])
    return taus, fractions


# ─── Curve-aware convenience accessors ──────────────────────────────────────
# Re-exported from .curves so callers can do everything through ``stats``.
# Imported at the bottom of the module to avoid a circular import (curves
# imports the aggregate statistics defined above).

def peak_scores(runset, group: str | None = None) -> np.ndarray:
    """Per-seed peak (max over the curve) scores for a group in a RunSet.

    Convenience accessor over :class:`honest_rl_bench.curves.RunSet`: returns the
    array of per-seed peaks for ``group``. If ``group`` is None and the RunSet
    has exactly one group, that group is used. The peak-vs-final pair is the
    single most common way RL results mislead, so always read both.
    """
    g = _resolve_group(runset, group)
    return g.peaks()


def final_scores(runset, group: str | None = None, last_k: int | None = None) -> np.ndarray:
    """Per-seed final (mean of the last ``last_k`` evals) scores for a group.

    Convenience accessor over :class:`honest_rl_bench.curves.RunSet`. ``last_k``
    defaults to the RunSet's ``final_k``. If ``group`` is None and the RunSet has
    exactly one group, that group is used.
    """
    g = _resolve_group(runset, group)
    k = last_k if last_k is not None else runset.final_k
    return g.finals(k)


def _resolve_group(runset, group: str | None):
    names = runset.group_names()
    if group is None:
        if len(names) == 1:
            return runset.groups[names[0]]
        raise ValueError(
            f"RunSet has {len(names)} groups {names}; pass group=... to select one")
    if group not in runset.groups:
        raise KeyError(f"no group {group!r}; available: {names}")
    return runset.groups[group]


# load_runs is the directory loader; expose it as ``stats.load_runs`` so the
# documented call resolves. The implementation lives in .curves, which imports
# this module, so we resolve it lazily via module __getattr__ to avoid a circular
# import at load time.
def __getattr__(name: str):
    if name == "load_runs":
        from .curves import load_runs as _load_runs
        return _load_runs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
