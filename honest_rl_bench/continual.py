"""Continual / multi-task RL evaluation: the evaluation matrix and its reads.

When an agent learns a *sequence* of tasks, a single final number hides
catastrophic forgetting. The right primitive is a matrix, not a scalar: train on
tasks ``1..T`` in order and, after finishing each task ``i``, evaluate on *every*
task ``j``. Call the result ``R[i][j]`` = performance on task ``j`` after training
through task ``i``. Every metric here is a different read of that matrix.

Definitions follow the standard continual-learning literature:
  - Backward / forward transfer: Lopez-Paz & Ranzato, "Gradient Episodic Memory
    for Continual Learning", NeurIPS 2017, https://arxiv.org/abs/1706.08840;
  - forgetting / intransigence framing: Chaudhry, Dokania, Ajanthan & Torr,
    "Riemannian Walk for Incremental Learning", ECCV 2018,
    https://arxiv.org/abs/1801.10112;
  - loss of plasticity: Dohare et al., "Loss of Plasticity in Deep Continual
    Learning", Nature 2024.

Pure numpy; no external dependencies.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "eval_matrix",
    "average_performance",
    "backward_transfer",
    "forward_transfer",
    "plasticity",
]


def eval_matrix(records, n_tasks: int | None = None) -> np.ndarray:
    """Build the continual eval matrix ``R[i][j]`` from per-evaluation records.

    Accepts either:
      - a 2-D array-like already shaped ``(T, T)`` (returned as a float array), or
      - an iterable of records ``(i, j, score)`` = score on task ``j`` after
        training through task ``i`` (0-based task indices). Repeated ``(i, j)``
        records are averaged (so each cell can itself be a seed aggregate).

    ``i`` indexes *training stage* (row), ``j`` indexes *eval task* (column).
    Returns a float ``(T, T)`` matrix; unobserved cells are ``nan``. ``n_tasks``
    fixes ``T`` (else inferred from the largest index seen).
    """
    arr = np.asarray(records, dtype=object)
    if isinstance(records, np.ndarray) and records.ndim == 2:
        return np.asarray(records, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and not _looks_like_records(records):
        return np.asarray(records, dtype=float)

    recs = list(records)
    if n_tasks is None:
        n_tasks = 0
        for i, j, _ in recs:
            n_tasks = max(n_tasks, int(i) + 1, int(j) + 1)
    tot = np.zeros((n_tasks, n_tasks), dtype=float)
    cnt = np.zeros((n_tasks, n_tasks), dtype=float)
    for i, j, score in recs:
        tot[int(i), int(j)] += float(score)
        cnt[int(i), int(j)] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(cnt > 0, tot / cnt, np.nan)
    return R


def _looks_like_records(records) -> bool:
    try:
        first = next(iter(records))
    except (TypeError, StopIteration):
        return False
    return hasattr(first, "__len__") and len(first) == 3 and np.isscalar(first[2])


def average_performance(R: np.ndarray) -> float:
    """Average final performance: ``mean_j R[T-1][j]`` over the last row.

    The headline "how good is the agent across all tasks at the end". Necessary
    but propped-up-able by a few easy tasks, so always report the per-task final
    row (``R[-1]``) alongside it (Lopez-Paz & Ranzato 2017). nan-safe.
    """
    R = np.asarray(R, dtype=float)
    if R.size == 0:
        return float("nan")
    last = R[-1]
    return float(np.nanmean(last)) if np.any(~np.isnan(last)) else float("nan")


def backward_transfer(R: np.ndarray) -> float:
    r"""Backward transfer (BWT) = forgetting (Lopez-Paz & Ranzato 2017).

    For each earlier task ``j < T-1``, compare its score *right after it was
    learned* (the diagonal ``R[j][j]``) to its score *at the end* (``R[T-1][j]``)
    and average:

        ``BWT = mean_{j<T-1} ( R[T-1][j] - R[j][j] )``.

    **Negative BWT is forgetting** (end-of-run worse than just-learned); positive
    BWT means later tasks *helped* earlier ones. This is the number a final-task
    report hides. Returns ``nan`` for ``T < 2``. nan-safe over tasks.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return float("nan")
    diffs = []
    for j in range(T - 1):
        end, learned = R[T - 1, j], R[j, j]
        if not (np.isnan(end) or np.isnan(learned)):
            diffs.append(end - learned)
    return float(np.mean(diffs)) if diffs else float("nan")


def forward_transfer(R: np.ndarray, random_baseline: Sequence[float] | None = None) -> float:
    r"""Forward transfer (FWT): does training earlier tasks help a future one?

    Following Lopez-Paz & Ranzato (2017): compare the zero-shot score on task
    ``i`` *before* training it (the entry just above the diagonal, ``R[i-1][i]``)
    to a from-scratch ``random_baseline[i]``, and average over ``i >= 1``:

        ``FWT = mean_{i>=1} ( R[i-1][i] - random_baseline[i] )``.

    Positive FWT means the curriculum builds reusable structure. If
    ``random_baseline`` is None it is taken as zeros (so FWT reduces to the mean
    pre-training zero-shot score, still a valid relative signal). Returns ``nan``
    for ``T < 2``. nan-safe over tasks.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return float("nan")
    if random_baseline is None:
        base = np.zeros(T)
    else:
        base = np.asarray(random_baseline, dtype=float)
    diffs = []
    for i in range(1, T):
        z = R[i - 1, i]
        if not np.isnan(z):
            diffs.append(z - base[i])
    return float(np.mean(diffs)) if diffs else float("nan")


def plasticity(R: np.ndarray) -> float:
    r"""Plasticity: can the agent *still learn* new tasks late in the sequence?

    Reads the diagonal ``R[i][i]`` -- performance on each task right after
    learning it. If that just-learned performance decays as ``i`` grows, the
    network has lost plasticity (Dohare et al. 2024): it is so committed to old
    tasks it can no longer fit new ones. We report the slope of ``R[i][i]`` vs
    ``i`` by least squares: **negative slope = loss of plasticity**, ~0 = stable,
    positive = warming up. Plasticity and forgetting (:func:`backward_transfer`)
    are the two ends of the stability-plasticity trade-off, so report both.

    Returns ``nan`` for ``T < 2`` or fewer than two observed diagonal cells.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return float("nan")
    idx = np.arange(T, dtype=float)
    diag = np.array([R[i, i] for i in range(T)], dtype=float)
    mask = ~np.isnan(diag)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(idx[mask], diag[mask], 1)[0]
    return float(slope)
