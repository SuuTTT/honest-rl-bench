"""Population-level evaluation for multi-agent RL.

Single-agent "score X beats score Y" is a total order; strategic interaction is
not. Skill in games can be cyclic (rock-paper-scissors), so a win-rate against
one opponent is a point on a cycle, not a rung on a ladder. This module builds
the *payoff matrix* over a pool of policies and summarises it the honest way:

  - :func:`payoff_matrix` / :func:`win_rate` -- the who-beats-whom primitive;
  - :func:`elo` -- the familiar scalar rating, which *assumes transitivity* and
    therefore cannot detect cycles (use it, don't trust it to);
  - :func:`nash_averaging` -- a maximum-entropy Nash equilibrium over the
    antisymmetric logit-payoff, which is *invariant to redundant agents*
    (Balduzzi, Tuyls, Perolat & Graepel, "Re-evaluating Evaluation", NeurIPS
    2018, https://arxiv.org/abs/1806.02643);
  - :func:`nontransitivity` -- the magnitude of the cyclic (rock-paper-scissors)
    component of the payoff, i.e. how wrong a scalar ladder would be.

The empirical-game framing (treat the pool as a game and analyse its
meta-strategies) follows the PSRO line (Lanctot et al., "A Unified
Game-Theoretic Approach to Multiagent Reinforcement Learning", NeurIPS 2017,
https://arxiv.org/abs/1711.00832).

Pure numpy; no external dependencies.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "payoff_matrix",
    "win_rate",
    "elo",
    "nash_averaging",
    "nontransitivity",
]


def payoff_matrix(results, agents: Sequence | None = None) -> np.ndarray:
    """Build the mean payoff matrix ``P[i][j]`` of agent ``i`` vs agent ``j``.

    Accepts either:
      - a 2-D array-like already shaped ``(n, n)`` (returned as a float array), or
      - an iterable of match records ``(i, j, score)`` where ``score`` is agent
        ``i``'s payoff in a match against agent ``j`` (e.g. +1 win / 0 loss, or a
        mean return). Repeated ``(i, j)`` records are averaged. ``i``/``j`` may be
        integer indices or hashable labels; pass ``agents`` to fix the ordering,
        otherwise labels are sorted.

    Returns a float ``(n, n)`` matrix; unobserved cells are ``nan``. For a
    symmetric zero-sum game ``P[i][j] + P[j][i]`` is constant (e.g. 1 for
    win-rates), but no symmetry is imposed here.
    """
    arr = np.asarray(results, dtype=object)
    # Case 1: already a square numeric matrix.
    if isinstance(results, np.ndarray) and results.ndim == 2:
        return np.asarray(results, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and not _looks_like_records(results):
        return np.asarray(results, dtype=float)

    # Case 2: iterable of (i, j, score) records.
    records = list(results)
    labels = list(agents) if agents is not None else None
    if labels is None:
        seen = set()
        for i, j, _ in records:
            seen.add(i)
            seen.add(j)
        labels = sorted(seen, key=lambda x: (str(type(x)), x))
    index = {lab: k for k, lab in enumerate(labels)}
    n = len(labels)
    tot = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=float)
    for i, j, score in records:
        a, b = index[i], index[j]
        tot[a, b] += float(score)
        cnt[a, b] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(cnt > 0, tot / cnt, np.nan)
    return P


def _looks_like_records(results) -> bool:
    try:
        first = next(iter(results))
    except (TypeError, StopIteration):
        return False
    return hasattr(first, "__len__") and len(first) == 3 and np.isscalar(first[2])


def win_rate(payoff: np.ndarray) -> np.ndarray:
    """Average payoff of each agent over the pool: ``mean_j P[i][j]`` (nan-safe).

    The naive scalar summary. It is *not* redundancy-invariant -- padding the
    pool with copies of a weak strategy an agent counters inflates its average --
    which is exactly the failure :func:`nash_averaging` fixes (Balduzzi et al.
    2018). Diagonal/self-play and missing cells are ignored.
    """
    P = np.asarray(payoff, dtype=float)
    out = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        row = P[i].copy()
        row[i] = np.nan  # ignore self-play
        if np.any(~np.isnan(row)):
            out[i] = float(np.nanmean(row))
    return out


def elo(results, agents: Sequence | None = None, k: float = 32.0,
        base: float = 1500.0, scale: float = 400.0, n_epochs: int = 50,
        seed: int = 0) -> np.ndarray:
    """Iterative Elo ratings from match outcomes (Elo 1978).

    ``results`` is an iterable of ``(i, j, score)`` records where ``score`` is in
    ``[0, 1]`` (1 = ``i`` beat ``j``, 0.5 = draw); a mean win-rate works too. Each
    record nudges the two ratings toward the observed outcome by the standard Elo
    update ``r_i += k * (score - expected)`` with logistic expectation
    ``1 / (1 + 10**((r_j - r_i)/scale))``. Records are shuffled and replayed for
    ``n_epochs`` passes so the ratings settle (order-independence). Returns the
    ratings in the order of ``agents`` (or sorted labels).

    Elo collapses a possibly-cyclic relation onto one number: it *assumes
    transitivity*. A high Elo can hide that you lose to a specific counter, so
    report :func:`nontransitivity` alongside it (Balduzzi et al. 2018).
    """
    records = list(results)
    if agents is not None:
        labels = list(agents)
    else:
        seen = set()
        for i, j, _ in records:
            seen.add(i)
            seen.add(j)
        labels = sorted(seen, key=lambda x: (str(type(x)), x))
    index = {lab: k_ for k_, lab in enumerate(labels)}
    r = np.full(len(labels), float(base))
    rng = np.random.default_rng(seed)
    order = np.arange(len(records))
    for _ in range(max(1, n_epochs)):
        rng.shuffle(order)
        for idx in order:
            i, j, score = records[idx]
            a, b = index[i], index[j]
            expected = 1.0 / (1.0 + 10.0 ** ((r[b] - r[a]) / scale))
            update = k * (float(score) - expected)
            r[a] += update
            r[b] -= update
    return r


def _antisymmetric_logits(payoff: np.ndarray) -> np.ndarray:
    """Antisymmetric logit-payoff used by Nash averaging (Balduzzi et al. 2018).

    For a win-probability matrix in [0,1], the logit ``log(p/(1-p))`` turns it
    into an antisymmetric zero-sum matrix ``A`` with ``A = -A.T``; for a matrix
    already centred around a constant we antisymmetrise directly. We detect the
    [0,1] case and use the logit, else fall back to ``(P - P.T)/2``.
    """
    P = np.asarray(payoff, dtype=float)
    P = np.where(np.isnan(P), 0.5 if _in_unit(P) else 0.0, P)
    if _in_unit(P):
        eps = 1e-9
        Pc = np.clip(P, eps, 1 - eps)
        A = np.log(Pc / (1 - Pc))
        return (A - A.T) / 2.0
    return (P - P.T) / 2.0


def _in_unit(P: np.ndarray) -> bool:
    vals = P[~np.isnan(P)]
    return vals.size > 0 and float(np.nanmin(P)) >= 0.0 and float(np.nanmax(P)) <= 1.0


def nash_averaging(payoff: np.ndarray, iters: int = 5000, lr: float = 0.1,
                   tol: float = 1e-9) -> np.ndarray:
    """Maximum-entropy Nash over the antisymmetric logit-payoff (Balduzzi 2018).

    Computes a distribution ``p`` over agents that is a (maxent) Nash equilibrium
    of the symmetric zero-sum game ``A = antisymmetric-logit(payoff)``: the
    meta-game where each player picks an agent and the row player's payoff is
    ``A``. Because the game is symmetric and antisymmetric, the value is 0 and a
    maxent Nash exists; the resulting weights are *invariant to redundant
    agents*, so duplicating a weak strategy cannot inflate anyone's score
    (Balduzzi, Tuyls, Perolat & Graepel, "Re-evaluating Evaluation", NeurIPS
    2018). Returns the Nash weight vector (sums to 1); read a high weight as
    "this agent is hard for the population to avoid playing".

    Solved by entropic mirror ascent / multiplicative-weights self-play on the
    symmetric game (a simple, correct method for symmetric zero-sum games): for a
    uniform start, iterate ``p <- softmax(log p + lr * A p)`` averaged over time.
    For a transitive ladder this concentrates on the best agent; for
    rock-paper-scissors it returns the uniform distribution.
    """
    A = _antisymmetric_logits(payoff)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0])
    log_p = np.full(n, -np.log(n))
    avg = np.zeros(n)
    p = np.exp(log_p)
    for t in range(1, iters + 1):
        grad = A @ p  # payoff of each pure strategy vs current mixture
        log_p = log_p + lr * grad
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= p.sum()
        avg += p
        if t > 1 and lr * float(np.max(np.abs(grad))) < tol:
            break
    avg /= avg.sum()
    return avg


def nontransitivity(payoff: np.ndarray) -> float:
    """Magnitude of the cyclic (rock-paper-scissors) component of the payoff.

    Decomposes the antisymmetric part ``A = (P - P.T)/2`` into a *transitive*
    component (a pure ladder, ``A_t[i,j] = r_i - r_j`` for some ratings ``r``)
    plus a *cyclic* residual, the discrete-Hodge / Schur decomposition behind the
    "games look like spinning tops" picture (Balduzzi et al. 2018; Czarnecki et
    al. 2020). The transitive ratings are ``r = mean_j A[i,j]`` (the least-squares
    fit of a ladder to ``A``); the cyclic residual is ``C = A - (r_i - r_j)``.

    Returns the fraction of the antisymmetric "energy" that is cyclic,
    ``||C||^2 / ||A||^2`` in [0, 1]: ~0 means a scalar ranking is faithful, ~1
    means it is meaningless (a perfect cycle, e.g. rock-paper-scissors). Returns
    0.0 for a degenerate all-equal payoff.
    """
    P = np.asarray(payoff, dtype=float)
    P = np.where(np.isnan(P), 0.0, P)
    A = (P - P.T) / 2.0
    n = A.shape[0]
    if n < 2:
        return 0.0
    r = A.mean(axis=1)  # least-squares ladder ratings
    trans = r[:, None] - r[None, :]
    cyclic = A - trans
    energy = float(np.sum(A ** 2))
    if energy == 0.0:
        return 0.0
    return float(np.sum(cyclic ** 2) / energy)
