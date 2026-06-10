"""Curve loading + per-group aggregation for honest RL benchmarking.

Reads a directory of run CSVs (columns ``step,return,seed`` plus an optional
``eval_type``), groups runs by *algorithm/group* (taken from a subdirectory name
or a filename prefix), and produces, per group:

  - the per-(group, seed) raw curve,
  - an aggregate IQM curve with a stratified bootstrap CI band over seeds,
  - per-seed ``peak`` (max over the curve) and ``final`` (mean of the last K
    evals) -- the peak-vs-final pair is the single most common way RL results
    mislead, so we always carry both.

Generalised from the tdmpc-glass dashboard data layer (``read_curve``,
``eval_summary`` peak/final logic, ``discover_csvs``, ``build_phase_ci``) but
decoupled from any fleet/SSH/queue concerns: it just points at a local folder.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .stats import bootstrap_ci, iqm, paired_diff_bootstrap

DEFAULT_FINAL_K = 5  # "final" = mean of the last K evals (sustained, not peak)


@dataclass
class SeedCurve:
    """One run's learning curve plus its peak/final scalars."""
    group: str
    seed: str
    steps: np.ndarray
    returns: np.ndarray
    eval_type: str | None = None
    path: str = ""

    @property
    def peak(self) -> float:
        return float(np.max(self.returns)) if self.returns.size else float("nan")

    def final(self, k: int = DEFAULT_FINAL_K) -> float:
        """Mean of the last ``k`` evals (by step order). Robust to a one-eval spike."""
        if self.returns.size == 0:
            return float("nan")
        order = np.argsort(self.steps, kind="stable")
        tail = self.returns[order][-k:]
        return float(np.mean(tail))


@dataclass
class GroupAggregate:
    """Per-group aggregate: seeds, IQM curve + CI band, peak/final pools."""
    group: str
    seeds: list[SeedCurve] = field(default_factory=list)
    grid: np.ndarray = field(default_factory=lambda: np.array([]))
    iqm_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_lo: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_hi: np.ndarray = field(default_factory=lambda: np.array([]))
    counts: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def peaks(self) -> np.ndarray:
        return np.array([c.peak for c in self.seeds], dtype=float)

    def finals(self, k: int = DEFAULT_FINAL_K) -> np.ndarray:
        return np.array([c.final(k) for c in self.seeds], dtype=float)


@dataclass
class RunSet:
    """A loaded directory of runs, grouped by algorithm/group."""
    groups: dict[str, GroupAggregate]
    final_k: int = DEFAULT_FINAL_K

    def group_names(self) -> list[str]:
        return sorted(self.groups)

    def compare(self, group_a: str, group_b: str, **kw) -> dict:
        """Paired-difference bootstrap of group_a vs group_b on peak AND final.

        Pairs seeds by their seed id (only seeds present in BOTH groups are
        compared -- that is what makes it a *paired* test). Returns a dict with
        ``peak`` and ``final`` sub-dicts (see
        :func:`honest_rl_bench.stats.paired_diff_bootstrap`) plus the list of
        paired seeds.
        """
        ga, gb = self.groups[group_a], self.groups[group_b]
        by_seed_a = {c.seed: c for c in ga.seeds}
        by_seed_b = {c.seed: c for c in gb.seeds}
        shared = sorted(set(by_seed_a) & set(by_seed_b))
        a_peak = np.array([by_seed_a[s].peak for s in shared])
        b_peak = np.array([by_seed_b[s].peak for s in shared])
        a_fin = np.array([by_seed_a[s].final(self.final_k) for s in shared])
        b_fin = np.array([by_seed_b[s].final(self.final_k) for s in shared])
        return {
            "group_a": group_a,
            "group_b": group_b,
            "paired_seeds": shared,
            "peak": paired_diff_bootstrap(a_peak, b_peak, **kw),
            "final": paired_diff_bootstrap(a_fin, b_fin, **kw),
        }


# ─── CSV reading ────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Read one CSV → (steps, returns, eval_types, seed_from_column).

    Tolerant of column naming: accepts ``return`` or ``reward`` for the score,
    ``step`` for the x-axis, and optional ``eval_type`` / ``seed`` columns.
    Returns parsed arrays plus any per-file seed found in a ``seed`` column
    (empty string if none).
    """
    steps: list[float] = []
    rets: list[float] = []
    ets: list[str] = []
    seed_col = ""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            field_lower = {(k or "").lower(): k for k in (reader.fieldnames or [])}
            step_key = field_lower.get("step")
            ret_key = field_lower.get("return") or field_lower.get("reward")
            et_key = field_lower.get("eval_type")
            seed_key = field_lower.get("seed")
            for row in reader:
                try:
                    s = float(row[step_key]) if step_key else float(len(steps))
                    r = float(row[ret_key]) if ret_key else 0.0
                except (TypeError, ValueError, KeyError):
                    continue
                steps.append(s)
                rets.append(r)
                ets.append((row.get(et_key) or "").strip() if et_key else "")
                if seed_key and not seed_col:
                    seed_col = (row.get(seed_key) or "").strip()
    except OSError:
        pass
    return (np.asarray(steps, dtype=float), np.asarray(rets, dtype=float), ets, seed_col)


def _group_and_seed(path: Path, root: Path, seed_from_col: str) -> tuple[str, str]:
    """Infer (group, seed) for a CSV.

    group: the immediate parent directory name if the file lives in a subdir of
      ``root``; otherwise a filename prefix (text before ``_seed`` / ``seed_`` /
      the trailing seed token).
    seed: from a ``seed`` CSV column if present, else parsed from the filename
      (``seed_3``, ``..._seed3``, or a trailing ``_3``), else the file stem.
    """
    parent = path.parent
    if parent.resolve() != root.resolve():
        group = parent.name
    else:
        stem = path.stem
        m = re.match(r"(.+?)[_-]?seed[_-]?\d+$", stem, flags=re.IGNORECASE)
        if m:
            group = m.group(1).rstrip("_-")
        else:
            m2 = re.match(r"(.+?)[_-]\d+$", stem)
            group = m2.group(1) if m2 else stem
    seed = seed_from_col
    if not seed:
        stem = path.stem
        m = re.search(r"seed[_-]?(\d+)", stem, flags=re.IGNORECASE)
        if m:
            seed = m.group(1)
        else:
            m2 = re.search(r"[_-](\d+)$", stem)
            seed = m2.group(1) if m2 else stem
    return group, seed


# ─── Aggregation ────────────────────────────────────────────────────────────

def _build_aggregate(group: str, seeds: list[SeedCurve], n_boot: int,
                     seed_rng: int, max_grid: int = 150) -> GroupAggregate:
    """Build the IQM curve + stratified bootstrap CI band for one group.

    On a shared step grid (union of all seeds' steps, downsampled), each seed's
    value at a grid step is its most recent eval at-or-before that step
    (step-function hold, matching the tdmpc-glass ``build_phase_ci`` logic).
    The band is a bootstrap CI of the IQM across seeds at each grid step.
    """
    agg = GroupAggregate(group=group, seeds=seeds)
    all_steps = sorted({float(s) for c in seeds for s in c.steps.tolist()})
    if not all_steps:
        return agg
    if len(all_steps) > max_grid:
        stride = max(1, len(all_steps) // max_grid)
        all_steps = all_steps[::stride]
    grid = np.asarray(all_steps, dtype=float)
    iqm_curve, lo, hi, counts = [], [], [], []
    for gs in grid:
        vals = []
        for c in seeds:
            mask = c.steps <= gs
            if np.any(mask):
                # most recent eval at-or-before gs
                idx = np.argsort(c.steps[mask])[-1]
                vals.append(c.returns[mask][idx])
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            iqm_curve.append(np.nan); lo.append(np.nan); hi.append(np.nan); counts.append(0)
            continue
        iqm_curve.append(iqm(vals))
        if vals.size >= 2:
            clo, chi = bootstrap_ci(vals, fn=iqm, n=n_boot, seed=seed_rng)
        else:
            clo = chi = float(vals[0])
        lo.append(clo); hi.append(chi); counts.append(vals.size)
    agg.grid = grid
    agg.iqm_curve = np.asarray(iqm_curve, dtype=float)
    agg.ci_lo = np.asarray(lo, dtype=float)
    agg.ci_hi = np.asarray(hi, dtype=float)
    agg.counts = np.asarray(counts, dtype=int)
    return agg


def load_runs(directory: str | Path, eval_type: str | None = None,
              final_k: int = DEFAULT_FINAL_K, n_boot: int = 2000,
              seed_rng: int = 0) -> RunSet:
    """Load every ``*.csv`` under ``directory`` into a grouped :class:`RunSet`.

    Args:
      directory: folder of run CSVs (``step,return,seed`` + optional
        ``eval_type``). CSVs may sit directly in the folder (group from
        filename prefix) or in per-group subfolders (group from subdir name).
      eval_type: if given, keep only rows whose ``eval_type`` matches (e.g.
        ``"mppi"`` vs ``"pi"`` for planner-vs-policy). Rows with no eval_type
        column are always kept.
      final_k: number of trailing evals averaged for the per-seed ``final``.
      n_boot: bootstrap resamples for the per-group CI band (kept modest because
        it runs per grid point; the report/dashboard A/B tests use more).

    Returns a :class:`RunSet`. Multiple files mapping to the same (group, seed)
    are merged (steps concatenated).
    """
    root = Path(directory)
    raw: dict[tuple[str, str], SeedCurve] = {}
    for path in sorted(root.rglob("*.csv")):
        steps, rets, ets, seed_col = _read_csv(path)
        if steps.size == 0:
            continue
        if eval_type is not None and any(ets):
            keep = np.array([e == eval_type for e in ets])
            steps, rets = steps[keep], rets[keep]
            if steps.size == 0:
                continue
        group, seed = _group_and_seed(path, root, seed_col)
        key = (group, seed)
        if key in raw:
            prev = raw[key]
            prev.steps = np.concatenate([prev.steps, steps])
            prev.returns = np.concatenate([prev.returns, rets])
        else:
            raw[key] = SeedCurve(group=group, seed=seed, steps=steps, returns=rets,
                                 eval_type=eval_type, path=str(path))
    by_group: dict[str, list[SeedCurve]] = {}
    for (group, _seed), curve in raw.items():
        by_group.setdefault(group, []).append(curve)
    groups: dict[str, GroupAggregate] = {}
    for group, seeds in by_group.items():
        seeds.sort(key=lambda c: (int(c.seed) if c.seed.isdigit() else 1 << 30, c.seed))
        groups[group] = _build_aggregate(group, seeds, n_boot=n_boot, seed_rng=seed_rng)
    return RunSet(groups=groups, final_k=final_k)
