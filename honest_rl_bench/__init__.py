"""honest-rl-bench: an honest RL benchmarking toolkit.

Point it at a folder of run CSVs (``step,return,seed`` + optional ``eval_type``)
and get robust aggregate statistics, peak-vs-final reporting, paired A/B
comparison, a static report card, and a live learning-curve dashboard.

Statistics follow rliable (Agarwal et al., NeurIPS 2021): IQM + stratified
bootstrap CIs instead of fragile means; paired-difference bootstrap for honest
head-to-head comparison; performance profiles for the whole distribution.

It also covers MARL (population/payoff-matrix evaluation -- :mod:`marl`) and
continual / multi-task RL (evaluation matrices and forgetting metrics --
:mod:`continual`), plus generic mechanism-check probes (e.g.
:func:`linear_decode_r2`, :func:`coef_of_variation`).

Public API::

    from honest_rl_bench import (
        iqm, mean, median, coef_of_variation,
        bootstrap_ci, bootstrap_ci_from_scores, mean_ci,
        paired_diff_bootstrap, paired_bootstrap, prob_improvement,
        performance_profile, linear_decode_r2,
        load_runs, peak_scores, final_scores, RunSet, build_report,
    )
    from honest_rl_bench import marl, continual
"""
from . import continual, marl
from .curves import GroupAggregate, RunSet, SeedCurve, load_runs
from .report import build_report
from .stats import (
    bootstrap_ci,
    bootstrap_ci_from_scores,
    coef_of_variation,
    final_scores,
    iqm,
    linear_decode_r2,
    mean,
    mean_ci,
    median,
    paired_bootstrap,
    paired_diff_bootstrap,
    peak_scores,
    performance_profile,
    prob_improvement,
)

__version__ = "0.1.0"

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
    "RunSet",
    "SeedCurve",
    "GroupAggregate",
    "build_report",
    "marl",
    "continual",
    "__version__",
]
