"""Sanity tests for honest_rl_bench.stats.

Run:  python -m pytest tests/ -q     (or  python tests/test_stats.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honest_rl_bench.stats import (  # noqa: E402
    bootstrap_ci,
    bootstrap_ci_from_scores,
    coef_of_variation,
    iqm,
    linear_decode_r2,
    mean,
    mean_ci,
    median,
    paired_bootstrap,
    paired_diff_bootstrap,
    performance_profile,
    prob_improvement,
)


def test_iqm_ignores_tails():
    # Middle 50% is all 10s; extreme tails are wild outliers. IQM must ignore them.
    xs = [-1000.0, 10, 10, 10, 10, 10, 10, 1000.0]
    assert abs(iqm(xs) - 10.0) < 1e-9
    # The plain mean would be dragged around; IQM is robust.
    assert abs(np.mean(xs) - 10.0) > 1.0


def test_iqm_small_sample_falls_back_to_mean():
    xs = [1.0, 2.0, 3.0]
    assert abs(iqm(xs) - 2.0) < 1e-9


def test_bootstrap_ci_brackets_iqm():
    rng = np.random.default_rng(0)
    xs = rng.normal(100, 10, size=40)
    lo, hi = bootstrap_ci(xs, fn=iqm, n=4000, seed=1)
    point = iqm(xs)
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_ci_stratified_runs():
    # Stratified resampling (within groups) should still bracket the IQM.
    rng = np.random.default_rng(2)
    xs = rng.normal(50, 5, size=20)
    groups = np.array([i % 4 for i in range(20)])  # 4 seed-groups
    lo, hi = bootstrap_ci(xs, fn=iqm, n=4000, seed=3, groups=groups)
    assert lo <= iqm(xs) <= hi


def test_mean_ci_brackets_mean():
    rng = np.random.default_rng(4)
    xs = rng.normal(7, 2, size=30)
    m, lo, hi = mean_ci(xs, n=4000, seed=5)
    assert abs(m - np.mean(xs)) < 1e-9
    assert lo <= m <= hi


def test_paired_diff_identical_straddles_zero():
    # Identical inputs → the per-pair difference is exactly 0 everywhere, so the
    # CI must straddle (contain) 0 and the comparison must be non-significant.
    xs = [1.0, 5.0, 3.0, 8.0, 2.0, 6.0]
    res = paired_diff_bootstrap(xs, xs, n=3000, seed=0)
    assert res["lo"] <= 0 <= res["hi"]
    assert res["significant"] is False
    assert abs(res["diff"]) < 1e-9


def test_paired_diff_detects_real_shift():
    # b is uniformly 50 points below a on every pair → strongly significant, A>B.
    rng = np.random.default_rng(6)
    a = rng.normal(200, 10, size=12)
    b = a - 50.0
    res = paired_diff_bootstrap(a, b, n=4000, seed=7)
    assert res["significant"] is True
    assert res["lo"] > 0
    assert res["prob_a_better"] > 0.99


def test_paired_diff_requires_aligned_lengths():
    try:
        paired_diff_bootstrap([1, 2, 3], [1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


def test_performance_profile_monotonic():
    rng = np.random.default_rng(8)
    xs = rng.normal(0, 1, size=100)
    taus, fracs = performance_profile(xs, num_taus=40)
    # Fractions are non-increasing as tau rises.
    assert np.all(np.diff(fracs) <= 1e-12)
    # At/below the min everything passes; above the max nothing does.
    assert fracs[0] == 1.0
    assert fracs[-1] <= 1.0 / xs.size + 1e-9


def test_performance_profile_known_values():
    xs = [0.0, 1.0, 2.0, 3.0]
    taus = [0.0, 1.5, 2.5, 4.0]
    _, fracs = performance_profile(xs, taus=taus)
    assert list(fracs) == [1.0, 0.5, 0.25, 0.0]


def test_mean_median_wrappers():
    xs = [1.0, 2.0, 3.0, 100.0]
    assert abs(mean(xs) - np.mean(xs)) < 1e-12
    assert abs(median(xs) - np.median(xs)) < 1e-12
    assert np.isnan(mean([]))
    assert np.isnan(median([]))


def test_coef_of_variation():
    xs = [10.0, 10.0, 10.0, 10.0]
    assert abs(coef_of_variation(xs)) < 1e-12          # no spread -> CV 0
    spread = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(coef_of_variation(spread) - (np.std(spread) / np.mean(spread))) < 1e-12
    assert np.isnan(coef_of_variation([1.0, -1.0]))    # zero mean -> undefined


def test_bootstrap_ci_from_scores_matches_bootstrap_ci():
    rng = np.random.default_rng(0)
    xs = rng.normal(100, 10, size=40)
    est, (lo, hi) = bootstrap_ci_from_scores(xs, aggregate="iqm", n=4000, ci=0.95, seed=1)
    blo, bhi = bootstrap_ci(xs, fn=iqm, n=4000, alpha=0.05, seed=1)
    assert abs(est - iqm(xs)) < 1e-9
    assert abs(lo - blo) < 1e-9 and abs(hi - bhi) < 1e-9
    assert lo <= est <= hi


def test_paired_bootstrap_is_alias():
    assert paired_bootstrap is paired_diff_bootstrap


def test_prob_improvement_identical_is_half():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    p, (lo, hi) = prob_improvement(xs, xs, n=2000, seed=0)
    assert abs(p - 0.5) < 1e-9
    assert lo <= 0.5 <= hi


def test_prob_improvement_dominance():
    rng = np.random.default_rng(3)
    a = rng.normal(100, 5, size=30)
    b = a - 50.0
    p, (lo, hi) = prob_improvement(a, b, n=3000, seed=1)
    assert p > 0.99
    assert lo > 0.5


def test_linear_decode_r2_perfect_when_target_is_a_feature():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, size=200)
    noise = rng.normal(0, 1, size=200)
    X = np.column_stack([y, noise])      # y is literally one of the features
    assert linear_decode_r2(X, y) > 0.999


def test_linear_decode_r2_low_for_unrelated():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, size=(300, 3))
    y = rng.normal(0, 1, size=300)       # independent of X
    assert linear_decode_r2(X, y) < 0.2


def test_linear_decode_r2_zero_variance_target_is_nan():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.full(10, 5.0)
    assert np.isnan(linear_decode_r2(X, y))


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
