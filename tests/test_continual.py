"""Sanity tests for honest_rl_bench.continual (forgetting metrics)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honest_rl_bench.continual import (  # noqa: E402
    average_performance,
    backward_transfer,
    eval_matrix,
    forward_transfer,
    plasticity,
)

# A forgetting example: each task is solved when learned (diagonal=1.0) but
# earlier tasks decay by the end (lower-left below the diagonal).
FORGET = np.array([
    [1.0, 0.0, 0.0],   # after task 0
    [0.7, 1.0, 0.0],   # after task 1 (task 0 fell 1.0 -> 0.7)
    [0.4, 0.6, 1.0],   # after task 2 (task 0 -> 0.4, task 1 -> 0.6)
])

# No forgetting: earlier tasks stay solved.
STABLE = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
])


def test_eval_matrix_from_records():
    records = [(0, 0, 1.0), (0, 0, 0.0), (1, 1, 0.8)]
    R = eval_matrix(records)
    assert R.shape == (2, 2)
    assert abs(R[0, 0] - 0.5) < 1e-12     # averaged
    assert abs(R[1, 1] - 0.8) < 1e-12
    assert np.isnan(R[0, 1])              # unobserved


def test_eval_matrix_passthrough():
    R = eval_matrix(FORGET)
    assert np.allclose(R, FORGET)


def test_average_performance_is_final_row_mean():
    assert abs(average_performance(FORGET) - np.mean([0.4, 0.6, 1.0])) < 1e-12


def test_backward_transfer_negative_when_forgetting():
    bwt = backward_transfer(FORGET)
    # task0: 0.4-1.0=-0.6 ; task1: 0.6-1.0=-0.4 ; mean = -0.5
    assert bwt < 0
    assert abs(bwt - (-0.5)) < 1e-12


def test_backward_transfer_zero_when_stable():
    assert abs(backward_transfer(STABLE)) < 1e-12


def test_forward_transfer():
    # Zero-shot entries above the diagonal are all 0 here, baseline 0 -> FWT 0.
    assert abs(forward_transfer(FORGET)) < 1e-12
    # Give positive zero-shot transfer and confirm it is detected.
    R = FORGET.copy()
    R[0, 1] = 0.3
    R[1, 2] = 0.5
    assert forward_transfer(R) > 0


def test_plasticity_slope():
    # FORGET has a flat diagonal (all 1.0) -> ~zero slope.
    assert abs(plasticity(FORGET)) < 1e-9
    # A network losing plasticity: diagonal decays as tasks accrue.
    decay = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.3],
    ])
    assert plasticity(decay) < 0


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
