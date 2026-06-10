"""Sanity tests for honest_rl_bench.marl (population evaluation)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honest_rl_bench.marl import (  # noqa: E402
    elo,
    nash_averaging,
    nontransitivity,
    payoff_matrix,
    win_rate,
)

# Rock-paper-scissors win-probability matrix (0=R, 1=P, 2=S): row beats col?
RPS = np.array([
    [0.5, 0.0, 1.0],   # rock loses to paper, beats scissors
    [1.0, 0.5, 0.0],   # paper beats rock, loses to scissors
    [0.0, 1.0, 0.5],   # scissors loses to rock, beats paper
])

# A clean transitive ladder: A > B > C.
LADDER = np.array([
    [0.5, 0.8, 0.9],
    [0.2, 0.5, 0.8],
    [0.1, 0.2, 0.5],
])


def test_payoff_matrix_from_records_averages():
    records = [(0, 1, 1.0), (0, 1, 0.0), (1, 0, 1.0)]
    P = payoff_matrix(records)
    assert P.shape == (2, 2)
    assert abs(P[0, 1] - 0.5) < 1e-12   # (1+0)/2
    assert abs(P[1, 0] - 1.0) < 1e-12
    assert np.isnan(P[0, 0])            # unobserved


def test_payoff_matrix_passthrough():
    P = payoff_matrix(RPS)
    assert np.allclose(P, RPS)


def test_win_rate_uniform_for_rps():
    wr = win_rate(RPS)
    # Every RPS strategy wins half its off-diagonal games on average.
    assert np.allclose(wr, 0.5)


def test_nontransitivity_high_for_rps_low_for_ladder():
    assert nontransitivity(RPS) > 0.95     # almost entirely cyclic
    assert nontransitivity(LADDER) < 0.05  # essentially a pure ladder


def test_nash_averaging_uniform_for_rps():
    nash = nash_averaging(RPS)
    assert np.all(nash >= 0)
    assert abs(nash.sum() - 1.0) < 1e-9
    assert np.allclose(nash, 1.0 / 3, atol=0.02)  # symmetric -> uniform


def test_nash_averaging_concentrates_on_ladder_top():
    nash = nash_averaging(LADDER)
    # The dominant strategy (index 0) should carry the most Nash weight.
    assert np.argmax(nash) == 0
    assert nash[0] > nash[1] >= nash[2] - 1e-9


def test_elo_orders_ladder():
    records = []
    for i in range(3):
        for j in range(3):
            if i != j:
                records.append((i, j, LADDER[i, j]))
    r = elo(records, n_epochs=200, seed=0)
    # Ratings must respect the transitive order A > B > C.
    assert r[0] > r[1] > r[2]


def test_elo_ties_rps():
    records = []
    for i in range(3):
        for j in range(3):
            if i != j:
                records.append((i, j, RPS[i, j]))
    r = elo(records, n_epochs=300, seed=0)
    # Symmetric cycle -> ratings should be close together (no real ladder).
    assert (r.max() - r.min()) < 80.0


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
