---
layout: page
title: "MARL-specific pitfalls"
permalink: /marl/
---

# MARL-specific pitfalls

Everything from the [statistics]({{ '/statistics/' | relative_url }}) and
[cherry-picking]({{ '/cherry-picking/' | relative_url }}) pages still applies in
multi-agent RL — but MARL adds failure modes that don't exist in single-agent
benchmarking. The core issue: **there is no fixed environment.** Your opponents and
teammates are themselves learning, the reward landscape moves, and "performance" is only
defined *relative to whom you played*. A win-rate against one opponent is not a measure of
skill; it is a measure of a matchup.

## Non-transitivity: A beats B beats C beats A

In single-agent RL, "better" is a total order — score X beats score Y, end of story. In
games with strategic interaction, **skill is not a total order.** The textbook example is
rock-paper-scissors: rock beats scissors, scissors beats paper, paper beats rock. No
policy is "best." Real MARL environments are full of these cycles, often hidden inside an
apparently linear skill axis.

This breaks the most common MARL claim outright: *"our agent achieves 95% win-rate
against the baseline."* If the strategy space is non-transitive, you can hand-pick (or
accidentally select) an opponent your agent happens to counter, post a crushing win-rate,
and lose to a third policy you never evaluated against. A single opponent's win-rate is a
**point on a cycle**, not a position on a ladder.

The honest move is to **evaluate against a population**, and to measure non-transitivity
explicitly rather than assume it away.

## Why a single opponent's win-rate misleads

Three concrete traps:

1. **Opponent cherry-picking.** Choosing the evaluation opponent after seeing results is
   best-of-N (see [cherry-picking]({{ '/cherry-picking/' | relative_url }})) in the
   opponent dimension. Pre-register the opponent pool.
2. **Overfitting to a fixed opponent.** Train against one frozen opponent and you learn a
   *counter* to it, not a robust policy. It will look great in eval (same opponent) and
   collapse against anything else.
3. **Self-play inflation.** "Win-rate vs an earlier copy of myself" measures
   *improvement over time*, not *absolute skill*. A policy can climb steadily against its
   own past and still be beaten by a simple fixed strategy it never trained against —
   classic in self-play that cycles.

## Population and tournament evaluation

Instead of one matchup, evaluate over a **pool** of policies (past checkpoints, baselines,
scripted bots, other methods) and summarize the whole payoff matrix:

- **Elo / TrueSkill** rank agents from pairwise outcomes. Cheap and familiar, but Elo
  *assumes transitivity* — it collapses a possibly-cyclic relation onto a single number,
  so a high Elo can hide that you lose to a specific counter. Use it, but don't trust it
  to detect cycles.
- **Nash averaging** (Balduzzi et al., 2018, *Re-evaluating Evaluation*) computes a
  maximum-entropy Nash equilibrium over the evaluation pool and scores agents by their
  weight in it. It is **invariant to redundant agents** — padding the pool with many
  copies of a weak strategy can't inflate your apparent skill, which Elo and naive
  average-win-rate both allow. This is the principled answer to "which agent is best
  *over this population*."
- **Relative population performance / empirical game-theoretic analysis** (the PSRO line,
  Lanctot et al., 2017) treats the policy pool as an empirical game and analyzes its
  meta-strategies — surfacing cycles and exploitability directly instead of hiding them
  behind an average.

```python
from honest_rl_bench import marl

# Match records: (i, j, score) = policy i's payoff vs policy j (win=1, loss=0,
# or a mean return), one row per seeded match. Repeated (i, j) are averaged.
records = [
    (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0),   # rock-paper-scissors cycle
    (1, 0, 0.0), (2, 1, 0.0), (0, 2, 0.0),
]
payoff = marl.payoff_matrix(records)         # P[i][j], symmetric or row-vs-col
# (you can also pass an already-built (n, n) matrix straight through)

elo  = marl.elo(records)                  # familiar, assumes transitivity
nash = marl.nash_averaging(payoff)        # redundancy-invariant population score
cyc  = marl.nontransitivity(payoff)       # cycle / rock-paper-scissors detector

print("Elo:", elo)
print("Nash score:", nash)
print("non-transitivity index:", cyc)     # ~1 means a single ladder is the wrong model
```

Report **Nash averaging or RPP as the headline**, show the **payoff matrix** so readers
can see who-beats-whom, and report a **non-transitivity index** so the reader knows
whether a scalar ranking is even meaningful.

## Self-play vs fixed-pool evaluation

These answer different questions; report both and don't conflate them.

- **Self-play curves** ("win-rate vs my own past") measure *training progress*. They are
  prone to **cycling**: the agent forgets how to beat strategies it has moved past, so the
  curve goes up while the policy goes in circles. A self-play curve climbing is *necessary
  but not sufficient* evidence of getting better.
- **Fixed-pool evaluation** (vs a held-out, frozen set of diverse opponents, including
  scripted and prior-method policies) measures *absolute* skill against things the agent
  cannot have overfit to. This is the closer analog of a single-agent test set, and it is
  what should gate any "our method is stronger" claim.

A robust protocol: train however you like (self-play, league, PBT), but **gate on a
held-out fixed pool** with population-level aggregation, pre-registered before you look.

## Non-stationarity

Because every agent is learning, each agent faces a **moving target**: the transition and
reward dynamics it experiences change as others adapt. This means a single snapshot of
"performance" can be transient — the same peak-vs-final problem from
[cherry-picking]({{ '/cherry-picking/' | relative_url }}), amplified. A policy that
dominates the population at iteration *k* may be routinely exploited by iteration *k+1*.
Mitigations: evaluate against a **frozen** pool (so the yardstick doesn't move), report
performance **over training** rather than at one snapshot, and check **exploitability**
(how much a best-response can beat your "final" policy) as a stability measure, not just
its average score.

A practical note on uncertainty: in MARL the score variance comes from *two* sources —
seeds **and** opponents. Bootstrap over both. A CI that only resamples seeds against a
single fixed opponent understates your uncertainty badly.

<div class="takeaways" markdown="1">

### Takeaways

- **Skill in games is not a total order.** A>B>C>A cycles are everywhere — a single opponent's win-rate is a point on a cycle, not a rung on a ladder.
- **Evaluate against a population**, report the **payoff matrix**, and headline with **Nash averaging / RPP** (redundancy-invariant), not average win-rate.
- **Elo assumes transitivity** — useful, but it can't *detect* cycles. Report a **non-transitivity index**.
- **Self-play curves measure progress; fixed-pool eval measures skill.** Gate claims on a frozen, diverse, held-out pool.
- **Non-stationarity** makes single snapshots transient — freeze the yardstick, report over training, and check **exploitability**.
- **Bootstrap over seeds *and* opponents.**

</div>

**Next:** [Continual & multi-task RL evaluation]({{ '/catastrophic-forgetting/' | relative_url }}) ·
[The ways RL results mislead]({{ '/cherry-picking/' | relative_url }}) ·
[back to home]({{ '/' | relative_url }})

### References
- Balduzzi, Tuyls, Pérolat, Graepel (2018). *Re-evaluating Evaluation.* NeurIPS. (Nash averaging)
- Lanctot et al. (2017). *A Unified Game-Theoretic Approach to Multiagent RL.* NeurIPS. (PSRO / empirical game-theoretic analysis)
- Czarnecki et al. (2020). *Real World Games Look Like Spinning Tops.* (transitive vs cyclic structure of games)
