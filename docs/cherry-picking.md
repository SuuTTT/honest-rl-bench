---
layout: page
title: "The ways RL results mislead — and how to stop"
permalink: /cherry-picking/
---

# The ways RL results mislead — and how to stop

Most misleading RL results are not fabricated. They are **selected**. You run many
things, you look at the numbers, and you report the framing that looks best. Every step
feels honest in isolation; the bias enters through the *selection*, after you've seen the
data. This page catalogs the common selection effects and the small set of disciplines
that defuse them. The running examples are from the [tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass)
campaign, where each of these mistakes was made, caught, and corrected.

## 1. Peak vs final — the big one

Pick the best checkpoint and you measure the **peak** of a noisy curve; read the last
evaluations and you measure the **final** performance. These can tell *opposite stories*,
and choosing one after seeing the data is cherry-picking.

The case study makes this vivid. On PandaPickCube, **vanilla TD-MPC2 collapses late** —
it loses ~35–45% between its peak and its final return. Measured on **final**, the jumpy
world model looked like a massive `+104%` win. But much of that gap was the *baseline
falling apart*, not the new method climbing. The fair best-checkpoint (**peak**)
comparison shrank the headline to `+44%`. Both numbers are true; reporting only one is a
distortion. The larger final gap is real too — but it's a **stability** finding ("jumpy
resists late collapse"), not a "plans better" finding. They are different claims and
deserve different words.

The mirror-image trap: a jumpy-on-CartpoleSparse "growing lead" **reversed entirely** by
450k steps — visible only if you read at ≥400k, not at the 250k snapshot where it looked
like a win.

**The rule: report peak *and* final, for every arm, always.** Best-checkpoint is a
legitimate *deployment* claim — but only if applied identically to all arms and disclosed
as peak. Gate on whichever you pre-committed to.

```python
from honest_rl_bench import stats

runs  = stats.load_runs("examples/runs")
peak  = stats.peak_scores(runs, "lucky_cherry")             # best eval per seed
final = stats.final_scores(runs, "lucky_cherry", last_k=2)  # last-2-eval per seed

for label, s in [("peak", peak), ("final", final)]:
    est, (lo, hi) = stats.bootstrap_ci_from_scores(s, aggregate="iqm")
    print(f"{label:5s} IQM {est:.1f}  95% CI [{lo:.1f}, {hi:.1f}]")
```

## 2. Best-of-N seed selection

Run N seeds, report the best one (or quietly drop the worst), and you have inflated your
result by exactly the spread of the seed distribution — which in deep RL is large. This
is the seed-luck that produced tdmpc-glass's original HopperHop "win": restarts and
population-based selection cherry-picked basin entries that **vanished under a clean
single-variable protocol** (neither arm even entered the high-reward basin: best 323 vs
286). If you ran 10 seeds, you report the aggregate over 10 — not the top 3.

## 3. Checkpoint / snapshot selection

Reading the curve at the step where your method happens to lead is best-of-N in the time
dimension. Pre-commit to an evaluation budget and a read step (e.g. "report at ≥400k,
last-2 evals"). The CartpoleSparse reversal above is what snapshot-shopping buys you.

## 4. Tuning on the test tasks

If you tune hyperparameters, environment variants, or architecture choices *on the same
tasks you report*, your "evaluation" is really training. The fix is the oldest one in ML:
separate the tasks (or seeds) you tune on from the ones you report on, and disclose the
tuning budget for *every* arm — including the baseline. An under-tuned baseline is
cherry-picking too. (tdmpc-glass found a real example: `rho`, a consistency-horizon knob,
helps PandaPickCube but *hurts* sparse tasks — a per-task tuning lever masquerading as an
architectural win until it was tested across tasks.)

## 5. Comparing marginal CIs instead of the paired difference

A subtle one. Two methods can have **overlapping** marginal confidence bands and still be
**reliably different**, because the seeds are *paired* (same task, often comparable
conditions) and the per-seed difference has much lower variance than either arm alone. The
correct, more powerful test for a head-to-head is the **paired-difference bootstrap**:
resample seeds, compute the per-resample difference, and CI *that*.

This is exactly how the one real win was stated honestly. The marginal bands for jumpy and
vanilla on Panda **lightly touch** at the final step — so a naive "the bars overlap"
reading would call it a null. But the **paired difference** is CI-separated (+992 peak,
+1106 final, both clear of zero). The campaign showed the overlapping marginal bands
*anyway*, because hiding the overlap would be the very thing the project exists to fight —
and reported the paired CI as the actual test.

```python
# Paired-difference bootstrap: the correct head-to-head test. It takes the two
# methods' per-seed peak arrays (aligned seed-for-seed) and returns a dict with
# the difference, its CI, and whether the CI is separated from zero. (For runs
# where seeds aren't paired, use stats.prob_improvement instead.)
a = stats.peak_scores(runs, "lucky_cherry")     # "jumpy"
b = stats.peak_scores(runs, "stable_baseline")  # "vanilla"
res = stats.paired_bootstrap(a, b, n=20_000, alpha=0.05)
sep = "separated" if res["significant"] else "OVERLAPS zero"
print(f"jumpy - vanilla (peak): {res['diff']:+.0f}  "
      f"95% CI [{res['lo']:.0f}, {res['hi']:.0f}]  -> {sep}")
```

## 6. Pre-registered gates

The antidote to all selection effects is to **decide before you look**. Write down, in
advance: how many seeds, which read step, which metric (peak/final/both), and the
threshold that counts as a win — e.g. *"≥10% improvement with non-overlapping paired CI on
≥3 of 4 tasks."* Then run, and report whatever the gate returns. Pre-registration is what
turns "we found a configuration that wins" into "this method wins." In the case study, the
pre-registered, CI-separated gate is what let eight mirages be called as nulls instead of
quietly written up at their flattering snapshots.

## Do / don't checklist

**Do**
- Report **peak AND final** for every arm.
- Aggregate over **all** seeds you ran; report the count.
- Use the **paired-difference bootstrap** for A-vs-B.
- **Pre-register** seeds, read step, metric, and win threshold before looking.
- Disclose the **tuning budget for the baseline too**.
- Publish the **estimate trajectory**, not just the final table.

**Don't**
- Pick the best checkpoint *after* seeing which arm it favors.
- Drop "bad" seeds or report best-of-N.
- Read the curve at the step where you happen to lead.
- Tune on the tasks you report.
- Call a win from **overlapping marginal CIs** — use the paired test.
- Tune the new method hard and the baseline lazily.

<div class="takeaways" markdown="1">

### Takeaways

- Misleading RL results are usually **selected, not fabricated** — the bias enters *after* you see the data.
- **Peak and final can tell opposite stories. Report both, every arm.**
- Best-of-N seeds, snapshot-shopping, and tuning-on-test are the same sin in different dimensions.
- For head-to-heads use the **paired-difference bootstrap**, not overlapping marginal bars.
- **Pre-register the gate** (seeds, step, metric, threshold). Then report what it returns.

</div>

**Next:** [MARL-specific pitfalls]({{ '/marl/' | relative_url }}) ·
[Report uncertainty]({{ '/statistics/' | relative_url }}) ·
[back to home]({{ '/' | relative_url }})

### References
- Henderson et al. (2018). *Deep Reinforcement Learning That Matters.* AAAI. (peak-picking bias)
- Agarwal et al. (2021). *Deep RL at the Edge of the Statistical Precipice.* NeurIPS.
- Case study: [tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass), "Eight Mirages" field report.
