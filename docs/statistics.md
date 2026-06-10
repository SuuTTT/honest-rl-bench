---
layout: page
title: "Report uncertainty, not point estimates"
permalink: /statistics/
---

# Report uncertainty, not point estimates

The single most common statistical sin in deep RL is the **point estimate**: a mean
return over three seeds, plotted as one curve, reported as one number. It is fast, it
is conventional, and it is a lie of omission. Returns in deep RL are heavy-tailed and
multimodal — some seeds find the good basin, some don't — so three samples tell you
almost nothing about where the true performance sits. This page is about reporting the
uncertainty you actually have.

## Why mean ± std over 3 seeds lies

Three problems compound:

1. **The mean is not robust.** A single lucky or unlucky seed drags it around. In our
   case study, one configuration's IQM read `0.818 → 0.736 → 0.829 → … → 0.767` as
   seeds accumulated — crossing "significant win" and "confirmed null" several times
   before settling in overlap. The point estimate at *any* snapshot was a coin flip.
   The same campaign also produced a single worst seed of `0.127` in an arm whose mean
   otherwise looked fine — a tail the mean quietly absorbs.

2. **±std is the wrong interval.** Standard deviation describes the *spread of seeds*,
   not your *uncertainty about the aggregate*. With n=3 it is also estimated from three
   points, so the error bar itself is noise. Readers interpret overlapping ±std bars as
   "no difference" and non-overlapping ones as "difference" — both inferences are unsound.

3. **Three seeds is underpowered.** Agarwal et al. (2021), *Deep Reinforcement Learning
   at the Edge of the Statistical Precipice*, show that the small-sample regime typical
   of deep RL produces confidence intervals so wide that most published "improvements"
   are statistically indistinguishable from noise. Henderson et al. (2018), *Deep
   Reinforcement Learning That Matters*, demonstrated empirically that changing only the
   random seed can flip which algorithm "wins." Colas et al. (2018, *How Many Random
   Seeds?*) give the uncomfortable arithmetic: detecting a small effect with decent power
   often needs **10–20+** seeds, not 3–5.

## IQM: a robust aggregate

The **interquartile mean (IQM)** discards the bottom and top 25% of runs and averages
the middle 50%. It is far more robust to outliers than the mean and far more
*statistically efficient* than the median (it uses 50% of the data, not one point). For
RL's heavy-tailed, multimodal score distributions it is the recommended default
(Agarwal et al. 2021).

```python
from honest_rl_bench import stats

runs = stats.load_runs("examples/runs")              # parses step,return,seed CSVs
final = stats.final_scores(runs, "stable_baseline", last_k=2)  # last-2-eval per seed

print(stats.iqm(final))                              # robust aggregate over seeds
print(stats.mean(final), stats.median(final))        # for contrast — don't ship these alone
```

## Stratified bootstrap confidence intervals

Don't report a point — report a **range you'd defend**. The
**stratified bootstrap** resamples seeds *with replacement, within each task*, and
recomputes the aggregate thousands of times to trace out its sampling distribution. The
2.5th and 97.5th percentiles are your 95% CI. "Stratified" matters when you aggregate
across tasks: it preserves the per-task structure so one heavily-sampled task can't
dominate.

```python
# 95% stratified-bootstrap CI on the IQM, resampling seeds within each task.
# bootstrap_ci_from_scores takes a flat array of per-run scores and returns
# (estimate, (lo, hi)); pass groups=<task-id-per-score> to stratify the resample.
scores = stats.final_scores(runs, "stable_baseline")
est, (lo, hi) = stats.bootstrap_ci_from_scores(
    scores, aggregate="iqm", n=20_000, ci=0.95,
)
print(f"IQM {est:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
```

Report the interval, not the point. In the tdmpc-glass campaign the one real win was
stated this way: jumpy world model vs vanilla on PandaPickCube, **peak +966 (+44%),
95% CI [714, 1248]; final +1266 (+88%), CI [877, 1642]** — both intervals clear of zero.
A win you can write as a CI that excludes zero is a win; a win that lives in CI overlap
is a hypothesis.

## Performance profiles and probability of improvement

Two aggregate summaries beat a table of means when you compare across many tasks:

- **Performance profiles** plot, for every threshold τ, the fraction of runs scoring
  above τ. The whole curve is shown, so saturation and floor effects are visible and a
  reader can read off any operating point. Profiles that don't cross indicate
  **stochastic dominance** — a much stronger statement than "higher mean."
- **Probability of improvement** P(X > Y) estimates how often a run of method X beats a
  run of method Y, with a bootstrap CI. It answers the question readers actually have
  ("if I switch, am I likely to be better off?") without assuming Gaussian scores.

```python
# performance_profile takes a flat score array and returns (taus, fractions);
# prob_improvement takes the two methods' per-seed score arrays.
jumpy   = stats.final_scores(runs, "lucky_cherry")
vanilla = stats.final_scores(runs, "stable_baseline")

taus, fracs     = stats.performance_profile(jumpy)
p_imp, (lo, hi) = stats.prob_improvement(jumpy, vanilla)
print(f"P(jumpy > vanilla) = {p_imp:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
```

## How many seeds?

There is no universal number, but there is a procedure: **decide the effect size you
care about, then run enough seeds for the CI to resolve it — and decide both *before*
you look.** Practical guidance distilled from the literature:

- **3 seeds:** a sanity check, never a claim. Report it as "preliminary."
- **5 seeds:** the bare floor for a marginal-effect claim, *with* IQM + bootstrap CIs,
  and only if the CI clears zero. Most "5-seed wins" don't.
- **10–20+ seeds:** what Colas et al. estimate you need to detect small effects with
  real power. If your effect needs 20 seeds to show up, ask whether it matters.
- **Report a CI either way.** Three seeds with an honest (wide) interval is more useful
  than ten seeds reported as a bare mean.

A corollary the campaign paid for: the *estimate trajectory* matters. Plot the aggregate
as seeds accumulate. If it wanders across your decision threshold (as the behavioral-Glass
IQM did, three times), you do not yet have a result — you have noise that hasn't averaged out.

<div class="takeaways" markdown="1">

### Takeaways

- **Never ship a bare mean.** Use **IQM** (robust + efficient) for the aggregate.
- **Report a CI, not a point** — a **stratified bootstrap** CI over seeds, stratified by task.
- A win is a **CI that excludes zero**; a win inside CI overlap is a hypothesis.
- Prefer **performance profiles** and **probability of improvement** over tables of means.
- **3 seeds is a sanity check.** Pick your effect size, then size your seeds — *before* you look.
- Watch the **estimate trajectory**: if it crosses your threshold as seeds accumulate, you have no result yet.

</div>

**Next:** [The ways RL results mislead — and how to stop]({{ '/cherry-picking/' | relative_url }}) ·
[back to home]({{ '/' | relative_url }})

### References
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS. (rliable)
- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). *Deep Reinforcement Learning That Matters.* AAAI.
- Colas, Sigaud, Oudeyer (2018). *How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments.*
