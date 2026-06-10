---
layout: page
title: "Mechanism-check: spend a day, save a month"
permalink: /mechanism-check/
---

# Mechanism-check: spend a day, save a month

The statistics and reporting disciplines on the other pages tell you how to *evaluate* a
method honestly. This page is about something cheaper and earlier: how to decide
**whether to run the experiment at all.** The single most cost-effective practice in the
[tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass) campaign was the
**mechanism-check** — a tiny test of the *assumption* a method depends on, run before any
multi-seed, multi-week campaign. It killed several ideas in an afternoon that would
otherwise have cost weeks each. And paired with it is the rule the project had to learn
the hard way: **never report a number you didn't read back from disk.**

## The idea: kill the mechanism, not the method

Every method rests on a load-bearing **assumption** — some structure in the data or the
model that, if absent, dooms the method no matter how well you tune it. A full evaluation
tests "does it work?" and needs the whole campaign. A mechanism-check tests the cheaper,
prior question:

> **Does the thing the method *assumes* even exist?**

That question is usually answerable on a *frozen checkpoint* in an afternoon — no
training, no fan-out. If the assumption fails, you've falsified the method for the cost of
a probe script. If it holds, you've earned the right to spend compute.

The contrast that defines the project: the original Glass effort spent **months** tuning a
"turn-Glass-off-at-1M" schedule. An afternoon mechanism-check — *"does Glass's structure
track anything control actually needs?"* — would have returned **no** immediately. They
paid the multi-week price once, then made the check mandatory.

## The case study: assumptions, kill-tests, verdicts

Here is the actual scoreboard from the campaign. Each lever names the assumption it needs,
the cheap test, and what the probe found — *every one of which the later full campaign
confirmed.*

| lever | assumption it needs | cheap kill-test | result | verdict |
|---|---|---|---|---|
| value-equivalent macro head | value is *hard* to recover from the latent (there's headroom) | linear V-decode R² on a frozen checkpoint | **0.9994** | no headroom — killed |
| value-critical adaptive horizon | criticality *varies* across states | coefficient of variation of criticality | **0.36** (near-uniform), only ~3% flat states | nothing to gate — killed |
| SE-k adaptive jump-length | community boundaries mark where the model *errs* | Spearman(boundary score, k-step error) | **+0.09 / −0.18** | uncorrelated — killed |
| uncertainty-gated horizon | error *varies* under planning perturbations | error inflation under MPPI-scale action noise | **1.06×** | uniform error, nothing to gate — killed |

Each test is hours, not weeks. Each one predicted the campaign verdict that followed. When
the team *did* spend the compute anyway (to corroborate), the multi-seed runs agreed —
e.g. the value-equivalence head not only failed to help but **hurt** (PandaPickCube −1076
peak / −1327 final vs the matched baseline), exactly as "no headroom" predicted.

And the discipline cuts both ways: a mechanism-check is also how you earn a green light.
The jumpy world model was *not* killed, because its assumption held — the k-step head is
measurably more accurate than iterating the 1-step model, and the edge *grows* with k
(`jumpy_err/iter1_err`: 0.99 at k=2 → 0.82 at k=8). Mechanism confirmed first, campaign
second; the win survived peak *and* final.

The toolkit ships the **generic** halves of a mechanism-check — a linear-decode
R² (`stats.linear_decode_r2`) and a coefficient of variation
(`stats.coef_of_variation`). It deliberately does **not** ship the
*task-specific* halves: loading a particular checkpoint's latents
(`load_probe`, below) and computing a domain quantity like per-state value
criticality (`value_criticality`) depend on your model and environment, so they
can't be a faithful library function. You write those few lines yourself, then
feed the result to the generic statistics. The snippet below is illustrative:
`load_probe`/`value_criticality` are **pseudocode stand-ins for your own probe
code**, while `linear_decode_r2` and `coef_of_variation` are real functions.

```python
from honest_rl_bench import stats

# --- task-specific (YOU write these for your model/env; pseudocode here) -------
# A mechanism-check is a cheap probe on a FROZEN checkpoint, not a training run.
latents, values = load_probe("checkpoints/panda_pickcube.pt")  # your loader
criticality     = value_criticality(latents, values)           # your domain metric

# --- generic (shipped by the toolkit) -----------------------------------------
# "Is there headroom for a value-equivalence objective?" -> linear V-decode R^2.
r2 = stats.linear_decode_r2(latents, values)        # 0.9994  -> no headroom

# "Is there spread to adapt a horizon to?" -> coefficient of variation.
cv = stats.coef_of_variation(criticality)           # 0.36 -> ~uniform

if r2 > 0.99 or cv < 0.5:
    print("Mechanism absent. Do NOT fan out. Saved a multi-week campaign.")
```

## How to design a mechanism-check

1. **Name the assumption.** Write the one sentence that must be true for the method to
   possibly help. ("Some states are decision-critical and others aren't." "The error is
   non-uniform, so adapting the horizon can pay.")
2. **Find the cheapest measurement of it.** Usually a probe on a frozen checkpoint: a
   linear decode R², a correlation, a coefficient of variation, a ratio under perturbation.
   No training.
3. **Pre-register the bar.** "If R² > 0.99 there's no headroom; kill it." Decide the
   threshold *before* you run the probe — same discipline as the
   [pre-registered gates]({{ '/cherry-picking/' | relative_url }}).
4. **If the mechanism is absent, stop.** Write the null and move on. If it's present, *now*
   spend the compute on the full pre-registered, peak-and-final, CI-gated campaign.

This is "spend a day, save a month" stated as a procedure. It is also the only honest way
to run a large negative-results campaign without going broke: thirteen abstraction levers
came back null, but most were called cheaply, so the budget went to the few that earned a
real run.

## The anti-fabrication rule: read it back from disk

The second half of the discipline is even simpler, and it is the project's most-repeated
rule because the project **broke it repeatedly**: the tdmpc-glass git history contains
roughly **seven separate corrections of fabricated or misremembered numbers** — figures
quoted from memory or from a notebook cell that no longer matched what was on disk. Every
one was caught and corrected, and out of that came the rule that now governs every claim:

> **Never report a number you didn't read back from disk.**

Concretely:

- **Numbers come from artifacts, not memory or chat.** Every figure in the case-study
  blogs is annotated as read from run CSVs or probe JSON — *"All numbers are read from run
  CSVs / probe JSON, not notebooks."*
- **The metric is computed by code, from the file, at report time.** Don't transcribe a
  number into prose and trust it later; re-derive it. That's the entire design intent of
  this toolkit's `report` command — point it at the runs directory and it reads the CSVs
  and computes IQM, CIs, peak and final itself, so the numbers in the report cannot drift
  from the numbers on disk.
- **If you can't point at the file the number came from, you don't have the number.** A
  remembered "+104%" that turns out to be peak/final confusion (it was +44% peak) is the
  cautionary tale: the prose was confident and wrong because nobody re-read the artifact.

```bash
# The honest report: numbers are computed from the CSVs at report time, never transcribed.
python -m honest_rl_bench.report examples/runs --out report.html
#   -> reads step,return,seed; computes IQM + bootstrap CI + peak AND final per arm.
```

Mechanism-check before fan-out saves you compute. Read-it-back-from-disk saves you
credibility. Together they are the cheapest two habits in this whole tutorial, and the two
that the case study would most have liked to have had from the start.

<div class="takeaways" markdown="1">

### Takeaways

- Before a multi-week campaign, run a **mechanism-check**: test the *assumption* the method needs, on a **frozen checkpoint**, in an afternoon.
- **Kill the mechanism, not the method** — "does the thing it assumes even exist?" is far cheaper than "does it work?"
- **Pre-register the bar** for the probe, just like an evaluation gate. Absent mechanism → stop and write the null. Present → spend the compute.
- A mechanism-check also *green-lights*: the one real win was confirmed cheaply (k-step error 0.99→0.82) before any campaign.
- **Never report a number you didn't read back from disk** — this project fabricated/misremembered numbers ~7× before adopting the rule.
- Compute the metric **from the artifact at report time**; if you can't point at the file, you don't have the number.

</div>

**Back to:** [Continual & multi-task RL evaluation]({{ '/catastrophic-forgetting/' | relative_url }}) ·
[Report uncertainty]({{ '/statistics/' | relative_url }}) ·
[home]({{ '/' | relative_url }})

### References
- Case study: [tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass) — "Eight Mirages" (Part 2) and "The Latent Was Already the Abstraction" (Part 3) field reports; `RESEARCH_LEDGER.md`.
- Ni, Eysenbach, et al. (2024). *Bridging State and History Representations.* (self-predictive objectives as sufficient abstractions — the theory the probes confirmed.)
