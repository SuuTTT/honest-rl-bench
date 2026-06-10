---
layout: page
title: "Continual & multi-task RL evaluation"
permalink: /catastrophic-forgetting/
---

# Continual & multi-task RL evaluation

When an agent learns a *sequence* of tasks, a single final number is dangerously
incomplete. An agent can ace the task it just finished while having silently destroyed
everything it learned before — **catastrophic forgetting**. Reporting only the last
task's performance hides exactly the thing continual learning is supposed to prevent. This
page is about evaluation matrices and the metrics that expose forgetting, backward and
forward transfer, and plasticity.

## What forgetting is

Train on task A, then task B. Gradient updates that improve B can overwrite the
representation that solved A, so performance on A drops — sometimes to scratch. That's
catastrophic forgetting (McCloskey & Cohen, 1989). In RL it is worse than in supervised
learning: the data distribution is non-stationary *within* a task too, and the agent
controls its own data, so a forgotten skill also means forgotten *exploration* of that
task. A model that scores 95% on task B after a 5-task curriculum tells you nothing about
tasks A, C, D — and the headline number is usually B.

## The evaluation matrix

The right primitive is not a scalar but a **matrix**. Train on tasks `1..T` in sequence;
after finishing each task `i`, evaluate on *every* task `j` (including ones not yet seen).
Call the result `R[i][j]` = performance on task `j` after training through task `i`.

```
              eval task j
            ┌─────────────────────
after  i=1  │ R11  R12  R13  ...     <- R1j for j>1 = forward transfer (zero/few-shot)
train  i=2  │ R21  R22  R23  ...
task   i=3  │ R31  R32  R33  ...
       ...  │  ...
       i=T  │ RT1  RT2  RT3 ... RTT   <- bottom row: final performance on every task
            └─────────────────────
                    diagonal Rii = performance right after learning task i
```

Every metric below is a different read of this matrix. Build the matrix first; derive the
scalars second. **Reporting only `R[T][T]` (the bottom-right corner) is the continual-RL
version of reporting only the final checkpoint** — it's the same selection sin as
peak-vs-final from the [cherry-picking]({{ '/cherry-picking/' | relative_url }}) page.

```python
from honest_rl_bench import continual

# Records: (i, j, score) = score on task j after training through task i (0-based),
# averaged over seeds per cell. You can also pass an already-built (T, T) matrix.
records = [
    (0, 0, 1.0), (0, 1, 0.0), (0, 2, 0.0),   # after task 0
    (1, 0, 0.7), (1, 1, 1.0), (1, 2, 0.0),   # after task 1 (forgot some of task 0)
    (2, 0, 0.4), (2, 1, 0.6), (2, 2, 1.0),   # after task 2 (forgot more)
]
R = continual.eval_matrix(records)        # shape (T, T), per-seed under the hood

print(continual.average_performance(R))   # mean of final row R[T-1][*]
print(continual.backward_transfer(R))     # forgetting (negative = forgot)
print(continual.forward_transfer(R))      # learning unseen tasks early
print(continual.plasticity(R))            # ability to still learn new tasks late
```

## The metrics

Following the standard continual-learning definitions (Lopez-Paz & Ranzato, 2017, GEM;
Chaudhry et al., 2018):

- **Average performance** — the mean of the final row, `mean_j R[T][j]`. The headline
  "how good is the agent across all tasks at the end." Necessary, but it can be propped up
  by a few easy tasks while a hard one was forgotten — always report the *per-task* final
  row alongside the average.

- **Backward transfer (BWT) = forgetting.** For each earlier task `j`, compare its score
  *right after you learned it* (`R[j][j]`) to its score *at the end* (`R[T][j]`):
  `BWT = mean_{j<T} (R[T][j] - R[j][j])`. **Negative BWT is forgetting** (the end-of-run
  score is worse than the just-learned score); positive BWT means later tasks *helped*
  earlier ones. This is the number a final-task report hides, and it is usually the one
  that matters most.

- **Forward transfer (FWT).** How much does training on earlier tasks help a *future*
  task before you train on it? Compare `R[i-1][i]` (zero/few-shot on task `i` before
  training it) to a from-scratch reference. Positive FWT means the curriculum builds
  reusable structure.

- **Plasticity (a.k.a. loss of plasticity).** Can the agent *still learn* new tasks late
  in the sequence? Track the diagonal `R[i][i]` as `i` grows: if it decays, the network
  has lost plasticity (Dohare et al., 2024) — it's so committed to old tasks it can no
  longer fit new ones. Plasticity and forgetting are the two ends of the
  stability–plasticity trade-off, and a method that wins on one often pays on the other,
  so **report both.**

## Why a single final number hides all of this

Consider two agents, both reporting "final average = 0.80" over 5 tasks:

- Agent X: `[0.80, 0.80, 0.80, 0.80, 0.80]` — flat, no forgetting (BWT ≈ 0).
- Agent Y: `[0.40, 0.60, 0.90, 1.00, 1.10]` — recency-biased; great at recent tasks,
  badly forgot the first two (BWT strongly negative).

Same average, opposite continual-learning behavior. Only the **per-task final row + BWT**
distinguishes them. This is the continual-RL echo of the
[peak-vs-final]({{ '/cherry-picking/' | relative_url }}) lesson: one aggregate, multiple
stories underneath, and the choice of what to report is where the misleading happens.

Two more practical notes:
- Every cell `R[i][j]` is itself an aggregate over seeds, so the
  [statistics]({{ '/statistics/' | relative_url }}) page applies *inside the matrix* —
  use IQM and bootstrap CIs per cell, and propagate them to BWT/FWT.
- **Pre-register the task sequence and ordering.** Task order changes forgetting
  dramatically, so reporting the friendliest ordering is cherry-picking; fix it before you
  look, or report across several random orderings.

<div class="takeaways" markdown="1">

### Takeaways

- A **final-task number hides forgetting.** Build the **evaluation matrix** `R[i][j]` first; derive scalars second.
- Report **average performance *and* the per-task final row** — averages can be propped up by easy tasks.
- **Backward transfer (BWT) is forgetting**: `R[T][j] − R[j][j]`, negative = forgot. It's the number the headline hides.
- Report **forward transfer** (does the curriculum help future tasks?) and **plasticity** (can it still learn late?) — the two ends of stability–plasticity.
- Each matrix cell is a seed aggregate: use **IQM + CIs inside the matrix**.
- **Pre-register the task ordering** — order strongly affects forgetting.

</div>

**Next:** [Mechanism-check: spend a day, save a month]({{ '/mechanism-check/' | relative_url }}) ·
[MARL pitfalls]({{ '/marl/' | relative_url }}) ·
[back to home]({{ '/' | relative_url }})

### References
- Lopez-Paz & Ranzato (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS. (BWT / FWT definitions)
- Chaudhry, Dokania, Ajanthan, Torr (2018). *Riemannian Walk for Incremental Learning.* ECCV. (forgetting / intransigence)
- Dohare et al. (2024). *Loss of Plasticity in Deep Continual Learning.* Nature.
- McCloskey & Cohen (1989). *Catastrophic Interference in Connectionist Networks.*
