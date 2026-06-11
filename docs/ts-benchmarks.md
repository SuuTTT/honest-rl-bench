---
layout: page
title: Time-Series Benchmarks
permalink: /ts-benchmarks/
---

# Honest Time-Series Forecasting Benchmarks

The long-horizon forecasting literature (LTSF) has a benchmark-integrity problem
that mirrors the RL one this site started with: published comparison tables mix
protocols, copy stale baseline numbers, and occasionally leak test information —
so "SOTA" claims often measure experimental setup, not architecture. This page
catalogues the failure modes, the critique literature documenting them, and a
case study from our own measurements.

## A taxonomy of how benchmark numbers go wrong

**1. Copied (stale) baseline numbers.** Papers paste baseline rows from earlier
papers' tables instead of rerunning them. The copied number was produced under a
different protocol — or was weak to begin with — and then propagates for years.
Once one paper inherits a weak CrossFormer or iTransformer row, every follow-up
"beats" it too.

**2. Protocol heterogeneity sold as architecture gains.** Look-back length,
model width, training epochs, learning rate, and batch size all shift MSE by
amounts comparable to claimed architectural improvements. Comparing your
hyperparameter-searched model against baselines at someone else's fixed config
is the single most common pattern. The honest test: rerun *every* model under
*one* protocol, then separately report tuned-per-method results if you must.

**3. Test-set leakage.** Recurring concrete forms:
- normalisation statistics fitted on the full series instead of the training
  split only;
- early stopping or model selection on the *test* split;
- the `drop_last` evaluation bug — discarding the final incomplete test batch,
  which silently drops the hardest windows and inflates scores. TFB (Qiu et
  al., PVLDB 2024) documented this in widely used pipelines.

**4. Seed and run cherry-picking.** Reporting the best of N seeds, omitting
variance, never running significance tests. With 3-seed std on PEMS comparable
to the inter-method gaps at short horizons, a single lucky seed can "win" a
benchmark row.

**5. Dataset selection bias.** Reporting only the datasets where the method
wins; quietly dropping the channel-rich datasets (Traffic, PEMS07) where a
proposed channel-mixing scheme underperforms, or the small-N datasets (ETT)
where it adds nothing.

**6. Divergence laundering.** When a baseline diverges in some seeds, the
choices are: report the diverged mean (inflates your gain), silently rerun, or
report convergence rates. Almost nobody states which they did. If your gain
over a baseline comes from the baseline blowing up in 1/3 seeds, that is a
stability claim, not an accuracy claim — label it as such.

**7. Capacity confounds.** A new architecture evaluated at 4x the parameter
count or training budget of its baselines. Parameter-matched and budget-matched
comparisons are the exception, not the rule.

## Critique literature worth reading

- **Zeng, Chen, Zhang, Xu — "Are Transformers Effective for Time Series
  Forecasting?" (AAAI 2023).** One-layer linear models beat sophisticated
  Transformers across the standard LTSF suite — the canonical demonstration
  that the field's comparison tables were not measuring what they claimed.
- **Elsayed et al. — "Do We Really Need Deep Learning Models for Time Series
  Forecasting?" (arXiv:2101.02118).** A well-configured GBRT matches or beats
  most deep baselines of its era.
- **Qiu et al. — "TFB: Towards Comprehensive and Fair Benchmarking of Time
  Series Forecasting Methods" (PVLDB 2024).** Systematic fair re-benchmark;
  documents pipeline bugs (including drop-last evaluation) and shows method
  rankings reorder under a clean protocol.
- **Shao et al. — BasicTS / "Exploring Progress in Multivariate Time Series
  Forecasting" (TKDE).** Fair benchmarking infrastructure for multivariate and
  spatio-temporal forecasting; shows heterogeneity across datasets dominates
  many claimed advances.
- **Hewamalage, Ackermann, Bergmeir — "Forecast evaluation for data
  scientists: common pitfalls and best practices" (DMKD 2023).** The most
  complete pitfalls catalogue: leakage, metric choice, aggregation traps.
- **Bergmeir & Benítez — "On the use of cross-validation for time series
  predictor evaluation" (Information Sciences 2012).** Foundational treatment
  of why naive CV leaks in temporal data and what to do instead.
- **Godahewa et al. — Monash Time Series Forecasting Archive (NeurIPS 2021
  D&B).** Standardised datasets + protocols precisely to stop ad-hoc
  evaluation drift.
- **Makridakis et al. — the M4/M5 competition analyses (IJF).** Blind, held-out
  competitions repeatedly rank methods very differently from self-reported
  benchmark tables.

Cross-domain reality checks with the same lesson: Dacrema et al., "Are We
Really Making Much Progress?" (RecSys 2019); Musgrave et al., "A Metric
Learning Reality Check" (ECCV 2020); Henderson et al., "Deep RL that Matters"
(AAAI 2018); Picard, "torch.manual_seed(3407) is all you need"
(arXiv:2109.08203).

## Case study: the PEMS leaderboard is protocol soup

While preparing a structural-entropy forecasting paper we reran standard
baselines under the canonical Time-Series-Library protocol (input 96, 3 seeds,
fixed budget) and compared against the numbers circulating in recent Mamba-era
papers. Two findings, both verified from our own `metrics.npy` outputs:

**Published CrossFormer numbers understate it badly.** On PEMS03, published
tables (propagated from iTransformer's Table 9) report MSE 0.121 (pred-24) and
0.262 (pred-96). Our self-run CrossFormer under the canonical protocol:
**0.0874** and **0.1997** — roughly 30% better than its own published row, and
strong enough to beat several methods that "beat CrossFormer" in their tables.

**The same model spans a 1.7x range across papers' protocols.** Canonical-
protocol iTransformer on PEMS03 pred-96 scores 0.2755; the iTransformer row
cited in recent Mamba papers reports 0.164 for the same model and dataset —
a stronger tuned configuration. Any method comparing itself against one of
these numbers while training under the other regime gains (or loses) more from
the protocol than most architectures contribute.

Consequence: ranking claims among recent PEMS forecasters (CrossFormer,
iTransformer, S-Mamba, DMamba, ...) are not currently decidable from published
tables alone. We are running a single-protocol re-benchmark of all of them —
progress on the [dashboard]({{ "/dashboard/" | relative_url }}).

## The checklist we hold ourselves to

1. Rerun every baseline under one canonical protocol; never copy table rows.
2. At least 3 seeds; report mean and per-seed values; release the raw
   per-seed JSON with the paper.
3. Read every reported number from the metrics file (`metrics.npy`/logs) —
   never retype from memory; keep the provenance log.
4. Fit scalers and graphs on the training split only; no test-window
   information anywhere upstream.
5. Report all datasets attempted, including the nulls and negatives.
6. State divergence handling explicitly; never launder instability into
   accuracy gains.
7. Significance-test headline comparisons (Diebold-Mariano for forecasts).
8. Separate "same-budget" from "tuned-per-method" tables when both exist.
