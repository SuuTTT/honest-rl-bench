# HANDOFF — honest-rl-bench (public spinoff)
**2026-07-01 04:29 UTC.**

## What this repo is
Public honest RL/MARL benchmarking toolkit + tutorial site (suuttt.github.io/honest-rl-bench), spun out of the private **TD-MPC-Glass** abstraction-redundancy campaign. This repo is the *public, sanitized* face; the live research + verified results live in the private repos on the EC2 control box:
- Code/blog: `tdmpc-glass`
- Results/paper: `wm-redundancy-paper` (ledger `bet2_null_results.md`, verdict `SYNTHESIS_beat_ppo.md`)
- Master handoff (with infra details, kept OFF public GitHub): `/home/ubuntu/HANDOFF_tdmpc-glass_2026-07-01.md`

## Status
No live jobs run from this repo. It's not part of the in-flight beat-PPO scan (that's on the vast boxes b3060/b3060b via the private stack). This repo's value is the methodology it packages: **honest RL benchmarking** — deterministic real-success eval, report n (seeds) + peak-vs-final, and every "beats X" gated by a **matched-budget** control (the class-controller budget trap). Those principles come straight from the campaign's hard-won lessons (the parent project fabricated numbers ~7× before this discipline was enforced).

## What to carry over if extending it
The campaign's clean, defensible findings worth turning into public benchmark cases:
- Structured prior (analytic skill + learned residual) = **sample-efficiency lever, not a ceiling lever** vs matched-budget PPO (Panda PickCube 0.716<0.810; OpenCabinet tie; ~1.6×/~7× faster).
- We beat PPO's ceiling only on **exploration-hard** tasks (HopperHop 367 vs 33). A live scan is testing dexterous manipulation (Leap in-hand) for more.
- JEPA/SimNorm anti-collapse is **downstream-dependent** (relational uniformity helps geometric latents, hurts value-based control).

## Next
Nothing required here for the live work. If publishing, sanitize (no box IPs/ports) and pull firmed numbers from `wm-redundancy-paper/bet2_null_results.md` only after the scan completes.
