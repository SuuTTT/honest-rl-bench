"""Static report card for a runs directory.

Usage:
    python -m honest_rl_bench.report <runs_dir> [--out report.html]
                                     [--eval-type mppi] [--final-k 5]

Prints a per-group table of IQM[CI] for peak and final, runs all pairwise
paired-difference A/B comparisons, and flags the two failure modes this toolkit
exists to catch:

  - **n < 5 seeds** -- aggregate statistics are unreliable below ~5 seeds
    (Agarwal et al. 2021); flagged so a reader does not over-trust the CI.
  - **peak >> final** -- a run that peaks then degrades (the classic
    "report the best checkpoint" trap). A large peak-minus-final gap means the
    headline peak number is not what the agent actually sustains.

Writes a self-contained HTML report card (no JS, no external deps beyond numpy).
"""
from __future__ import annotations

import argparse
import sys
from html import escape

import numpy as np

from .curves import RunSet, load_runs
from .stats import bootstrap_ci, iqm

MIN_SEEDS = 5            # below this, flag aggregate stats as low-confidence
PEAK_FINAL_GAP_FRAC = 0.10  # flag if (peak - final) / |peak| exceeds this


def _fmt(v: float, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _group_rows(rs: RunSet, n_boot: int) -> list[dict]:
    rows = []
    for name in rs.group_names():
        g = rs.groups[name]
        peaks, finals = g.peaks(), g.finals(rs.final_k)
        peak_iqm = iqm(peaks)
        final_iqm = iqm(finals)
        p_lo, p_hi = bootstrap_ci(peaks, fn=iqm, n=n_boot) if peaks.size else (np.nan, np.nan)
        f_lo, f_hi = bootstrap_ci(finals, fn=iqm, n=n_boot) if finals.size else (np.nan, np.nan)
        gap = peak_iqm - final_iqm
        denom = abs(peak_iqm) if peak_iqm not in (0, np.nan) else 1.0
        rows.append({
            "group": name,
            "n_seeds": g.n_seeds,
            "peak_iqm": peak_iqm, "peak_lo": p_lo, "peak_hi": p_hi,
            "final_iqm": final_iqm, "final_lo": f_lo, "final_hi": f_hi,
            "gap": gap,
            "low_seeds": g.n_seeds < MIN_SEEDS,
            "peak_final_gap": (not np.isnan(gap)) and gap > PEAK_FINAL_GAP_FRAC * denom,
        })
    return rows


def _comparisons(rs: RunSet, n_boot: int) -> list[dict]:
    names = rs.group_names()
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append(rs.compare(names[i], names[j], n=n_boot))
    return out


def _print_report(rows: list[dict], comps: list[dict], rs: RunSet, runs_dir: str):
    print(f"\n=== honest-rl-bench report card · {runs_dir} ===")
    print(f"    (final = mean of last {rs.final_k} evals; IQM with 95% bootstrap CI)\n")
    hdr = f"  {'group':<20} {'n':>3}  {'peak IQM [95% CI]':<26} {'final IQM [95% CI]':<26} {'gap':>7}  flags"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        peak = f"{_fmt(r['peak_iqm'])} [{_fmt(r['peak_lo'])},{_fmt(r['peak_hi'])}]"
        fin = f"{_fmt(r['final_iqm'])} [{_fmt(r['final_lo'])},{_fmt(r['final_hi'])}]"
        flags = []
        if r["low_seeds"]:
            flags.append(f"n<{MIN_SEEDS}")
        if r["peak_final_gap"]:
            flags.append("PEAK>>FINAL")
        fstr = ("  ⚠ " + ", ".join(flags)) if flags else ""
        print(f"  {r['group']:<20} {r['n_seeds']:>3}  {peak:<26} {fin:<26} {_fmt(r['gap']):>7}{fstr}")

    if comps:
        print("\n  pairwise paired-difference bootstrap (A − B; CI excludes 0 ⇒ significant):")
        for c in comps:
            for metric in ("peak", "final"):
                d = c[metric]
                if d["n_pairs"] == 0:
                    print(f"    {c['group_a']} vs {c['group_b']} [{metric}]: no shared seeds")
                    continue
                sig = "  *SIGNIFICANT*" if d["significant"] else ""
                print(f"    {c['group_a']} vs {c['group_b']} [{metric}]: "
                      f"Δ={_fmt(d['diff'])} [{_fmt(d['lo'])},{_fmt(d['hi'])}] "
                      f"P(A>B)={d['prob_a_better']:.2f} (n_pairs={d['n_pairs']}){sig}")
    else:
        print("\n  (only one group — no A/B comparison)")

    any_flags = any(r["low_seeds"] or r["peak_final_gap"] for r in rows)
    if any_flags:
        print("\n  ⚠ flags raised — see notes above. Peak>>Final means the headline peak")
        print("    is not sustained; n<5 means the CI is not trustworthy. Read peak AND final.")
    print()


def _html_report(rows: list[dict], comps: list[dict], rs: RunSet, runs_dir: str) -> str:
    def td(v, cls=""):
        c = f' class="{cls}"' if cls else ""
        return f"<td{c}>{escape(str(v))}</td>"

    trs = []
    for r in rows:
        flags = []
        if r["low_seeds"]:
            flags.append('<span class="flag">n&lt;5 seeds</span>')
        if r["peak_final_gap"]:
            flags.append('<span class="flag">peak&gt;&gt;final</span>')
        peak = f"{_fmt(r['peak_iqm'])} <span class=ci>[{_fmt(r['peak_lo'])}, {_fmt(r['peak_hi'])}]</span>"
        fin = f"{_fmt(r['final_iqm'])} <span class=ci>[{_fmt(r['final_lo'])}, {_fmt(r['final_hi'])}]</span>"
        trs.append(
            f"<tr><td class=mono>{escape(r['group'])}</td>"
            f"<td>{r['n_seeds']}</td><td>{peak}</td><td>{fin}</td>"
            f"<td>{_fmt(r['gap'])}</td><td>{' '.join(flags)}</td></tr>"
        )

    crows = []
    for c in comps:
        for metric in ("peak", "final"):
            d = c[metric]
            if d["n_pairs"] == 0:
                crows.append(f"<tr><td class=mono>{escape(c['group_a'])} vs {escape(c['group_b'])}</td>"
                             f"<td>{metric}</td><td colspan=3>no shared seeds</td></tr>")
                continue
            sig = '<span class="sig">significant</span>' if d["significant"] else '<span class=ns>n.s.</span>'
            crows.append(
                f"<tr><td class=mono>{escape(c['group_a'])} vs {escape(c['group_b'])}</td>"
                f"<td>{metric}</td>"
                f"<td>{_fmt(d['diff'])} <span class=ci>[{_fmt(d['lo'])}, {_fmt(d['hi'])}]</span></td>"
                f"<td>{d['prob_a_better']:.2f}</td><td>{sig}</td></tr>"
            )

    comp_table = ("<table><thead><tr><th>comparison</th><th>metric</th>"
                  "<th>Δ (A−B) [95% CI]</th><th>P(A&gt;B)</th><th></th></tr></thead>"
                  f"<tbody>{''.join(crows)}</tbody></table>") if crows else \
        "<p class=muted>Only one group — no A/B comparison.</p>"

    return f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<title>honest-rl-bench report · {escape(runs_dir)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e0f12;color:#dbe1eb;margin:0;font-size:14px}}
 .wrap{{max-width:960px;margin:0 auto;padding:24px}}
 h1{{font-size:20px}} h2{{font-size:15px;color:#4ec9b0;text-transform:uppercase;letter-spacing:.04em;margin-top:28px}}
 table{{width:100%;border-collapse:collapse;margin-top:8px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #2a2f3b}}
 th{{color:#7e8ba0;font-weight:500}}
 .mono{{font-family:ui-monospace,Menlo,monospace}}
 .ci{{color:#7e8ba0;font-size:12px}}
 .flag{{background:#54391c;color:#e0a44c;border-radius:10px;padding:1px 8px;font-size:11px;margin-right:4px}}
 .sig{{background:#1f3d22;color:#7dd87b;border-radius:10px;padding:1px 8px;font-size:11px}}
 .ns{{color:#7e8ba0;font-size:12px}} .muted{{color:#7e8ba0}}
 .note{{background:#161922;border:1px solid #2a2f3b;border-radius:6px;padding:10px 14px;color:#9099a8;font-size:12px;line-height:1.5}}
</style></head><body><div class=wrap>
<h1>honest-rl-bench report card</h1>
<p class=muted>{escape(runs_dir)} · final = mean of last {rs.final_k} evals · IQM with 95% bootstrap CI</p>
<h2>Per-group results</h2>
<table><thead><tr><th>group</th><th>n seeds</th><th>peak IQM [95% CI]</th>
<th>final IQM [95% CI]</th><th>peak−final</th><th>flags</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<h2>Paired A/B comparisons</h2>
{comp_table}
<div class=note style=margin-top:24px>
<b>How to read this.</b> Compare <b>final</b>, not peak — peak is the best checkpoint
ever seen and is inflated by best-of-N luck. A <b>peak&gt;&gt;final</b> flag means the
run degraded after its peak. <b>n&lt;5 seeds</b> means the CI is not trustworthy.
A/B claims use a <b>paired-difference bootstrap</b> on per-seed differences; only call
a difference real if its 95% CI excludes 0. (rliable: Agarwal et al. 2021.)
</div>
</div></body></html>"""


def build_report(runs_dir: str, eval_type: str | None = None, final_k: int = 5,
                 n_boot: int = 10000) -> tuple[list[dict], list[dict], RunSet]:
    rs = load_runs(runs_dir, eval_type=eval_type, final_k=final_k)
    rows = _group_rows(rs, n_boot)
    comps = _comparisons(rs, n_boot)
    return rows, comps, rs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static report card for a runs directory.")
    ap.add_argument("runs_dir", help="folder of run CSVs (step,return,seed[,eval_type])")
    ap.add_argument("--out", help="write a static HTML report card to this path")
    ap.add_argument("--eval-type", default=None, help="keep only rows with this eval_type (e.g. mppi)")
    ap.add_argument("--final-k", type=int, default=5, help="final = mean of last K evals (default 5)")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    args = ap.parse_args(argv)

    rows, comps, rs = build_report(args.runs_dir, eval_type=args.eval_type,
                                   final_k=args.final_k, n_boot=args.n_boot)
    if not rows:
        print(f"No runs found under {args.runs_dir!r} (expected *.csv with step,return,seed).",
              file=sys.stderr)
        return 1
    _print_report(rows, comps, rs, args.runs_dir)
    if args.out:
        html = _html_report(rows, comps, rs, args.runs_dir)
        with open(args.out, "w") as f:
            f.write(html)
        print(f"  wrote HTML report card → {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
