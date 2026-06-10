"""Live learning-curve dashboard for a local runs directory (Flask app factory).

Serves a single HTML page with three panels, all re-reading the runs folder on
each request (so it stays live as runs append to their CSVs):

  - learning curves: per-seed faint lines + per-group IQM line with a 95%
    bootstrap CI band;
  - peak-vs-final table: per-group IQM[CI] for peak and final, with n<5 and
    peak>>final flags;
  - A/B paired-comparison panel: paired-difference bootstrap between two groups.

Generalised from the tdmpc-glass dashboard but fully decoupled — there is no
box/SSH/queue machinery, no fleet registry, no remote mirror. It just points at
``RUNS_DIR``.

JSON API:
  GET /api/curves   -> per-group {seeds:[...], iqm curve + ci band}
  GET /api/summary  -> per-group peak/final IQM[CI] + flags
  GET /api/compare?a=<g>&b=<g> -> paired-diff bootstrap on peak & final
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from ..curves import load_runs
from ..report import MIN_SEEDS, PEAK_FINAL_GAP_FRAC, _group_rows
from ..stats import iqm

DEFAULT_PORT = 5055


def _clean(v):
    """JSON-safe scalar (NaN/inf -> None)."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _curves_payload(runs_dir: str, eval_type: str | None):
    rs = load_runs(runs_dir, eval_type=eval_type)
    out = []
    for name in rs.group_names():
        g = rs.groups[name]
        seeds = [{
            "seed": c.seed,
            "steps": [int(s) for s in c.steps.tolist()],
            "returns": [_clean(r) for r in c.returns.tolist()],
            "peak": _clean(c.peak),
            "final": _clean(c.final(rs.final_k)),
        } for c in g.seeds]
        out.append({
            "group": name,
            "n_seeds": g.n_seeds,
            "seeds": seeds,
            "grid": [int(s) for s in g.grid.tolist()],
            "iqm": [_clean(v) for v in g.iqm_curve.tolist()],
            "ci_lo": [_clean(v) for v in g.ci_lo.tolist()],
            "ci_hi": [_clean(v) for v in g.ci_hi.tolist()],
        })
    return out


def create_app(runs_dir: str, eval_type: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["RUNS_DIR"] = str(runs_dir)
    app.config["EVAL_TYPE"] = eval_type

    @app.route("/")
    def index():
        return render_template("index.html", runs_dir=app.config["RUNS_DIR"])

    @app.route("/api/curves")
    def api_curves():
        et = request.args.get("eval_type", app.config["EVAL_TYPE"]) or None
        return jsonify({"groups": _curves_payload(app.config["RUNS_DIR"], et),
                        "runs_dir": app.config["RUNS_DIR"]})

    @app.route("/api/summary")
    def api_summary():
        et = request.args.get("eval_type", app.config["EVAL_TYPE"]) or None
        rs = load_runs(app.config["RUNS_DIR"], eval_type=et)
        rows = _group_rows(rs, n_boot=2000)
        for r in rows:
            for k, v in list(r.items()):
                r[k] = _clean(v)
        return jsonify({"rows": rows, "min_seeds": MIN_SEEDS,
                        "gap_frac": PEAK_FINAL_GAP_FRAC, "final_k": rs.final_k})

    @app.route("/api/compare")
    def api_compare():
        a, b = request.args.get("a"), request.args.get("b")
        et = request.args.get("eval_type", app.config["EVAL_TYPE"]) or None
        rs = load_runs(app.config["RUNS_DIR"], eval_type=et)
        if a not in rs.groups or b not in rs.groups:
            return jsonify({"error": "unknown group(s)",
                            "groups": rs.group_names()}), 400
        c = rs.compare(a, b, n=5000)
        for metric in ("peak", "final"):
            c[metric] = {k: _clean(v) for k, v in c[metric].items()}
        return jsonify(c)

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Live learning-curve dashboard for a runs directory.")
    ap.add_argument("runs_dir", help="folder of run CSVs (step,return,seed[,eval_type])")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--eval-type", default=None, help="keep only rows with this eval_type")
    args = ap.parse_args(argv)
    if not Path(args.runs_dir).exists():
        print(f"runs_dir {args.runs_dir!r} does not exist")
        return 1
    app = create_app(args.runs_dir, eval_type=args.eval_type)
    print(f"honest-rl-bench dashboard → http://localhost:{args.port}  (runs: {args.runs_dir})")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
