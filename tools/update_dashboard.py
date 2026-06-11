#!/usr/bin/env python3
"""Regenerate docs/assets/dashboard.json.

Merges:
  - live GPU probes (ssh + nvidia-smi) of the boxes listed in BOXES
  - the manually maintained run/queue registry in tools/runs.json

Usage:  python3 tools/update_dashboard.py        # probe + write
        python3 tools/update_dashboard.py --no-probe   # registry only

This only does ssh I/O (no computation); safe to run from the center box.
"""
import json, subprocess, sys, datetime, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SSH_KEY = os.path.expanduser("~/.ssh/vastai_id_ed25519")

# instance_id -> (host, port, gpu, cores, project, note)
BOXES = {
    "40230626": ("ssh5.vast.ai", 30627, "RTX A4000",        None, "struct-mamba",
                 "PEMS04/07/08 grid"),
    "40424707": ("ssh1.vast.ai", 24707, "(probe)",          None, "struct-mamba",
                 "PEMS03 grid done 2026-06-11"),
    "40230497": ("ssh8.vast.ai", 30497, "RTX 3060 12GB",    12,   "ts-bench-audit",
                 "handed over by mahjong 2026-06-11"),
    "40121712": ("ssh2.vast.ai", 11713, "RTX 5060",         28,   "mahjong-eval + ts-bench-audit",
                 "mahjong keeps CPUs until 06-14; GPU co-tenant"),
    "22734":    ("ssh5.vast.ai", 22734, "RTX 3070 Laptop",  16,   "ts-bench-audit",
                 "KNOWN FLAKY - short runs only"),
}


def probe(host, port):
    cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
           "-i", SSH_KEY, "-p", str(port), f"root@{host}",
           "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
           "--format=csv,noheader 2>/dev/null"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout.strip()
        line = [l for l in out.splitlines() if "," in l]
        if not line:
            return None
        name, util, used, total = [x.strip() for x in line[-1].split(",")]
        return {"gpu": name, "util": util, "mem": f"{used} / {total}"}
    except Exception:
        return None


def main():
    do_probe = "--no-probe" not in sys.argv
    registry = json.load(open(os.path.join(HERE, "runs.json")))

    fleet = []
    for iid, (host, port, gpu, cores, project, note) in BOXES.items():
        entry = {"instance": iid, "endpoint": f"{host}:{port}", "gpu": gpu,
                 "cores": cores, "project": project, "note": note,
                 "util": "?", "mem": "?", "reachable": None}
        if do_probe:
            p = probe(host, port)
            if p:
                entry.update({"gpu": p["gpu"], "util": p["util"],
                              "mem": p["mem"], "reachable": True})
            else:
                entry["reachable"] = False
        fleet.append(entry)

    dash = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "fleet": fleet,
        "runs": registry["runs"],
        "queue": registry["queue"],
    }
    out_path = os.path.join(REPO, "docs", "assets", "dashboard.json")
    json.dump(dash, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    for b in fleet:
        print(f"  {b['instance']:>9} {b['endpoint']:<22} {b['gpu']:<22} "
              f"util={b['util']:<5} reachable={b['reachable']}")


if __name__ == "__main__":
    main()
