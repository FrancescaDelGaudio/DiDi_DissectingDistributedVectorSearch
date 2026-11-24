#!/usr/bin/env python3
"""MSMARCO query sweeps (submit_queries_msmarco.sh, 2026-11-03) -> msmarco_curves.csv.

Systems: GP-ANN*, Pyramid*, CoTra on MSMARCO10M (4/6/8 nodes), CoTra on MSMARCO100M (4/6/8 nodes).
DistributedANN* and Harmony are not tested on MSMARCO (cancelled; partial logs are ignored).
CoTra: every sweep aborts at its highest ef (500-700) after the points written here; the "-hi" jobs
(ef 1000-2000) produced no point.
Usage: python3 collect_msmarco_curves.py
"""
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCR = "/gpfs/scratch/delgaudio"
EXP = "/gpfs/users/delgaudio/exp_eval"
GP_LOG = f"{SCR}/vectors/MSMARCOWebSearch/curve_parallel/logs"
PY_LOG = f"{SCR}/dhnsw/logs/sweep_v2"
CO_LOG = f"{EXP}/CoTra/script_slurm/logs"
GP_JOBS = {4: "6995979", 6: "6995980", 8: "6995981"}

pts = defaultdict(dict)  # (setup, system) -> config -> (recall, qps)
for n in (4, 6, 8):
    s10, s100 = f"MSMARCO10M-{n}n", f"MSMARCO100M-{n}n"
    for f in glob.glob(f"{GP_LOG}/gp-curve-msmarco10M-k{n}_{GP_JOBS[n]}_*.log"):
        t = open(f, errors="replace").read()
        c = re.search(r"num_probes=(\d+) ef_search=(\d+)", t)
        r = re.search(r"recall@10 \(ID overlap\) ([0-9.]+)", t)
        q = re.search(r"Query QPS ([0-9.]+)", t)
        if c and r and q:
            pts[(s10, "GP-ANN*")][f"nprobe={c[1]} ef={c[2]}"] = (float(r[1]), float(q[1]))
    for f in glob.glob(f"{PY_LOG}/pyramid_v2_sweep_msmarco10m_{n}n_*[0-9].tsv"):
        # A "#[prof ]" line printed between "time || recall" and the result makes the sweep script
        # write "]" as recall (3 rows of 6n): recover it from the log line just before the same qps.
        log = open(f.replace("pyramid_v2_sweep_", "pyr-v2-").replace(".tsv", ".log"), errors="replace").read()
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["rc"] != "0" or r["qps"] in ("", "NA"):
                continue
            rec = r["recall_at_10"]
            if not re.fullmatch(r"[0-9.]+", rec):
                m = re.search(r"\n[0-9.]+ ([0-9.]+)\n#\[timer\] [^\n]* qps=" + re.escape(r["qps"]) + " ", log)
                if not m:
                    continue
                rec = m[1]
            pts[(s10, "Pyramid*")][f"K={r['K']} ef={r['ef']}"] = (float(rec), float(r["qps"]))
    for setup, mil in ((s10, "10M"), (s100, "100M")):
        for f in glob.glob(f"{CO_LOG}/scala_anns_msmarco{mil}_{n}p-[0-9]*.out"):
            for line in open(f, errors="replace"):
                m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+0\.00\s+([0-9.]+)\s*$", line)
                if m:
                    pts[(setup, "CoTra")][f"ef={m[1]}"] = (float(m[2]), float(m[3]))

with open(HERE / "msmarco_curves.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["setup", "system", "config", "recall10", "qps"])
    for (s, sy), d in sorted(pts.items()):
        for c, (r, q) in sorted(d.items(), key=lambda x: x[1][0]):
            w.writerow([s, sy, c, r, round(q, 1)])
for (s, sy), d in sorted(pts.items()):
    v = list(d.values())
    print(f"{s:16} {sy:9} pts={len(v):3} recall {min(x[0] for x in v):.3f}-{max(x[0] for x in v):.3f}")
