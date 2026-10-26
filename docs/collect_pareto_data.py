#!/usr/bin/env python3
"""Collect (recall@10, QPS) points per system and setup from the experiment logs.

Setups: DEEP10M / TTI10M on 4 and 6 nodes, original code of every system
(Pyramid 1 thread, DistributedANN v1, GP-ANN original, CoTra 16 threads, Harmony 16 threads).
Output: pareto_data.csv next to this file (one row per configuration).

QPS conventions:
  Pyramid        10000 / query-phase wall (receiver)                 ("qps=" in the log)
  DistributedANN 10000 / elapsed (one query at a time, v1)           ("QPS :" line)
  GP-ANN         10000 / max-rank query phase                         (tables_*_recall_time.md)
  CoTra          cluster QPS printed by the leader ("ef recall 0.00 qps"); follower lines are per machine
  Harmony        nq / query_time                                      ("query_time:" line)
"""
import csv
import glob
import os
import re
from pathlib import Path

SCR = "/gpfs/scratch/delgaudio"
EXP = "/gpfs/users/delgaudio/exp_eval"
rows = []


def add(system, dataset, nodes, config, recall, qps, source):
    rows.append(dict(system=system, setup=f"{dataset}10M-{nodes}n", dataset=dataset, nodes=nodes,
                     config=config, recall10=round(float(recall), 5), qps=round(float(qps), 1),
                     source=os.path.basename(source)))


# ---------------- Pyramid (1 thread, original code) ----------------
for nodes in (4, 6):
    for f in glob.glob(f"{SCR}/dhnsw/logs/dhnsw_{nodes}n_retimed_*.log"):
        t = open(f, errors="replace").read()
        m = re.search(r"\((dhnsw-\dn-(deep|tti)-K(\d+)-ef(\d+)-k(\d+))\)", t)
        if not m or m.group(5) != "10":
            continue
        q = re.search(r"qps=([0-9.eE+-]+)", t)
        r = re.search(r"time \|\| recall\s*\n\s*[0-9.eE+-]+\s+([0-9.]+)", t)
        if q and r:
            add("Pyramid", m.group(2).upper(), nodes, f"K={m.group(3)},ef={m.group(4)}", r.group(1), q.group(1), f)

# ---------------- DistributedANN v1 (one query at a time) ----------------
dann_dirs = {
    ("DEEP", 4): (f"{SCR}/dann/logs/query_bw_h_sweep", "index_deep10m_k32"),
    ("DEEP", 6): (f"{SCR}/dann/logs/query_bw_h_sweep/deep_w6", None),
    ("TTI", 4): (f"{SCR}/dann/logs/query_bw_h_sweep/tti_ipfix", None),
    ("TTI", 6): (f"{SCR}/dann/logs/query_bw_h_sweep/tti_w6", None),
}
for (ds, nodes), (d, index_must) in dann_dirs.items():
    for f in glob.glob(f"{d}/*.log"):
        t = open(f, errors="replace").read()
        if f"(W={nodes})" not in t and f"W={nodes}" not in t:
            continue
        if index_must and not re.search(rf"{index_must}\b(?!_)", t):
            continue
        m = re.search(r"SWEEP_SUMMARY k=10 BW=(\d+) H=(\d+) L=(\d+) .*?recall@10=([0-9.]+).*?qps=([0-9.]+)", t)
        if m:
            add("DistributedANN", ds, nodes, f"BW={m.group(1)},H={m.group(2)},L={m.group(3)}", m.group(4), m.group(5), f)

# ---------------- GP-ANN (tables built from arrays 6869739/40/42/43) ----------------
gp_tables = {("DEEP", 4): "tables_k4_recall_time.md", ("DEEP", 6): "tables_k6_recall_time.md",
             ("TTI", 4): "tables_tti_k4_recall_time.md", ("TTI", 6): "tables_tti_k6_recall_time.md"}
for (ds, nodes), name in gp_tables.items():
    path = f"{EXP}/gp-ann/{name}"
    section = overlap = None
    for line in open(path):
        if line.startswith("## "):
            section = line.strip()
        elif line.startswith("### overlap"):
            overlap = line.split("=")[1].strip()
        elif section == "## recall@10" and re.match(r"\|\s*\d", line):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            add("GP-ANN", ds, nodes, f"o={overlap},np={c[0]},ef={c[1]}", c[2], c[4], path)

# ---------------- CoTra (16 threads, original code) ----------------
cotra = {("DEEP", 4): "scala_anns-cotra4_deep_k10_comm-6911247.out", ("DEEP", 6): "scala_anns_6p-cotra6_deep_k10_comm-6911249.out",
         ("TTI", 4): "scala_anns-cotra4_tti_k10_comm-6911248.out", ("TTI", 6): "scala_anns_6p-cotra6_tti_k10_comm-6911250.out"}
for (ds, nodes), name in cotra.items():
    path = f"{EXP}/CoTra/script_slurm/logs/{name}"
    seen = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+0\.00\s+([0-9.]+)\s*$", line)
        if m:
            seen[int(m.group(1))] = (m.group(2), m.group(3))
    for ef, (rec, qps) in sorted(seen.items()):
        add("CoTra", ds, nodes, f"ef={ef}", rec, qps, path)

# ---------------- Harmony (16 threads, original code) ----------------
for ds in ("DEEP", "TTI"):
    for nodes in (4, 6):
        latest = {}
        for f in sorted(glob.glob(f"{EXP}/Harmony/slurm_script/log/harmony_{ds.lower()}10m_{nodes}n_omp16*.out"),
                        key=os.path.getmtime):
            t = open(f, errors="replace").read()
            h = re.search(r"HARMONY_K=(\d+) HARMONY_NPROBE=(\d+)", t)
            r = re.search(r"recall: ([0-9.]+)", t)
            qt = re.search(r" query_time: ([0-9.]+)", t)
            nq = re.search(r"nq:(\d+)", t)
            if h and h.group(1) == "10" and r and qt and float(qt.group(1)) > 0:
                n = int(nq.group(1)) if nq else 10000
                latest[int(h.group(2))] = (r.group(1), n / float(qt.group(1)), f)   # later file wins
        for npb, (rec, qps, f) in sorted(latest.items()):
            add("Harmony", ds, nodes, f"nprobe={npb}", rec, qps, f)

out = Path(__file__).resolve().parent / "pareto_data.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["setup"], r["system"], r["recall10"])))
print(f"wrote {out} ({len(rows)} rows)")
