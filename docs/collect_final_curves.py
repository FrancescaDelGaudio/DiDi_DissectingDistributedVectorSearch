#!/usr/bin/env python3
"""Collect (recall@10, QPS) points of the final campaign, latest version of every system.
Setups: DEEP10M / TTI10M on 4 and 6 nodes, DEEP100M / TTI100M on 8 nodes.
Versions: GP-ANN* multi-threaded (parallel_timer, QA_THREADS=16), Pyramid* v2 (16 threads),
DistributedANN* batched (16 threads), CoTra and Harmony at 16 threads.
Output: final_curves.csv next to this file (one row per configuration).

QPS conventions:
  GP-ANN*        "Query QPS" printed by DistributedBench (10000 / max-rank query phase)
  Pyramid*       10000 / query-phase wall (receiver), "qps" column of the sweep summary
  DistributedANN 10000 / elapsed of the batched query phase ("SWEEP_SUMMARY ... qps=")
  CoTra          cluster QPS printed by the leader ("ef recall 0.00 qps")
  Harmony        nq / query_time ("query_time:" line)
"""
import csv
import glob
import os
import re
from pathlib import Path

SCR = "/gpfs/scratch/delgaudio"
EXP = "/gpfs/users/delgaudio/exp_eval"
SETUPS = {  # setup -> (dataset, scale, nodes)
    "DEEP10M-4n": ("DEEP", "10M", 4), "DEEP10M-6n": ("DEEP", "10M", 6),
    "TTI10M-4n": ("TTI", "10M", 4), "TTI10M-6n": ("TTI", "10M", 6),
    "DEEP100M-8n": ("DEEP", "100M", 8), "TTI100M-8n": ("TTI", "100M", 8),
}
rows = []


def add(system, setup, config, recall, qps, source):
    rows.append(dict(system=system, setup=setup, config=config, recall10=round(float(recall), 5),
                     qps=round(float(qps), 1), source=os.path.basename(source)))


# ---------------- GP-ANN* (multi-threaded curve, overlap 0.2) ----------------
gp_logs = {
    "DEEP10M-4n": f"{SCR}/vectors/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k4_*.log",
    "DEEP10M-6n": f"{SCR}/vectors/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k6_*.log",
    "TTI10M-4n": f"{SCR}/vectors/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k4_*.log",
    "TTI10M-6n": f"{SCR}/vectors/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k6_*.log",
    "DEEP100M-8n": f"{SCR}/vectors/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M_*.log",
    "TTI100M-8n": f"{SCR}/vectors/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M_*.log",
}
for setup, pattern in gp_logs.items():
    for f in glob.glob(pattern):
        t = open(f, errors="replace").read()
        cfg = re.search(r"nprobe=(\d+) ef=(\d+)", t)
        r = re.search(r"recall@10 \(ID overlap\) ([0-9.]+)", t)
        q = re.search(r"Query QPS ([0-9.]+)", t)
        if cfg and r and q:
            add("GP-ANN*", setup, f"o=0.2 nprobe={cfg.group(1)} ef={cfg.group(2)}", r.group(1), q.group(1), f)

# GP-ANN* DEEP100M: the breakdown jobs (same build, index and nodes) also measured two
# configurations not in the curve sweep: nprobe=1 ef=70 (fastest above recall 0.80, job 6961001)
# and nprobe=4 ef=400 (recall 0.992, beyond the sweep's maximum ef=250, job 6960834).
GP_BD = f"{SCR}/vectors/DEEP/distbench_100M_ef_nprobe/logs"
GP_EXTRA = [("DEEP100M-8n", f"{GP_BD}/gp_breakdown_6961001.log", ("1", "70")),
            ("DEEP100M-8n", f"{GP_BD}/gp_breakdown_6960834.log", ("4", "400"))]
for setup, f, (np_, ef) in GP_EXTRA:
    for block in re.split(r"---- run ", open(f, errors="replace").read()):
        if not block.startswith(f"nprobe={np_} ef={ef} "):
            continue
        r = re.search(r"recall@10 \(ID overlap\) ([0-9.]+)", block)
        q = re.search(r"Query QPS ([0-9.]+)", block)
        add("GP-ANN*", setup, f"o=0.2 nprobe={np_} ef={ef}", r.group(1), q.group(1), f)

# ---------------- Pyramid* v2 (16 threads) ----------------
pyr_tags = {"DEEP10M-4n": "deep10m_4n", "DEEP10M-6n": "deep10m_6n", "TTI10M-4n": "tti10m_4n",
            "TTI10M-6n": "tti10m_6n", "DEEP100M-8n": "deep100m_8n", "TTI100M-8n": "tti100m_8n"}
for setup, tag in pyr_tags.items():
    for f in glob.glob(f"{SCR}/dhnsw/logs/sweep_v2/pyramid_v2_sweep_{tag}_*[0-9].tsv"):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["rc"] == "0" and r["recall_at_10"] not in ("", "NA") and r["qps"] not in ("", "NA"):
                add("Pyramid*", setup, f"K={r['K']} ef={r['ef']}", r["recall_at_10"], r["qps"], f)

# ---------------- DistributedANN* (batched, 16 threads) ----------------
da_logs = {
    "DEEP10M-4n": [f"{SCR}/dann/logs/final_campaign/da-final-deep10m-w4_*.log"],
    "DEEP10M-6n": [f"{SCR}/dann/logs/final_campaign/da-final-deep10m-w6_*.log"],
    "TTI10M-4n": [f"{SCR}/dann/logs/final_campaign/da-final-tti10m-w4_*.log"],
    "TTI10M-6n": [f"{SCR}/dann/logs/final_campaign/da-final-tti10m-w6_*.log"],
    "DEEP100M-8n": [f"{SCR}/dann/logs/query_batch/deep100m_w8/dann-batch-d100m_6923907.log",
                    f"{SCR}/dann/logs/final_campaign/da-final-deep100m-*.log"],
}
for setup, patterns in da_logs.items():
    seen = {}
    for pattern in patterns:
        for f in glob.glob(pattern):
            for line in open(f, errors="replace"):
                m = re.search(r"SWEEP_SUMMARY k=10 BW=(\d+) H=(\d+) L=(\d+) .*recall@10=([0-9.]+) "
                              r"elapsed_s=[0-9.]+ qps=([0-9.]+) mode=batch W=\d+ threads=(\d+)", line)
                if m and m.group(6) == "16":
                    seen[(m.group(1), m.group(2), m.group(3))] = (m.group(4), m.group(5), f)  # later run wins
    for (bw, h, l), (rec, qps, f) in seen.items():
        add("DistributedANN*", setup, f"BW={bw} H={h} L={l}", rec, qps, f)

# ---------------- CoTra (16 threads) ----------------
cotra = {"DEEP10M-4n": "scala_anns-cotra4_deep_k10_comm-6911247.out",
         "DEEP10M-6n": "scala_anns_6p-cotra6_deep_k10_comm-6911249.out",
         "TTI10M-4n": "scala_anns-cotra4_tti_k10_comm-6911248.out",
         "TTI10M-6n": "scala_anns_6p-cotra6_tti_k10_comm-6911250.out",
         "DEEP100M-8n": "scala_anns_8p-cotra8_deep_k10-6911862.out",
         "TTI100M-8n": "scala_anns_8p-cotra8_tti_k10-6911864.out"}
for setup, name in cotra.items():
    path = f"{EXP}/CoTra/script_slurm/logs/{name}"
    seen = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+0\.00\s+([0-9.]+)\s*$", line)
        if m:
            seen[int(m.group(1))] = (m.group(2), m.group(3))
    for ef, (rec, qps) in sorted(seen.items()):
        add("CoTra", setup, f"ef={ef}", rec, qps, path)

# ---------------- Harmony (16 threads) ----------------
harmony = {"DEEP10M-4n": "harmony_deep10m_4n_omp16", "DEEP10M-6n": "harmony_deep10m_6n_omp16",
           "TTI10M-4n": "harmony_tti10m_4n_omp16", "TTI10M-6n": "harmony_tti10m_6n_omp16",
           "DEEP100M-8n": "harmony_deep100m_8n_omp16"}
for setup, prefix in harmony.items():
    latest = {}
    for f in sorted(glob.glob(f"{EXP}/Harmony/slurm_script/log/{prefix}*.out"), key=os.path.getmtime):
        t = open(f, errors="replace").read()
        h = re.search(r"HARMONY_K=(\d+) HARMONY_NPROBE=(\d+)", t)
        r = re.search(r"recall: ([0-9.]+)", t)
        qt = re.search(r" query_time: ([0-9.]+)", t)
        nq = re.search(r"nq:(\d+)", t)
        if h and h.group(1) == "10" and r and qt and float(qt.group(1)) > 0:
            n = int(nq.group(1)) if nq else 10000
            latest[int(h.group(2))] = (r.group(1), n / float(qt.group(1)), f)  # later file wins
    for npb, (rec, qps, f) in sorted(latest.items()):
        add("Harmony", setup, f"nprobe={npb}", rec, qps, f)

out = Path(__file__).resolve().parent / "final_curves.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["setup"], r["system"], r["recall10"])))
print(f"wrote {out} ({len(rows)} rows)")
