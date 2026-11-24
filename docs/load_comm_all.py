#!/usr/bin/env python3
"""Load imbalance and communication per query at the recall targets of all 15 setups of the 3 x 5 curve
grid (plot_curves_panels6.py --five): DEEP / TTI 10M / 100M and MSMARCO10M, on 4 / 6 / 8 nodes.

The 6 setups already in the paper (DEEP10M 4/6n, TTI10M 4/6n, DEEP100M 8n, TTI100M 8n) are copied from
final_load_comm.csv unchanged. For the other 9 the operating point of each target is the fastest
configuration with recall >= target among the points of scaling_curves.csv / msmarco_curves.csv (as in
the curves), and its load / communication are read from the same logs, with the formulas of
load_comm_tables.py (see its docstring). MSMARCO: d = 768, 9376 queries. When a configuration was run in
more than one job, the latest log that holds the metric is used.

Output: load_comm_all.csv (same columns as final_load_comm.csv)
Usage:
    module load python/3.12.2
    python3 load_comm_all.py
"""
import csv
import glob
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_comm_tables as lc  # noqa: E402  formulas and the logs of the 6 original setups
from plot_curves_panels6 import load as load_curves  # noqa: E402  merged curve points

SCR, EXP = lc.SCR, lc.EXP
V = f"{SCR}/vectors"
NQ = {"MSMARCO": 9376}
DIM = {"DEEP": 96, "TTI": 200, "MSMARCO": 768}
STD = [0.80, 0.90, 0.95, 0.99]
SYSTEMS = {"DEEP10M": lc.SYS_ORDER, "DEEP100M": lc.SYS_ORDER,
           "TTI10M": ["GP-ANN*", "Pyramid*", "CoTra", "DistributedANN*"],  # Harmony < 0.70
           "TTI100M": ["GP-ANN*", "Pyramid*", "CoTra"], "MSMARCO10M": ["GP-ANN*", "Pyramid*", "CoTra"]}
NEW = ["DEEP10M-8n", "DEEP100M-4n", "DEEP100M-6n", "TTI10M-8n", "TTI100M-4n", "TTI100M-6n",
       "MSMARCO10M-4n", "MSMARCO10M-6n", "MSMARCO10M-8n"]
GP_LOGS = {
    "DEEP10M-8n": f"{V}/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k8_*.log",
    "TTI10M-8n": f"{V}/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k8_*.log",
    "DEEP100M-4n": f"{V}/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M-k4_*.log",
    "DEEP100M-6n": f"{V}/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M-k6_*.log",
    "TTI100M-4n": f"{V}/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M-k4_*.log",
    "TTI100M-6n": f"{V}/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M-k6_*.log",
    "MSMARCO10M-4n": f"{V}/MSMARCOWebSearch/curve_parallel/logs/gp-curve-msmarco10M-k4_6995979_*.log",
    "MSMARCO10M-6n": f"{V}/MSMARCOWebSearch/curve_parallel/logs/gp-curve-msmarco10M-k6_6995980_*.log",
    "MSMARCO10M-8n": f"{V}/MSMARCOWebSearch/curve_parallel/logs/gp-curve-msmarco10M-k8_6995981_*.log",
}
DA_LOGS = {"DEEP10M-8n": "deep10m-w8", "TTI10M-8n": "tti10m-w8", "DEEP100M-4n": "deep100m-w4-*",
           "DEEP100M-6n": "deep100m-w6-*"}
COTRA = {"DEEP10M-8n": "scala_anns_deep10M_8p-[0-9]*.out", "TTI10M-8n": "scala_anns_tti10M_8p-[0-9]*.out",
         "DEEP100M-4n": "scala_anns_deep100M_4p-[0-9]*.out",
         "DEEP100M-6n": "scala_anns_6p-cotra6_deep100_k10-6975613.out",
         "TTI100M-4n": "scala_anns_tti100M_4p-[0-9]*.out", "TTI100M-6n": "scala_anns_tti100M_6p-[0-9]*.out",
         "MSMARCO10M-4n": "scala_anns_msmarco10M_4p-[0-9]*.out", "MSMARCO10M-6n": "scala_anns_msmarco10M_6p-[0-9]*.out",
         "MSMARCO10M-8n": "scala_anns_msmarco10M_8p-[0-9]*.out"}


def ds_of(setup):
    return setup.split("-")[0]


def family(setup):
    return re.match(r"[A-Z]+", setup).group(0)


def latest(pattern):
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


def gp(setup, config):
    np_, ef = re.search(r"nprobe=(\d+) ef=(\d+)", config).groups()
    d, nq = DIM[family(setup)], NQ.get(family(setup), lc.NQ)
    for f in latest(GP_LOGS[setup]):
        t = open(f, errors="replace").read()
        ranks = re.findall(r"remote_requests_received=(\d+) local_requests_resolved=\d+ searches_on_rank=(\d+)", t)
        if ranks and re.search(rf"nprobe={np_} ef={ef}\b", t):
            remote = sum(int(r) for r, _ in ranks)
            return lc.imbalance([s for _, s in ranks]), remote * ((4 * d + 20) + (8 * lc.K + 24)) / nq
    raise LookupError(f"GP {setup} {config}")


def pyramid(setup, config):
    kk, ef = re.search(r"K=(\d+) ef=(\d+)", config).groups()
    nq = NQ.get(family(setup), lc.NQ)
    tag = setup.lower().replace("-", "_")
    for f in latest(f"{SCR}/dhnsw/logs/sweep_v2_fixed/pyramid_v2_sweep_{tag}_*_prof.tsv"):  # corrected launch
        tasks, comm = [], 0.0
        for r in csv.DictReader(open(f), delimiter="\t"):
            if (r["K"], r["ef"]) != (kk, ef):
                continue
            line = r["prof_line"]
            if line.startswith("slave"):
                tasks.append(re.search(r"tasks=(\d+)", line).group(1))
            elif line.startswith("sender"):
                m = re.search(r"task_msgs=(\d+) task_B=(\d+)", line)
                comm += int(m.group(1)) * int(m.group(2))
            elif line.startswith("receiver"):
                m = re.search(r"result_msgs=(\d+) result_B=(\d+)", line)
                comm += int(m.group(1)) * int(m.group(2))
        if tasks:
            return lc.imbalance(tasks), comm / nq
    raise LookupError(f"Pyramid {setup} {config}")


def da(setup, config):
    lc.DA_LOGS[setup] = [f"{SCR}/dann/logs/scaling_queries/da-q-{DA_LOGS[setup]}_*.log"]
    return lc.da(setup, config)  # later log wins


def cotra(setup, config):
    ef = re.search(r"ef=(\d+)", config).group(1)
    nq = NQ.get(family(setup), lc.NQ)
    for f in latest(f"{EXP}/CoTra/script_slurm/logs/{COTRA[setup]}"):
        loads, comm = None, None
        for line in open(f, errors="replace"):
            if line.startswith(f"{ef}\tcore_load"):
                loads = re.findall(r"\tm\d+=(\d+)", line)
            m = re.match(rf"{ef}\tcomm_sum\twrites=\d+\treads=\d+\twrite_bytes=(\d+)\tread_bytes=(\d+)", line)
            if m:
                comm = (int(m.group(1)) + int(m.group(2))) / nq
        if loads and comm is not None:
            return lc.imbalance(loads), comm
    raise LookupError(f"CoTra {setup} {config}")


def harmony(setup, config):
    lc.HARMONY[setup] = f"harmony_{setup.lower().replace('-', '_')}_omp16"
    return lc.harmony(setup, config)


FN = {"GP-ANN*": gp, "Pyramid*": pyramid, "DistributedANN*": da, "CoTra": cotra, "Harmony": harmony}


def op_row(setup, s, t, pts):
    ok = [p for p in pts.get((setup, s), []) if p[0] >= t]
    if not ok:
        return dict(setup=setup, system=s, target=t, config="", recall10="", qps="", imbalance="", bytes_per_query="")
    rec, qps, config = max(ok, key=lambda p: p[1])
    imb, comm = FN[s](setup, config)
    return dict(setup=setup, system=s, target=t, config=config, recall10=rec, qps=qps,
                imbalance=round(imb, 3), bytes_per_query=round(comm, 1))


def main():
    # Pyramid*: every setup recomputed from the corrected sweeps (sweep_v2_fixed, 16 real cores): the
    # operating point of a target may change with the new QPS; tasks / bytes per configuration do not
    rows = [r for r in csv.DictReader(open(HERE / "final_load_comm.csv"))
            if not (r["setup"].startswith("TTI10M") and r["system"] == "Harmony")
            and float(r["target"]) >= (0.80 if r["setup"].startswith("TTI10M") else 0.0)]
    old_pyr = [(r["setup"], float(r["target"])) for r in rows if r["system"] == "Pyramid*"]
    rows = [r for r in rows if r["system"] != "Pyramid*"]
    pts = load_curves()
    for setup, t in old_pyr:
        rows.append(op_row(setup, "Pyramid*", t, pts))
    for setup in NEW:
        for s in SYSTEMS[ds_of(setup)]:
            for t in STD:
                ok = [p for p in pts.get((setup, s), []) if p[0] >= t]
                if not ok:
                    rows.append(dict(setup=setup, system=s, target=t, config="", recall10="", qps="",
                                     imbalance="", bytes_per_query=""))
                    continue
                rec, qps, config = max(ok, key=lambda p: p[1])
                imb, comm = FN[s](setup, config)
                rows.append(dict(setup=setup, system=s, target=t, config=config, recall10=rec, qps=qps,
                                 imbalance=round(imb, 3), bytes_per_query=round(comm, 1)))
    with open(HERE / "load_comm_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote load_comm_all.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
