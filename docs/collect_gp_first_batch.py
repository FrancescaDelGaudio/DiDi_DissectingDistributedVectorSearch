#!/usr/bin/env python3
"""First batch (single pass, as the throughput curves) of GP-ANN* over MPI vs native RDMA, from the two
A/B jobs that ran MPI and RDMA in separate processes on the same saved shard graphs (DEEP100M, 8 nodes):
  job 6987766: one batch per mpirun, 5 runs per variant and target (logs/gp_ab_6987766_prof.tsv)
  job 6994617: batch 1 of 11 per mpirun, 5 runs per variant and target (logs/gp_abb_6994617.log)
Per run: T = batch time (max over ranks), comm = T - busiest rank's route + post + search + merge.
Writes gp_first_batch.csv (target, recall, variant, job, rep, T_s, comm_s).
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

LOGS = Path("/gpfs/scratch/delgaudio/vectors/DEEP/distbench_100M_ef_nprobe/logs")
TARGET = {(1, 70): ("0.80", 0.8203), (3, 50): ("0.90", 0.9023), (2, 150): ("0.95", 0.9487), (4, 400): ("0.99", 0.9913)}
KV = re.compile(r"(\w+)=([\d.eE+-]+)")


def batch_values(ranks):
    T = max(r["total_s"] for r in ranks)
    comp = max(r["route_wall_s"] + r["post_req_s"] + r["search_wall_s"] + r["merge_s"] for r in ranks)
    return T, T - comp


def main():
    rows = []
    prof = defaultdict(list)
    for line in open(LOGS / "gp_ab_6987766_prof.tsv").readlines()[1:]:
        mode, v, n, e, rep, rest = line.rstrip("\n").split("\t", 5)
        if mode == "real":
            prof[(v, int(n), int(e), int(rep))].append({k: float(x) for k, x in KV.findall(rest)})
    for (v, n, e, rep), rs in prof.items():
        T, c = batch_values(rs)
        rows.append((*TARGET[(n, e)], v, 6987766, rep, T, c))
    cur, prof = None, defaultdict(list)
    for line in open(LOGS / "gp_abb_6994617.log", errors="replace"):
        m = re.search(r"---- (\w+) nprobe=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            cur = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        elif cur and line.startswith("#[prof ]"):
            prof[cur].append({k: float(x) for k, x in KV.findall(line)})
    for (v, n, e, rep), rs in prof.items():
        T, c = batch_values(rs[:8])          # batch 1 = first 8 prof lines of the mpirun
        rows.append((*TARGET[(n, e)], v, 6994617, rep, T, c))
    with open(Path(__file__).resolve().parent / "gp_first_batch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "recall", "variant", "job", "rep", "T_s", "comm_s"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} runs")


if __name__ == "__main__":
    main()
