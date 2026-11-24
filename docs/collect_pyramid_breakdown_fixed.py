#!/usr/bin/env python3
"""Pyramid* breakdown and communication split with the method of the paper (plot_breakdown.py /
plot_subfigs_comm.py PYRAMID_A/B), recomputed on the corrected runs (rankfile slot ranges, 16 real cores):
job 7017380 (real runs x5, comm-only runs x3, transfer-only replay MPI vs RDMA), DEEP100M, 8 shards.
Per real run, % of the receiver wall time W (average shard):
  search  = search_util x window              routing = min(sender wait for routed queries, shard idle)
  idle    = window - search - thread-0 MPI    communication = 100 - routing - search (merge = 0)
  cap     = W - slowest shard's search time (protocol + transfer must stay below it)
Communication split: transfer = T_MPI (replay TOTAL), protocol = C_replay - T_MPI with C_replay = sender
time in Send in the comm-only runs (median of 3), waiting = communication - protocol - transfer;
RDMA gain on transfer = T_MPI - T_RDMA. Medians over the runs.
"""
import re
import statistics as st
from collections import defaultdict

D = "/gpfs/scratch/delgaudio/comm_replay/7017380"
OPS = [("0.80", 3, 50), ("0.90", 10, 100), ("0.95", 20, 200), ("0.99", 100, 800)]
KV = re.compile(r"(\w+)=([\d.eE+-]+)")


def runs(path):
    out = defaultdict(lambda: {"slaves": []})
    for line in open(path).readlines()[1:]:
        k, ef, rep, prof = line.rstrip("\n").split("\t", 3)
        kv = {a: float(b) for a, b in KV.findall(prof)}
        r = out[(int(k), int(ef), int(rep))]
        if prof.startswith("sender"):
            r["sender"] = kv
        elif prof.startswith("receiver"):
            r["receiver"] = kv
        elif prof.startswith("slave"):
            r["slaves"].append(kv)
    return out


def main():
    real, comm = runs(f"{D}/real_prof.tsv"), runs(f"{D}/comm_prof.tsv")
    tr = {}
    for line in open(f"{D}/replay_summary.tsv").readlines()[1:]:
        f = line.split("\t")
        m = re.search(r"pyramid_K(\d+)_ef(\d+)\.trace", f[0])
        if m and f[1] == "TOTAL":
            tr[(int(m.group(1)), int(m.group(2)))] = (float(f[2]), float(f[4]))
    print("target K:ef  W_ms  QPS  | routing search comm | transfer protocol waiting rdma_gain | cap  (all % of W)")
    for t, k, e in OPS:
        vals = defaultdict(list)
        for (kk, ee, _), r in real.items():
            if (kk, ee) != (k, e) or "receiver" not in r:
                continue
            W = r["receiver"]["wall_s"]
            sl = r["slaves"]
            search = st.mean(x["search_util"] * x["window_s"] for x in sl)
            idle = st.mean(x["window_s"] * (1 - x["search_util"]) - x["recv_s"] - x["send_s"] for x in sl)
            routing = min(r["sender"]["send_wait_s"], max(idle, 0.0))
            slowest = max(x["search_util"] * x["window_s"] for x in sl)
            vals["W"].append(W)
            vals["routing"].append(100 * routing / W)
            vals["search"].append(100 * search / W)
            vals["cap"].append(100 * (W - slowest) / W)
        W = st.median(vals["W"])
        routing, search, cap = (st.median(vals[x]) for x in ("routing", "search", "cap"))
        comm_share = 100 - routing - search
        c_replay = st.median(r["sender"]["send_s"] for (kk, ee, _), r in comm.items() if (kk, ee) == (k, e))
        t_mpi, t_rdma = tr[(k, e)]
        transfer = 100 * t_mpi / W
        protocol = 100 * (c_replay - t_mpi) / W
        waiting = comm_share - protocol - transfer
        gain = 100 * (t_mpi - t_rdma) / W
        print(f"{t}  {k:3d}:{e:<4d} {1e3*W:6.1f} {1e4/W:6.0f} | {routing:6.2f} {search:6.2f} {comm_share:6.2f} | "
              f"{transfer:6.3f} {protocol:6.2f} {waiting:6.2f} {gain:7.4f} | {cap:5.1f}   C_replay {1e3*c_replay:.1f} ms T_MPI {1e3*t_mpi:.2f} T_RDMA {1e3*t_rdma:.2f}")


if __name__ == "__main__":
    main()
