#!/usr/bin/env python3
"""Communication split (waiting / protocol / transfer) of GP-ANN* and Pyramid* over MPI and over native
RDMA, single pass, with the method of the comm zoom, from the A/B jobs (same runs for both variants):
  transfer = T_x (transfer-only replay with transport x), protocol = C_replay,x - T_x,
  waiting  = C_real,x - C_replay,x;  all in ms, and in % of the MPI query time T_MPI.
GP-ANN*:  C_real = mean over ranks of (total - route - post - search - merge), median of the 10
          first-batch real runs (jobs 6987766 + 6994617); C_replay = last router, batch 1 of the
          comm-only runs of job 6996535 (median of 3); T_x = transport_replay job 6975717.
Pyramid*: C_real = W - routing - search, with W the receiver wall time, routing = time the sender waits
          for routed queries, search = mean shard search utilisation x window; median of 6 real runs of
          job 7001311; C_replay = sender time in Send in its comm-only runs (median of 3); T_x =
          transport_replay job 6977296 (tasks + results).
Writes rdma_split.csv.
"""
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KV = re.compile(r"(\w+)=([\d.eE+-]+)")
GP_LOGS = Path("/gpfs/scratch/delgaudio/vectors/DEEP/distbench_100M_ef_nprobe/logs")
GP_CFG = {(1, 70): "0.80", (3, 50): "0.90", (2, 150): "0.95", (4, 400): "0.99"}
GP_T = {"0.80": {"MPI": 0.093, "RDMA": 0.078}, "0.90": {"MPI": 0.272, "RDMA": 0.253},     # ms, job 6975717
        "0.95": {"MPI": 0.206, "RDMA": 0.183}, "0.99": {"MPI": 0.324, "RDMA": 0.300}}
PY_LOG = Path("/gpfs/scratch/delgaudio/comm_replay/logs/pyr_ab_7001311.log")
PY_CFG = {(3, 50): "0.80", (10, 100): "0.90", (20, 200): "0.95", (100, 800): "0.99"}
PY_T = {"0.80": {"MPI": 6.099, "RDMA": 5.530}, "0.90": {"MPI": 8.272, "RDMA": 7.588},     # ms, job 6977296
        "0.95": {"MPI": 9.890, "RDMA": 9.134}, "0.99": {"MPI": 16.143, "RDMA": 14.903}}


def gp_run(ranks):
    comp = [r["route_wall_s"] + r["post_req_s"] + r["search_wall_s"] + r["merge_s"] for r in ranks]
    return max(r["total_s"] for r in ranks), st.mean(r["total_s"] - c for r, c in zip(ranks, comp))


def gp():
    real, replay = defaultdict(list), defaultdict(list)
    prof = defaultdict(list)
    for line in open(GP_LOGS / "gp_ab_6987766_prof.tsv").readlines()[1:]:
        mode, v, n, e, rep, rest = line.rstrip("\n").split("\t", 5)
        if mode == "real":
            prof[(v, int(n), int(e), int(rep))].append({k: float(x) for k, x in KV.findall(rest)})
    for (v, n, e, _), rs in prof.items():
        real[(GP_CFG[(n, e)], v.upper())].append(gp_run(rs))
    for logname, kind in (("gp_abb_6994617.log", "real"), ("gp_abb_6996535.log", "replay")):
        cur, prof = None, defaultdict(list)
        for line in open(GP_LOGS / logname, errors="replace"):
            m = re.search(r"---- (\w+) nprobe=(\d+) ef=(\d+) rep=(\d+) ----", line)
            if m:
                cur = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            elif cur and line.startswith("#[prof ]"):
                prof[cur].append({k: float(x) for k, x in KV.findall(line)})
        for (v, n, e, _), rs in prof.items():
            if len(rs) < 8:
                continue
            b1 = rs[:8]
            key = (GP_CFG[(n, e)], v.upper())
            if kind == "real":
                real[key].append(gp_run(b1))
            else:
                last = max(b1, key=lambda r: r["route_wall_s"] + r["post_req_s"])
                replay[key].append(last["total_s"] - last["route_wall_s"] - last["post_req_s"]
                                   - last["search_wall_s"] - last["merge_s"])
    out = {}
    for (t, v), runs in real.items():
        out[(t, v)] = dict(T=1e3 * st.median(x[0] for x in runs), C_real=1e3 * st.median(x[1] for x in runs),
                           C_replay=1e3 * st.median(replay[(t, v)]), T_tr=GP_T[t][v], n_real=len(runs),
                           n_replay=len(replay[(t, v)]))
    return out


def pyramid():
    cur, runs = None, defaultdict(lambda: {"slaves": []})
    for line in open(PY_LOG, errors="replace"):
        m = re.search(r"---- (\w+) mode=(\w+) K=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            cur = (m.group(1).upper(), m.group(2), PY_CFG[(int(m.group(3)), int(m.group(4)))], int(m.group(5)))
            continue
        if not cur or not line.startswith("#[prof ]"):
            continue
        kv = {k: float(x) for k, x in KV.findall(line)}
        body = line[len("#[prof ] "):]
        if body.startswith("sender"):
            runs[cur]["sender"] = kv
        elif body.startswith("receiver"):
            runs[cur]["receiver"] = kv
        elif body.startswith("slave"):
            runs[cur]["slaves"].append(kv)
    real, replay = defaultdict(list), defaultdict(list)
    for (v, mode, t, _), r in runs.items():
        if mode == "comm":
            replay[(t, v)].append(1e3 * r["sender"]["send_s"])
            continue
        W = r["receiver"]["wall_s"]
        routing = r["sender"]["send_wait_s"]
        search = st.mean(x["search_util"] * x["window_s"] for x in r["slaves"])
        real[(t, v)].append((1e3 * W, 1e3 * (W - routing - search)))
    out = {}
    for (t, v), rr in real.items():
        out[(t, v)] = dict(T=st.median(x[0] for x in rr), C_real=st.median(x[1] for x in rr),
                           C_replay=st.median(replay[(t, v)]), T_tr=PY_T[t][v], n_real=len(rr),
                           n_replay=len(replay[(t, v)]))
    return out


def main():
    rows = []
    for system, data in (("GP-ANN*", gp()), ("Pyramid*", pyramid())):
        for t in ("0.80", "0.90", "0.95", "0.99"):
            t_mpi = data[(t, "MPI")]["T"]
            for v in ("MPI", "RDMA"):
                x = data[(t, v)]
                comp = dict(transfer=x["T_tr"], protocol=x["C_replay"] - x["T_tr"], waiting=x["C_real"] - x["C_replay"])
                rows.append(dict(system=system, target=t, variant=v, T_ms=round(x["T"], 3), C_real_ms=round(x["C_real"], 3),
                                 C_replay_ms=round(x["C_replay"], 3), **{f"{k}_ms": round(c, 3) for k, c in comp.items()},
                                 **{f"{k}_pct_of_T_MPI": round(100 * c / t_mpi, 3) for k, c in comp.items()},
                                 n_real=x["n_real"], n_replay=x["n_replay"]))
    with open(HERE / "rdma_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['system']:9s} {r['target']} {r['variant']:4s} T={r['T_ms']:8.1f} C_real={r['C_real_ms']:7.1f} "
              f"C_replay={r['C_replay_ms']:6.1f} | %T_MPI: wait {r['waiting_pct_of_T_MPI']:6.2f} prot {r['protocol_pct_of_T_MPI']:6.2f} "
              f"transf {r['transfer_pct_of_T_MPI']:5.2f}  (n {r['n_real']}/{r['n_replay']})")


if __name__ == "__main__":
    main()
