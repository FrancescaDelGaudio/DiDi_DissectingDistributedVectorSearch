#!/usr/bin/env python3
"""Load-balancing and communication tables at the recall targets of final_targets.csv
(written by plot_final_curves.py): same operating points as the recall/QPS figures.

Load imbalance = L_max / L_mean over nodes (1.00 = perfectly balanced), per-node work:
  GP-ANN*          searches per rank (remote requests received + local requests resolved)
  Pyramid*         search tasks per shard (slave)
  Harmony          compared vectors per team (workers of a team split dimensions, same work)
  CoTra            core participations per node
  DistributedANN*  shard scoring TIME, max/mean per round averaged over rounds
                   (score_imbalance; the batched log has no per-rank key counts)

Communication = payload bytes per query:
  GP-ANN*          remote requests x (request (4d+20 B) + response (8k+24 B))
  Pyramid*         task messages x task size + result messages x result size (all MPI messages)
  DistributedANN*  measured remote request + response bytes ("network bytes/query (remote only)")
  CoTra            RDMA bytes written + read, summed over nodes ("comm_sum")
  Harmony          estimate: partial distances exchanged inside teams (4 B per compared vector)
                   + broadcast of queries, probe lists and heap tops to every worker
                   + per-team result heaps (12 B per neighbour)

Outputs: final_load_comm.csv and tables_load_comm.tex next to this file.
"""
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCR = "/gpfs/scratch/delgaudio"
EXP = "/gpfs/users/delgaudio/exp_eval"
NQ, K = 10000, 10
DIM = {"DEEP": 96, "TTI": 200}

GP_LOGS = {
    "DEEP10M-4n": f"{SCR}/vectors/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k4_*.log",
    "DEEP10M-6n": f"{SCR}/vectors/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k6_*.log",
    "TTI10M-4n": f"{SCR}/vectors/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k4_*.log",
    "TTI10M-6n": f"{SCR}/vectors/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k6_*.log",
    "DEEP100M-8n": f"{SCR}/vectors/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M_*.log",
    "TTI100M-8n": f"{SCR}/vectors/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M_*.log",
}
PYR_TAGS = {"DEEP10M-4n": "deep10m_4n", "DEEP10M-6n": "deep10m_6n", "TTI10M-4n": "tti10m_4n",
            "TTI10M-6n": "tti10m_6n", "DEEP100M-8n": "deep100m_8n", "TTI100M-8n": "tti100m_8n"}
DA_LOGS = {
    "DEEP10M-4n": [f"{SCR}/dann/logs/final_campaign/da-final-deep10m-w4_*.log"],
    "DEEP10M-6n": [f"{SCR}/dann/logs/final_campaign/da-final-deep10m-w6_*.log"],
    "TTI10M-4n": [f"{SCR}/dann/logs/final_campaign/da-final-tti10m-w4_*.log"],
    "TTI10M-6n": [f"{SCR}/dann/logs/final_campaign/da-final-tti10m-w6_*.log"],
    "DEEP100M-8n": [f"{SCR}/dann/logs/query_batch/deep100m_w8/dann-batch-d100m_6923907.log",
                    f"{SCR}/dann/logs/final_campaign/da-final-deep100m-*.log"],
}
COTRA = {"DEEP10M-4n": "scala_anns-cotra4_deep_k10_comm-6911247.out",
         "DEEP10M-6n": "scala_anns_6p-cotra6_deep_k10_comm-6911249.out",
         "TTI10M-4n": "scala_anns-cotra4_tti_k10_comm-6911248.out",
         "TTI10M-6n": "scala_anns_6p-cotra6_tti_k10_comm-6911250.out",
         "DEEP100M-8n": "scala_anns_8p-cotra8_deep_k10-6911862.out",
         "TTI100M-8n": "scala_anns_8p-cotra8_tti_k10-6911864.out"}
HARMONY = {"DEEP10M-4n": "harmony_deep10m_4n_omp16", "DEEP10M-6n": "harmony_deep10m_6n_omp16",
           "TTI10M-4n": "harmony_tti10m_4n_omp16", "TTI10M-6n": "harmony_tti10m_6n_omp16",
           "DEEP100M-8n": "harmony_deep100m_8n_omp16"}


def imbalance(loads):
    loads = [float(x) for x in loads]
    return max(loads) / (sum(loads) / len(loads))


def gp(setup, config):
    np_, ef = re.search(r"nprobe=(\d+) ef=(\d+)", config).groups()
    d = DIM["TTI" if setup.startswith("TTI") else "DEEP"]
    files = [(f, open(f, errors="replace").read()) for f in glob.glob(GP_LOGS[setup])]
    if setup == "DEEP100M-8n":  # breakdown jobs hold nprobe=1 ef=70 and nprobe=4 ef=400 (one block per run)
        for job in ("6961001", "6960834"):
            bd = f"{SCR}/vectors/DEEP/distbench_100M_ef_nprobe/logs/gp_breakdown_{job}.log"
            files += [(bd, blk) for blk in re.split(r"---- run ", open(bd, errors="replace").read())[1:]]
    for f, t in files:
        if not re.search(rf"nprobe={np_} ef={ef}\b", t):
            continue
        ranks = re.findall(r"remote_requests_received=(\d+) local_requests_resolved=\d+ searches_on_rank=(\d+)", t)
        remote = sum(int(r) for r, _ in ranks)
        return imbalance([s for _, s in ranks]), remote * ((4 * d + 20) + (8 * K + 24)) / NQ
    raise LookupError(f"GP {setup} {config}")


def pyramid(setup, config):
    kk, ef = re.search(r"K=(\d+) ef=(\d+)", config).groups()
    f = glob.glob(f"{SCR}/dhnsw/logs/sweep_v2/pyramid_v2_sweep_{PYR_TAGS[setup]}_*_prof.tsv")[0]
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
    return imbalance(tasks), comm / NQ


def da(setup, config):
    bw, h, l = re.search(r"BW=(\d+) H=(\d+) L=(\d+)", config).groups()
    found = None
    for pattern in DA_LOGS[setup]:
        for f in sorted(glob.glob(pattern)):
            blocks = re.split(r"\n=+ RUN ", open(f, errors="replace").read())
            for b in blocks:
                if not re.match(rf"K=10 BW={bw} L={l} H={h} THREADS=16\b", b):
                    continue
                imb = re.search(r"score_imbalance\(mean max/mean per round\)=([0-9.]+)", b)
                net = re.search(r"network bytes/query \(remote only\): requests=([0-9.]+) KiB\s+responses=([0-9.]+) KiB", b)
                if imb and net:
                    found = (float(imb.group(1)), (float(net.group(1)) + float(net.group(2))) * 1024)
    if not found:
        raise LookupError(f"DA {setup} {config}")
    return found


def cotra(setup, config):
    ef = re.search(r"ef=(\d+)", config).group(1)
    loads, comm = None, None
    for line in open(f"{EXP}/CoTra/script_slurm/logs/{COTRA[setup]}", errors="replace"):
        if line.startswith(f"{ef}\tcore_load"):
            loads = re.findall(r"\tm\d+=(\d+)", line)
        m = re.match(rf"{ef}\tcomm_sum\twrites=\d+\treads=\d+\twrite_bytes=(\d+)\tread_bytes=(\d+)", line)
        if m:
            comm = (int(m.group(1)) + int(m.group(2))) / NQ
    return imbalance(loads), comm


def harmony(setup, config):
    npb = int(re.search(r"nprobe=(\d+)", config).group(1))
    d = DIM["TTI" if setup.startswith("TTI") else "DEEP"]
    best = None
    for f in sorted(glob.glob(f"{EXP}/Harmony/slurm_script/log/{HARMONY[setup]}*.out"), key=os.path.getmtime):
        t = open(f, errors="replace").read()
        if re.search(rf"HARMONY_K=10 HARMONY_NPROBE={npb}\b", t) and "TeamQueryStats: team=" in t:
            best = t  # later file wins, as in collect_final_curves.py
    hdr = re.search(r"teamCount=(\d+) teamSize=(\d+)", best)
    teams, size = int(hdr.group(1)), int(hdr.group(2))
    work = [int(x) for x in re.findall(r"TeamQueryStats: team=\d+ .*?compare_vectors=(\d+)", best)]
    workers = teams * size
    team_b = 4 * sum(work)                                   # partial distances inside teams
    dist_b = workers * NQ * (4 * d + 8 * npb + 2 * 8 + 4)    # queries, probe lists, sizes, heap tops
    heap_b = teams * NQ * K * 12                             # result heaps
    return imbalance(work), (team_b + dist_b + heap_b) / NQ


FN = {"GP-ANN*": gp, "Pyramid*": pyramid, "DistributedANN*": da, "CoTra": cotra, "Harmony": harmony}
SETUP_ORDER = ["DEEP10M-4n", "DEEP10M-6n", "TTI10M-4n", "TTI10M-6n", "DEEP100M-8n", "TTI100M-8n"]
SYS_ORDER = ["GP-ANN*", "Pyramid*", "Harmony", "CoTra", "DistributedANN*"]


def fmt_bytes(b):
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024:.1f} KB"


def main():
    rows = []
    for r in csv.DictReader(open(HERE / "final_targets.csv")):
        t = float(r["target"])
        if not r["config"]:
            rows.append(dict(setup=r["setup"], system=r["system"], target=t, config="",
                             recall10="", qps="", imbalance="", bytes_per_query=""))
            continue
        imb, comm = FN[r["system"]](r["setup"], r["config"])
        rows.append(dict(setup=r["setup"], system=r["system"], target=t, config=r["config"],
                         recall10=r["recall10"], qps=r["qps"], imbalance=round(imb, 3),
                         bytes_per_query=round(comm, 1)))
    with open(HERE / "final_load_comm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cell = {(r["setup"], r["system"], r["target"]): r for r in rows}
    targets = defaultdict(set)
    for r in rows:
        targets[(r["setup"], r["system"])].add(r["target"])

    def table(metric, caption, label, fmt):
        # Blocks of rows sharing the same targets: standard targets first, then TTI100M,
        # then the low-recall systems on TTI10M.
        keys = [(st, sy) for st in SETUP_ORDER for sy in SYS_ORDER if targets.get((st, sy))]
        blocks = []
        for k in keys:
            ts = tuple(sorted(targets[k]))
            for b in blocks:
                if b[0] == ts:
                    b[1].append(k)
                    break
            else:
                blocks.append((ts, [k]))
        blocks.sort(key=lambda b: (-len(b[1]), -b[0][0]))
        out = [r"\begin{table*}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
               r"\begin{tabular}{llcccc}", r"\toprule"]
        for i, (ts, ks) in enumerate(blocks):
            if i:
                out.append(r"\midrule")
            hdr = " & ".join(f"$\\geq {t:.2f}$" for t in ts) + " & " * (4 - len(ts))
            out += [f"\\textbf{{Setup}} & \\textbf{{System}} & {hdr} \\\\", r"\midrule"]
            prev = None
            for setup, sysname in ks:
                vals = [fmt(cell[(setup, sysname, t)][metric]) if cell[(setup, sysname, t)][metric] != "" else "n/r"
                        for t in ts]
                vals += [""] * (4 - len(vals))
                name = sysname.replace("*", "$^*$")
                lab = setup.replace("-", " ") if setup != prev else ""
                if prev is not None and setup != prev:
                    out.append(r"\addlinespace")
                prev = setup
                out.append(f"{lab} & {name} & " + " & ".join(vals) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        return "\n".join(out)

    tex = table("imbalance",
                "Load imbalance $L_{\\max}/\\bar{L}$ at each recall@10 target (same operating points as the "
                "recall/QPS figures; 1.00 = perfectly balanced). Per-node work: searches per rank (GP-ANN$^*$), "
                "search tasks per shard (Pyramid$^*$), compared vectors per team (Harmony), core participations "
                "(CoTra), shard scoring time (DistributedANN$^*$). n/r: target not reached.",
                "tab:load-imbalance", lambda v: f"{float(v):.2f}")
    tex += "\n" + table("bytes_per_query",
                        "Communication volume per query at each recall@10 target (same operating points as the "
                        "recall/QPS figures). Harmony is estimated from the compared-vector counts. n/r: target not "
                        "reached.", "tab:comm-cost", lambda v: fmt_bytes(float(v)))
    (HERE / "tables_load_comm.tex").write_text(tex)

    # Summary of the main setup: QPS, load imbalance and communication side by side.
    setup, ts = "DEEP100M-8n", [0.80, 0.90, 0.95, 0.99]
    out = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{DEEP100M on 8 nodes: throughput, load imbalance and communication at each recall@10 "
           r"target (fastest configuration reaching the target; same operating points as the recall/QPS, "
           r"load and communication figures). Imb.\ = extra work on the busiest node, $(L_{\max}/\bar{L}-1)$; "
           r"KB/q = payload kilobytes per query. n/r: target not reached.}",
           r"\label{tab:summary-deep100m}",
           r"\begin{tabular}{l" + "rrr" * len(ts) + "}", r"\toprule",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{recall $\\geq {t:.2f}$}}" for t in ts) + r" \\",
           "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(ts))),
           r"\textbf{System}" + " & QPS & Imb. & KB/q" * len(ts) + r" \\", r"\midrule"]
    for sysname in SYS_ORDER:
        cells = []
        for t in ts:
            c = cell.get((setup, sysname, t))
            if not c or c["qps"] == "":
                cells.append(r"\multicolumn{3}{c}{n/r}")
                continue
            q = float(c["qps"])
            qs = f"{q / 1000:.1f}k" if q >= 1000 else f"{q:.0f}"
            imb = (float(c["imbalance"]) - 1) * 100
            kb = float(c["bytes_per_query"]) / 1024
            kbs = f"{kb:.1f}" if kb < 1000 else f"{kb / 1024:.1f}\\,MB"
            cells.append(f"{qs} & +{imb:.0f}\\% & {kbs}")
        out.append(sysname.replace("*", "$^*$") + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (HERE / "tables_summary_deep100m.tex").write_text("\n".join(out))
    print(f"wrote final_load_comm.csv ({len(rows)} rows) and tables_load_comm.tex")


if __name__ == "__main__":
    main()
