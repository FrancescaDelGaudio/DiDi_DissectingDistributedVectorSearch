#!/usr/bin/env python3
"""GP-ANN* over MPI vs native RDMA: communication in the communication-only replay, FIRST BATCH
(single pass, as the throughput curves and the comm zoom). Same quantity as C_replay of the zoom:
communication of the rank that finishes routing last (total - route - post - search - merge).
Data: job 6996535 (sbatch_ab_batches_deep100M_k8.sh with GP_COMM_ONLY=1), batch 1 of every mpirun,
3 runs per variant and target (2 for MPI at 0.95: one run failed). Bars: median (no whiskers; one RDMA run at 0.80 took 159 ms).
Label: communication time saved by RDMA (median MPI - median RDMA) as a % of the query-phase runtime
of the zoom; the assert checks it stays below the protocol-plus-transfer share of the zoom (the bound).
Writes subfigs/gp_rdma_comm_replay.pdf/.png
"""
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gp_mpi_vs_rdma import COL, INK, INK_2, SURFACE, clean, style

HERE = Path(__file__).resolve().parent
LOG = Path("/gpfs/scratch/delgaudio/vectors/DEEP/distbench_100M_ef_nprobe/logs/gp_abb_6996535.log")
CFG = {(1, 70): "0.80", (3, 50): "0.90", (2, 150): "0.95", (4, 400): "0.99"}
TARGETS = ["0.80", "0.90", "0.95", "0.99"]
# comm zoom of GP-ANN* (plot_breakdown6.py, job 6976103 / 6975717): query time T and
# protocol + transfer share of T, in %.
T_ZOOM_MS = {"0.80": 275.0, "0.90": 351.0, "0.95": 415.0, "0.99": 1070.0}
BOUND = {"0.80": 6.38 + 0.00545, "0.90": 5.35 + 0.00542, "0.95": 4.00 + 0.00555, "0.99": 1.82 + 0.00224}  # protocol + RDMA gain on transfer
RANKS = 8


def load():
    cur, prof = None, defaultdict(list)
    for line in open(LOG, errors="replace"):
        m = re.search(r"---- (\w+) nprobe=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            cur = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        elif cur and line.startswith("#[prof ]"):
            prof[cur].append({k: float(x) for k, x in re.findall(r"(\w+)=([\d.eE+-]+)", line)})
    d = defaultdict(list)
    for (v, n, e, _rep), rs in prof.items():
        if len(rs) < RANKS:
            continue
        b1 = rs[:RANKS]                                  # batch 1 of the process
        last = max(b1, key=lambda r: r["route_wall_s"] + r["post_req_s"])
        c = last["total_s"] - last["route_wall_s"] - last["post_req_s"] - last["search_wall_s"] - last["merge_s"]
        d[(CFG[(n, e)], v.upper())].append(1e3 * c)
    return d


def main():
    style()
    d = load()
    # half a column (two panels side by side in one column, e.g. GP-ANN* and Pyramid*), drawn at print size
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(1.65, 1.9), constrained_layout=True)
    w, ymax = 0.40, 28
    for j, t in enumerate(TARGETS):
        med = {}
        for k, name in enumerate(("MPI", "RDMA")):
            vals = d[(t, name)]
            m = st.median(vals)
            med[name] = m
            x = j + (k - 0.5) * (w + 0.03)
            ax.bar(x, m, width=w, color=COL[name], edgecolor=SURFACE, linewidth=0.6,
                   label=name if j == 0 else None)
            # no whiskers: 3 runs, and one RDMA run at 0.80 took 159 ms (stated in the caption)
        saved = 100 * (med["MPI"] - med["RDMA"]) / T_ZOOM_MS[t]
        assert saved <= BOUND[t], (t, saved, BOUND[t])
        # vertical, above the (short) RDMA bar: horizontal labels do not fit in half a column
        ax.text(j + 0.5 * (w + 0.03), med["RDMA"] + 0.6, f"−{saved:.1f}%", ha="center", va="bottom",
                rotation=90, fontsize=7, color=INK)
        print(f"{t}: MPI {med['MPI']:.2f} ms (n={len(d[(t, 'MPI')])})  RDMA {med['RDMA']:.2f} ms "
              f"(n={len(d[(t, 'RDMA')])})  saved {med['MPI'] - med['RDMA']:.2f} ms = {saved:.2f}% of T  bound {BOUND[t]:.2f}%")
    ax.set_xticks(range(len(TARGETS)), [t[1:] for t in TARGETS])   # ".80" ... to fit half a column
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Comm. time (ms)")
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.1)
    clean(ax)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "subfigs" / f"gp_rdma_comm_replay.{ext}", dpi=200)


if __name__ == "__main__":
    main()
