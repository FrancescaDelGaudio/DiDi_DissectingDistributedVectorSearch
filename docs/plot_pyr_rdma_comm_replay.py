#!/usr/bin/env python3
"""Pyramid* over MPI vs native RDMA: communication in the communication-only replay (PYR_COMM_ONLY=1,
shards answer with a dummy top-k), single pass. Same quantity as C_replay of the Pyramid* comm zoom:
time the sender spends in Send. Twin of plot_gp_rdma_comm_replay.py (same size and style, half a column).
Data: job 7017381 (corrected launch, 16 real cores) (Distributed-HNSW-parallel_timer/script/sbatch_ab_mpi_rdma_deep100m_8n.sh), comm runs,
3 per variant and target. Bars: median. Label: communication time saved by RDMA (median MPI - median RDMA)
as a % of the query-phase runtime of the zoom (receiver wall time W, job 7017380); the assert checks it
stays below the protocol-plus-transfer share of the zoom (the bound).
Writes subfigs/pyr_rdma_comm_replay.pdf/.png
"""
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gp_mpi_vs_rdma import COL, INK, SURFACE, clean, style

HERE = Path(__file__).resolve().parent
LOG = Path("/gpfs/scratch/delgaudio/comm_replay/logs/pyr_ab_7017381.log")
CFG = {(3, 50): "0.80", (10, 100): "0.90", (20, 200): "0.95", (100, 800): "0.99"}
TARGETS = ["0.80", "0.90", "0.95", "0.99"]
# Pyramid* comm zoom (plot_subfigs_comm.py PYRAMID_A/B, job 7017380): W = 10K / QPS, and
# protocol + transfer share of W in %.
T_ZOOM_MS = {"0.80": 1e7 / 56811, "0.90": 1e7 / 33215, "0.95": 1e7 / 18256, "0.99": 1e7 / 3833}
BOUND = {"0.80": 48.80 + 1.1146, "0.90": 28.91 + 0.8613, "0.95": 16.62 + 0.5544, "0.99": 3.45 + 0.2096}  # protocol + RDMA gain on transfer (no cap)


def load():
    cur, d = None, defaultdict(list)
    for line in open(LOG, errors="replace"):
        m = re.search(r"---- (\w+) mode=(\w+) K=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            key = (int(m.group(3)), int(m.group(4)))
            cur = (m.group(1).upper(), m.group(2), CFG[key]) if key in CFG else None   # only the 4 targets
            continue
        m = re.search(r"#\[prof \] sender .* send_s=([\d.]+) ", line)
        if m and cur and cur[1] == "comm":
            d[(cur[2], cur[0])].append(1e3 * float(m.group(1)))
    return d


def main():
    style()
    d = load()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(1.65, 1.9), constrained_layout=True)
    w, ymax = 0.40, 160
    for j, t in enumerate(TARGETS):
        med = {}
        for k, name in enumerate(("MPI", "RDMA")):
            vals = d[(t, name)]
            med[name] = st.median(vals)
            x = j + (k - 0.5) * (w + 0.03)
            ax.bar(x, med[name], width=w, color=COL[name], edgecolor=SURFACE, linewidth=0.6,
                   label=name if j == 0 else None)
        saved = 100 * (med["MPI"] - med["RDMA"]) / T_ZOOM_MS[t]
        assert saved <= BOUND[t], (t, saved, BOUND[t])
        ax.text(j + 0.5 * (w + 0.03), med["RDMA"] + ymax * 0.02, f"−{saved:.1f}%", ha="center", va="bottom",
                rotation=90, fontsize=7, color=INK)
        print(f"{t}: MPI {med['MPI']:.1f} ms (n={len(d[(t, 'MPI')])})  RDMA {med['RDMA']:.1f} ms "
              f"(n={len(d[(t, 'RDMA')])})  saved {med['MPI'] - med['RDMA']:.1f} ms = {saved:.2f}% of T  bound {BOUND[t]:.2f}%")
    ax.set_xticks(range(len(TARGETS)), [t[1:] for t in TARGETS])
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Comm. time (ms)")
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.1)
    clean(ax)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "subfigs" / f"pyr_rdma_comm_replay.{ext}", dpi=200)


if __name__ == "__main__":
    main()
