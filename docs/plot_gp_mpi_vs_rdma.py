#!/usr/bin/env python3
"""GP-ANN* over MPI vs over native RDMA (same protocol and termination, only the transport changes),
real runs at steady state: job 6996614 (sbatch_alt_transport_deep100M_k8.sh). DEEP100M, 8 nodes,
16 threads, k=10, 10K queries. Each mpirun loads the shard graphs once and alternates the transport
batch by batch in the same process (batches 1-2 excluded, then MPI RDMA RDMA MPI ...); per mpirun the
median of 10 MPI and 10 RDMA batches, 3 mpiruns per target. QPS = 10K / batch time.
Panels: (a) QPS-recall curve, (b) QPS per recall target with the RDMA - MPI difference.
Writes gp_mpi_vs_rdma.pdf/.png and subfigs/gp_mpi_vs_rdma_{curve,bars}.pdf
"""
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
COL = {"MPI": "#7a7974", "RDMA": "#6250d6"}
QUERIES = 10000
# target, recall@10, batch time (ms) of the 3 mpiruns: MPI, RDMA  (gp_alt_6996614_analysis.txt)
RUNS = [
    ("0.80", 0.8203, (201.3, 198.6, 203.2), (197.2, 195.2, 202.3)),
    ("0.90", 0.9023, (263.1, 277.6, 261.8), (262.9, 275.4, 260.4)),
    ("0.95", 0.9487, (422.6, 380.1, 378.6), (419.8, 380.1, 379.1)),
    ("0.99", 0.9913, (1042.9, 1066.1, 1055.2), (1035.6, 1070.6, 1049.0)),
]


def qps(ms_list):
    q = [QUERIES / (m / 1e3) for m in ms_list]
    return st.median(q), min(q), max(q)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK_2, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def curve(ax):
    for i, (name, marker, ls) in enumerate((("MPI", "o", "-"), ("RDMA", "s", "--"))):
        xs = [r[1] for r in RUNS]
        q = [qps(r[2 + i]) for r in RUNS]
        ax.plot(xs, [m for m, _, _ in q], ls, marker=marker, color=COL[name], linewidth=1.6, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"GP-ANN* over {name}", zorder=3 + i)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_xlim(0.80, 1.0)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 55000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v else "0"))
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)


def bars(ax):
    w = 0.36
    for j, (target, _rec, mpi, rdma) in enumerate(RUNS):
        vals = {}
        for k, (name, data) in enumerate((("MPI", mpi), ("RDMA", rdma))):
            m, lo, hi = qps(data)
            vals[name] = m
            x = j + (k - 0.5) * (w + 0.03)
            ax.bar(x, m, width=w, color=COL[name], edgecolor=SURFACE, linewidth=0.6,
                   label=f"GP-ANN* over {name}" if j == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK_2, linewidth=0.7, capsize=2)
        d = 100 * (vals["RDMA"] - vals["MPI"]) / vals["MPI"]
        top = max(qps(mpi)[2], qps(rdma)[2])
        ax.text(j, top * 1.03 + 600, f"RDMA {d:+.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(RUNS)), [r[0] for r in RUNS])
    ax.set_xlabel("Recall@10 target")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_ylim(0, 55000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v else "0"))
    ax.legend(frameon=False, loc="upper right")
    clean(ax)


def main():
    style()
    out = Path(__file__).resolve().parent
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    curve(a)
    bars(b)
    a.set_title("(a) QPS-recall", fontsize=8, color=INK)
    b.set_title("(b) Per recall target", fontsize=8, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gp_mpi_vs_rdma.{ext}", dpi=200)
    plt.close(fig)
    for name, fn in (("curve", curve), ("bars", bars)):
        f, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
        fn(ax)
        f.savefig(out / "subfigs" / f"gp_mpi_vs_rdma_{name}.pdf")
        f.savefig(out / "subfigs" / f"gp_mpi_vs_rdma_{name}.png", dpi=200)
        plt.close(f)
    for target, rec, mpi, rdma in RUNS:
        m, r = qps(mpi)[0], qps(rdma)[0]
        print(f"{target} recall {rec}: QPS MPI {m:.0f}  RDMA {r:.0f}  diff {100*(r-m)/m:+.2f}%")


if __name__ == "__main__":
    main()
