#!/usr/bin/env python3
"""GP-ANN* over MPI vs over native RDMA, QPS-recall in the style and on the axes of
final_curves/curve_deep100m_8n (same size, fonts, log QPS axis with the same range, recall targets as
dashed lines). Data: plot_gp_mpi_vs_rdma.RUNS (job 6996614, steady state, same process, median of 3 runs).
Writes final_curves/curve_deep100m_8n_gp_mpi_vs_rdma.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter, MultipleLocator

from plot_gp_mpi_vs_rdma import RUNS, qps

HERE = Path(__file__).resolve().parent
INK, GRID = "#1f1f1e", "#d9d8d4"
STYLE = {  # MPI keeps the GP-ANN* identity of the main curves; RDMA a colour no system uses
    "MPI":  dict(color="#2a78d6", marker="o", ls="-"),
    "RDMA": dict(color="#6250d6", marker="p", ls=(0, (4, 2))),
}
Y_RANGE = (381.9, 210291.7)   # min / max QPS of all systems in curve_deep100m_8n: same autoscaled y axis

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "axes.labelsize": 18,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 15.5,
    "axes.edgecolor": INK, "axes.linewidth": 1.2, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def main():
    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_axes([0.16, 0.31, 0.80, 0.60])
    for t in (0.80, 0.90, 0.95, 0.99):
        ax.axvline(t, color="#7a7974", lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.plot([0.9, 0.9], Y_RANGE, alpha=0)   # same y autoscale as the main figure
    for i, name in enumerate(("MPI", "RDMA")):
        st = STYLE[name]
        ax.plot([r[1] for r in RUNS], [qps(r[2 + i])[0] for r in RUNS], color=st["color"], ls=st["ls"],
                lw=2.6, marker=st["marker"], ms=10 if name == "MPI" else 9, mec="white", mew=1.0,
                label=f"GP-ANN* over {name}", zorder=3 + i)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlim(0.69, 1.0)
    ax.grid(True, which="major", color=GRID, lw=0.9, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=6, width=1.2, colors=INK)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (log scale)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               handlelength=2.2, columnspacing=1.2, handletextpad=0.4)
    out = HERE / "final_curves"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"curve_deep100m_8n_gp_mpi_vs_rdma.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
