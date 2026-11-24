#!/usr/bin/env python3
"""QPS-recall of GP-ANN* and Pyramid* over MPI and over native RDMA (same protocol), DEEP100M, 8 nodes,
single pass, in the style and on the axes of final_curves/curve_deep100m_8n (plot_final_curves.py).
GP-ANN*: gp_first_batch.csv (jobs 6987766 + 6994617, first batch, median of 10 runs per point).
Pyramid*: job 7017381 real runs (corrected launch) (median of 6 runs per point).
MPI: solid line, filled marker; RDMA: dashed line, hollow marker; system colours as in the main curves.
Writes final_curves/curve_deep100m_8n_mpi_vs_rdma.{png,pdf}; with argument "est", GP-ANN* over RDMA is
ESTIMATED for every point (gp_estimated) and drawn dotted: final_curves/curve_deep100m_8n_mpi_vs_rdma_est.*
"""
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
PY_LOG = Path("/gpfs/scratch/delgaudio/comm_replay/logs/pyr_ab_7017381.log")
INK, GRID = "#1f1f1e", "#d9d8d4"
STYLE = {"GP-ANN*": dict(color="#2a78d6", marker="o"), "Pyramid*": dict(color="#eb6834", marker="^"),
         "CoTra": dict(color="#eda100", marker="D")}
Y_LIM = (400, 150000)          # tighter than curve_deep100m_8n, to show the MPI-RDMA gap
Y_LIM_COTRA = (400, 260000)   # with the CoTra reference curve (argument "cotra")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "axes.labelsize": 18, "xtick.labelsize": 17,
    "ytick.labelsize": 17, "legend.fontsize": 14.5, "axes.edgecolor": INK, "axes.linewidth": 1.2,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def gp_points():
    d, rec = defaultdict(list), {}
    for r in csv.DictReader(open(HERE / "gp_first_batch.csv")):
        d[(r["target"], r["variant"].upper())].append(1e4 / float(r["T_s"]))
        rec[r["target"]] = float(r["recall"])
    return {v: sorted((rec[t], st.median(d[(t, v)])) for t in rec) for v in ("MPI", "RDMA")}


def pyr_points():
    cur, qps, rec = None, defaultdict(list), defaultdict(set)
    want = False
    for line in open(PY_LOG, errors="replace"):
        m = re.search(r"---- (\w+) mode=(\w+) K=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            kef = (int(m.group(3)), int(m.group(4)))
            # only the configurations of the recall targets (the corrected job also ran other candidates)
            cur = (m.group(1).upper(), m.group(2), kef[0]) if kef in {(3, 50), (10, 100), (20, 200), (100, 800)} else None
            continue
        if not cur or cur[1] != "real":
            continue
        m = re.search(r"#\[timer\] .*qps=([\d.eE+-]+)", line)
        if m:
            qps[(cur[0], cur[2])].append(float(m.group(1)))
        if line.startswith("time || recall"):
            want = True
        elif want and re.match(r"^[0-9.]+ [0-9.]+$", line.strip()):
            rec[(cur[0], cur[2])].add(float(line.split()[1]))
            want = False
    out = {}
    for v in ("MPI", "RDMA"):
        out[v] = sorted((min(rec[(v, k)]), st.median(qps[(v, k)])) for (vv, k) in qps if vv == v)
    return out


def gp_estimated(mpi_pts):
    """GP-ANN* over RDMA, ESTIMATED for every point: median MPI query time of the same first-batch runs
    minus the communication saved by RDMA in the communication-only replay (job 6996535, batch 1,
    median MPI - median RDMA, plot_gp_rdma_comm_replay.load)."""
    from plot_gp_rdma_comm_replay import load as load_replay
    rep = load_replay()
    d = defaultdict(list)
    for r in csv.DictReader(open(HERE / "gp_first_batch.csv")):
        if r["variant"] == "mpi":
            d[r["target"]].append(1e3 * float(r["T_s"]))
    out = []
    for t, (rec, _q) in zip(("0.80", "0.90", "0.95", "0.99"), mpi_pts):
        saved = st.median(rep[(t, "MPI")]) - st.median(rep[(t, "RDMA")])
        out.append((rec, 1e7 / (st.median(d[t]) - saved)))
    return out


ESTIMATE = "est" in __import__("sys").argv[1:]
COTRA = "cotra" in __import__("sys").argv[1:]
X0 = 0.79 if "from80" in __import__("sys").argv[1:] else 0.69   # left end of the recall axis


def cotra_points():
    """CoTra (native RDMA) on DEEP100M-8n: its Pareto front from final_curves.csv, as in the main curves."""
    from plot_final_curves import load, pareto
    return [(r, q) for r, q, _c in pareto(load()[("DEEP100M-8n", "CoTra")])]


def main():
    data = {"GP-ANN*": gp_points(), "Pyramid*": pyr_points()}
    if ESTIMATE:
        data["GP-ANN*"]["RDMA"] = gp_estimated(data["GP-ANN*"]["MPI"])
    if COTRA:
        cotra_pts = [p for p in cotra_points() if p[0] >= X0 - 0.01]   # import before drawing (it sets rcParams)
    fig = plt.figure(figsize=(6.8, 5.9 if COTRA else 5.4))
    ax = fig.add_axes([0.16, 0.36, 0.80, 0.56] if COTRA else [0.16, 0.31, 0.80, 0.60])
    for t in (0.80, 0.90, 0.95, 0.99):
        ax.axvline(t, color="#7a7974", lw=1.4, ls=(0, (4, 3)), zorder=1)
    if COTRA:
        pts = cotra_pts
        st_ = STYLE["CoTra"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st_["color"], lw=2.6, marker=st_["marker"], ms=9,
                mec="white", mew=1.0, label="CoTra (native RDMA)", zorder=2)
    for system in ("GP-ANN*", "Pyramid*"):
        st_ = STYLE[system]
        for v in ("MPI", "RDMA"):
            pts = data[system][v]
            rdma = v == "RDMA"
            est = ESTIMATE and rdma and system == "GP-ANN*"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st_["color"], lw=2.6,
                    ls=(0, (1, 1.6)) if est else (0, (4, 2)) if rdma else "-", marker=st_["marker"], ms=10,
                    mfc="white" if rdma else st_["color"], mec=st_["color"] if rdma else "white", mew=1.6 if rdma else 1.0,
                    label=f"{system} {v}", zorder=4 if rdma else 3)
            print(system, v, [(round(r, 3), round(q)) for r, q in pts])
    # measured gain of every point (median QPS, RDMA vs MPI), in the empty lower-left corner
    lines = ["QPS gain, RDMA vs MPI", f"{'':9s}{'.80':>7s}{'.90':>7s}{'.95':>7s}{'.99':>7s}"]
    for system in ("GP-ANN*", "Pyramid*"):
        g = [100 * (r[1] - m[1]) / m[1] for m, r in zip(data[system]["MPI"], data[system]["RDMA"])]
        lines.append(f"{system:9s}" + "".join(f"{x:+6.1f}%" for x in g))
    ax.text(0.025, 0.04, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5,
            family="DejaVu Sans Mono", color=INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#9a9994", linewidth=1.0))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    step = 0.05 if X0 >= 0.79 else 0.1           # as plot_final_curves.py: 0.05 steps on a 0.79-1.0 range
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}" if step < 0.1 else f"{v:.1f}"))
    ax.set_xlim(X0, 1.0)
    ax.set_ylim(*(Y_LIM_COTRA if COTRA else Y_LIM))
    ax.grid(True, which="major", color=GRID, lw=0.9, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=6, width=1.2, colors=INK)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (log scale)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=14.5,
               handlelength=2.4, columnspacing=1.2, handletextpad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "final_curves" / f"curve_deep100m_8n_mpi_vs_rdma{'_est' if ESTIMATE else ''}{'_cotra' if COTRA else ''}{'_80' if X0 > 0.7 else ''}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
