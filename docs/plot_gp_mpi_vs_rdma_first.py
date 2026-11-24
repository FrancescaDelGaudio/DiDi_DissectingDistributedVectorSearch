#!/usr/bin/env python3
"""GP-ANN* over MPI vs over native RDMA, FIRST BATCH (single pass, as the throughput curves).
Data: gp_first_batch.csv (collect_gp_first_batch.py): jobs 6987766 + 6994617, MPI and RDMA in separate
processes on the same saved shard graphs, 10 runs per variant and target. Same style as
plot_gp_mpi_vs_rdma.py: (a) QPS-recall, (b) QPS per target (median, whiskers min-max, RDMA - MPI);
extra panel (c) critical-path communication per target (T - busiest rank's work), median and IQR.
Writes gp_mpi_vs_rdma_first.pdf/.png and subfigs/gp_mpi_vs_rdma_first_{curve,bars,comm}.pdf/.png
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gp_mpi_vs_rdma import COL, INK, INK_2, SURFACE, clean, style

HERE = Path(__file__).resolve().parent
QUERIES = 10000
TARGETS = ["0.80", "0.90", "0.95", "0.99"]


def load():
    d, rec = defaultdict(list), {}
    for r in csv.DictReader(open(HERE / "gp_first_batch.csv")):
        d[(r["target"], r["variant"].upper())].append((float(r["T_s"]), float(r["comm_s"])))
        rec[r["target"]] = float(r["recall"])
    return d, rec


def stats(vals, iqr=False):
    if iqr:
        q = st.quantiles(vals, n=4)
        return st.median(vals), q[0], q[2]
    return st.median(vals), min(vals), max(vals)


def kfmt(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v else "0"))


def curve(ax, d, rec):
    for i, (name, marker, ls) in enumerate((("MPI", "o", "-"), ("RDMA", "s", "--"))):
        q = [stats([QUERIES / t for t, _ in d[(t_, name)]])[0] for t_ in TARGETS]
        ax.plot([rec[t] for t in TARGETS], q, ls, marker=marker, color=COL[name], linewidth=1.6, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"GP-ANN* over {name}", zorder=3 + i)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_xlim(0.80, 1.0)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 45000)
    kfmt(ax)
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.5)


def bars(ax, d, value, ylabel, ymax, fmt_delta, iqr=False, legend_loc="upper right", label_target=False):
    w = 0.36
    for j, t in enumerate(TARGETS):
        meds, tops = {}, []
        for k, name in enumerate(("MPI", "RDMA")):
            m, lo, hi = stats([value(x) for x in d[(t, name)]], iqr)
            meds[name] = m
            tops.append(hi)
            x = j + (k - 0.5) * (w + 0.03)
            ax.bar(x, m, width=w, color=COL[name], edgecolor=SURFACE, linewidth=0.6,
                   label=f"GP-ANN* over {name}" if j == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK_2, linewidth=0.7, capsize=2)
        txt = fmt_delta(t, meds["MPI"], meds["RDMA"]) if label_target else fmt_delta(meds["MPI"], meds["RDMA"])
        ax.text(j, max(tops) + ymax * 0.02, txt, ha="center", va="bottom", fontsize=6.5, color=INK_2,
                linespacing=1.1)
    ax.set_xticks(range(len(TARGETS)), TARGETS)
    ax.set_xlabel("Recall@10 target")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, loc=legend_loc, ncol=2 if legend_loc == "upper center" else 1)
    clean(ax)


def qps_bars(ax, d, rec):
    bars(ax, d, lambda x: QUERIES / x[0], "Throughput (QPS)", 45000,
         lambda a, b: f"RDMA {100 * (b - a) / a:+.1f}%")
    kfmt(ax)


# Protocol + transfer share of the runtime in the comm zoom of GP-ANN* (plot_breakdown6.py, single
# pass): the bound on what a native transport can save.
BOUND = {"0.80": 6.4, "0.90": 5.4, "0.95": 4.0, "0.99": 1.9}


def comm_bars(ax, d, rec):
    # whiskers: interquartile range (a few runs have tens of ms more, from search-time noise of the
    # busiest rank, which would hide the bars). Label: communication saved by RDMA as a share of the
    # MPI query time (medians), next to the bound of the zoom.
    def label(t, a, b):
        t_mpi = 1e3 * st.median(x[0] for x in d[(t, "MPI")])
        return f"−{100 * (a - b) / t_mpi:.1f}% of T\n(bound {BOUND[t]:.1f}%)"
    bars(ax, d, lambda x: 1e3 * x[1], "Critical-path communication (ms)", 50,
         label, iqr=True, legend_loc="upper center", label_target=True)


def main():
    style()
    d, rec = load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    curve(a, d, rec)
    qps_bars(b, d, rec)
    a.set_title("(a) QPS-recall, first batch", fontsize=8, color=INK)
    b.set_title("(b) Per recall target, first batch", fontsize=8, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"gp_mpi_vs_rdma_first.{ext}", dpi=200)
    plt.close(fig)
    for name, fn in (("curve", curve), ("bars", qps_bars), ("comm", comm_bars)):
        f, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
        fn(ax, d, rec)
        for ext in ("pdf", "png"):
            f.savefig(HERE / "subfigs" / f"gp_mpi_vs_rdma_first_{name}.{ext}", dpi=200)
        plt.close(f)


if __name__ == "__main__":
    main()
