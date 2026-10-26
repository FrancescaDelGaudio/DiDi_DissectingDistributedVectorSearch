#!/usr/bin/env python3
"""Recall@10 vs QPS, one figure per setup, from final_curves.csv (collect_final_curves.py).

Each system is drawn as its Pareto front (fastest configuration for a given recall); every
system has its own colour AND marker. At every recall target the operating point is the fastest real configuration with
recall >= target (no interpolation); its QPS is listed in the table under the plot ("n/r" = target not reached, with the
system's maximum recall). The same operating points are written to final_targets.csv,
to be reused by the load-balancing and communication tables.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_curves"

# Fixed identity per system (same in every figure): categorical slots 1-5 of the
# reference palette, plus a distinct marker so identity never relies on colour alone.
STYLE = {
    "GP-ANN*":         dict(color="#2a78d6", marker="o"),
    "Pyramid*":        dict(color="#eb6834", marker="^"),
    "Harmony":         dict(color="#1baf7a", marker="s"),
    "CoTra":           dict(color="#eda100", marker="D"),
    "DistributedANN*": dict(color="#e87ba4", marker="v"),
}
ORDER = list(STYLE)
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d4"

STD = [0.80, 0.90, 0.95, 0.99]
FIGS = [  # file, setup, title, systems, targets, x-range
    ("deep10m_4n", "DEEP10M-4n", "DEEP10M · 4 nodes", ORDER, STD, (0.79, 1.0)),
    ("deep10m_6n", "DEEP10M-6n", "DEEP10M · 6 nodes", ORDER, STD, (0.79, 1.0)),
    # 0.70 is also a target on TTI10M, so that DistributedANN* (max 0.76-0.77) shares the panel
    ("tti10m_4n", "TTI10M-4n", "TTI10M · 4 nodes", ["GP-ANN*", "Pyramid*", "CoTra"], [0.70] + STD, (0.79, 1.0)),
    ("tti10m_6n", "TTI10M-6n", "TTI10M · 6 nodes", ["GP-ANN*", "Pyramid*", "CoTra"], [0.70] + STD, (0.79, 1.0)),
    ("tti10m_4n_low", "TTI10M-4n", "TTI10M · 4 nodes (low-recall)",
     ["Harmony", "DistributedANN*"], [0.30, 0.40, 0.50, 0.70], (0.15, 0.80)),
    ("tti10m_6n_low", "TTI10M-6n", "TTI10M · 6 nodes (low-recall)",
     ["Harmony", "DistributedANN*"], [0.30, 0.40, 0.50, 0.70], (0.15, 0.80)),
    ("deep100m_8n", "DEEP100M-8n", "DEEP100M · 8 nodes", ORDER, STD, (0.79, 1.0)),
    ("tti100m_8n", "TTI100M-8n", "TTI100M · 8 nodes", ["GP-ANN*", "Pyramid*", "CoTra"],
     [0.80, 0.90, 0.92], (0.79, 0.98)),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16,
    "axes.edgecolor": INK, "axes.linewidth": 1.2, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def load():
    pts = defaultdict(list)
    for r in csv.DictReader(open(HERE / "final_curves.csv")):
        pts[(r["setup"], r["system"])].append((float(r["recall10"]), float(r["qps"]), r["config"]))
    return pts


def pareto(points):
    """Points not dominated by any other (higher-or-equal recall and higher QPS)."""
    front, best = [], -1.0
    for p in sorted(points, key=lambda p: (-p[0], -p[1])):
        if p[1] > best:
            front.append(p)
            best = p[1]
    return sorted(front)


def operating_point(points, target):
    ok = [p for p in points if p[0] >= target]
    return max(ok, key=lambda p: p[1]) if ok else None


def fmt_qps(q):
    return f"{q / 1000:.1f}k" if q >= 1000 else f"{q:.0f}"


def plot(fig_name, setup, title, systems, targets, xr, pts, rows_out):
    # Drawn at print size for a half-page subfigure (~3.4 in wide): every size below is
    # twice its printed size, so the figure is meant to be included at about 50%.
    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_axes([0.16, 0.31, 0.80, 0.60])

    for t in targets:
        ax.axvline(t, color="#7a7974", lw=1.4, ls=(0, (4, 3)), zorder=1)

    for s in systems:
        data = pts.get((setup, s), [])
        if not data:
            continue
        st = STYLE[s]
        front = pareto(data)
        ax.plot([p[0] for p in front], [p[1] for p in front], color=st["color"], lw=2.6,
                marker=st["marker"], ms=10, mec="white", mew=1.0, label=s, zorder=3, clip_on=True)
        for t in targets:
            op = operating_point(data, t)
            if op:
                rows_out.append(dict(setup=setup, system=s, target=t, config=op[2],
                                     recall10=op[0], qps=op[1]))
            else:
                rows_out.append(dict(setup=setup, system=s, target=t, config="", recall10="", qps=""))

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    step = 0.05 if xr[1] - xr[0] <= 0.3 else 0.1
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlim(*xr)
    ax.grid(True, which="major", color=GRID, lw=0.9, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=6, width=1.2, colors=INK)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (log scale)")
    ax.set_title(title, pad=10)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(systems), 3),
               frameon=False, handlelength=1.6, columnspacing=0.9, handletextpad=0.4, fontsize=15.5)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"curve_{fig_name}.png", dpi=300)
    plt.close(fig)


def qps_table(rows):
    """LaTeX table of the QPS at every recall target, one block per set of targets."""
    def fmt(r):
        return fmt_qps(float(r["qps"])) if r["qps"] != "" else "n/r"
    by_key = {}
    for r in rows:
        by_key.setdefault((r["setup"], r["system"]), {})[float(r["target"])] = r
    order = ["DEEP10M-4n", "DEEP10M-6n", "TTI10M-4n", "TTI10M-6n", "DEEP100M-8n", "TTI100M-8n"]
    blocks = []
    for (setup, sysname), cells in sorted(by_key.items(), key=lambda kv: (order.index(kv[0][0]), ORDER.index(kv[0][1]))):
        ts = tuple(sorted(cells))
        for b in blocks:
            if b[0] == ts:
                b[1].append((setup, sysname, cells))
                break
        else:
            blocks.append((ts, [(setup, sysname, cells)]))
    blocks.sort(key=lambda b: (-len(b[1]), -b[0][0]))
    out = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Throughput (queries per second) at each recall@10 target: fastest configuration "
           r"whose recall is at least the target, i.e.\ the operating points of "
           r"Figures~\ref{fig:curves-deep10m}--\ref{fig:curves-tti100m}. n/r: target not reached.}",
           r"\label{tab:qps-targets}", r"\begin{tabular}{llcccc}", r"\toprule"]
    for k, (ts, items) in enumerate(blocks):
        if k:
            out.append(r"\midrule")
        hdr = " & ".join(f"$\\geq {t:.2f}$" for t in ts) + " & " * (4 - len(ts))
        out += [f"\\textbf{{Setup}} & \\textbf{{System}} & {hdr} \\\\", r"\midrule"]
        prev = None
        for setup, sysname, cells in items:
            if prev is not None and setup != prev:
                out.append(r"\addlinespace")
            lab = setup.replace("-", " ") if setup != prev else ""
            prev = setup
            vals = [fmt(cells[t]) for t in ts] + [""] * (4 - len(ts))
            out.append(f"{lab} & {sysname.replace('*', '$^*$')} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (HERE / "tables_qps_targets.tex").write_text("\n".join(out))


def main():
    pts = load()
    rows_out = []
    for spec in FIGS:
        plot(*spec, pts, rows_out)
    with open(HERE / "final_targets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["setup", "system", "target", "config", "recall10", "qps"])
        w.writeheader()
        seen = set()
        for r in rows_out:  # TTI10M systems appear in one figure each; keep one row per key
            key = (r["setup"], r["system"], r["target"])
            if key not in seen:
                seen.add(key)
                w.writerow(r)
    qps_table(list(csv.DictReader(open(HERE / "final_targets.csv"))))
    print(f"wrote {len(FIGS)} figures to {OUT}, final_targets.csv and tables_qps_targets.tex")


if __name__ == "__main__":
    main()
