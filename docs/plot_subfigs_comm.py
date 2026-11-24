#!/usr/bin/env python3
"""Two-step breakdown subfigures (DEEP100M, 8 nodes), same panel size and fonts as plot_subfigs.py.

  A. breakdown4_system_{gpann,harmony,cotra,dann,pyramid}: routing | local search | merge | communication
     (| other for DistributedANN*), % of the query-phase runtime T, recall targets on x.
  B. comm_zoom_{gpann,harmony,dann,pyramid}: the communication only, split into transfer | protocol |
     waiting (| other for DistributedANN*), still % of T, same y scale in every panel.
  legend_breakdown4.pdf/png, legend_comm_zoom.pdf/png: one legend per row.

Data: GP-ANN*, Harmony and DistributedANN* from plot_breakdown6.py (see its docstring for jobs and
formulas); DistributedANN*'s "other" is split here into the part of its exchanges that takes longer
in the real run than in the replay (communication) and the time outside its profiled phases (other).
CoTra: plot_breakdown.py (job 6962171): routing = query intake, search = local subqueries + work
stealing + remote tasks, merge = result merge, communication = RDMA polling + idle; CoTra is not replayed.

Usage:
    module load anaconda3/2023.09
    python3 plot_subfigs_comm.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_breakdown6 import SYSTEMS as SYS6, INK, INK_2, SURFACE, ink_on  # noqa: E402
from plot_breakdown import SYSTEMS as SYS_OLD  # noqa: E402

OUT = Path(__file__).resolve().parent / "subfigs"
TEXTWIDTH = 7.0
H = 2.3

A_CATS = {  # label, colour, hatch
    "routing": ("Routing",       "#2a78d6", None),
    "search":  ("Local search",  "#1baf7a", None),
    "merge":   ("Merge",         "#008300", None),
    "comm":    ("Communication", "#e87ba4", "////"),
    "other":   ("Other",         "#e2e1dd", None),
}
B_CATS = {  # zoom: one bar per component (log axis), in this order inside each recall group
    "transfer": ("Transfer (MPI)", "#7a7974", None),
    "rdma":     ("RDMA gain on transfer", "#c4c3bf", "////"),
    "protocol": ("Protocol",       "#eda100", None),
    "waiting":  ("Waiting",        "#6250d6", None),
    "extra":    ("Other (exchanges slower in the real run)", "#e2e1dd", None),
}
# RDMA gain = T_MPI - T_RDMA of the transfer-only replays (job 6975717), % of T.
RDMA_GAIN = {
    "GP-ANN*":         {"0.80": 0.00545, "0.90": 0.00542, "0.95": 0.00555, "0.99": 0.00224},
    "Harmony":         {"0.80*": 0.00647, "0.90": 0.00560, "0.95": 0.00109, "0.99": 0.00107},
    "DistributedANN*": {"0.80": 0.110, "0.90": 0.120, "0.93*": 0.193},
    "Pyramid*":        {"0.80": 1.1146, "0.90": 0.8613, "0.95": 0.5544, "0.99": 0.2096},  # job 7017380
}
ZOOM_LOG = (1e-4, 100.0)
# DistributedANN*: "other" of plot_breakdown6 = extra (exchange time not reproduced by the replay)
# + unprofiled (time outside the profiled phases), from the profiled runs of job 6975717.
DA_SPLIT = {"0.80": (10.18, 11.85), "0.90": (8.97, 19.33), "0.93*": (10.61, 18.16)}
SLUG = {"GP-ANN*": "gpann", "Harmony": "harmony", "DistributedANN*": "dann", "CoTra": "cotra",
        "Pyramid*": "pyramid"}
# Pyramid* (job 7017380: corrected launch, rankfile slot ranges = 16 real cores per process; the earlier
# job 6977296 pinned every process to one core). K:ef = 3:50 / 10:100 / 20:200 / 100:800, medians of 5 real
# runs; collect_pyramid_breakdown_fixed.py. Average shard, % of the receiver wall time W. Routing = time the
# sender waits for routed queries, at most the shards' idle time (routing delays the pipeline only then);
# search = search utilisation x window; merge = 0 (on the receiver); communication = the rest.
PYRAMID_A = [  # (target, QPS, routing, search)
    ("0.80", 56811, 0.33, 35.93), ("0.90", 33215, 0.16, 42.98), ("0.95", 18256, 0.12, 48.06), ("0.99", 3833, 1.10, 52.92),
]
# Communication split (job 7017380, % of W): transfer = T_MPI of the transfer-only replay; protocol =
# C_replay - T_MPI with C_replay = time the sender spends in MPI_Send in the PYR_COMM_ONLY runs (median
# of 3; the sender's wait for routed queries is routing, not communication); waiting = the rest of the
# communication. No cap (removed from the method for every system): protocol + transfer exceed
# T - slowest shard (47.3 / 28.8 / 16.1 / 3.3 %), so the protocol share is an upper bound.
PYRAMID_B = {
    "0.80": {"transfer": 4.433, "protocol": 48.80, "waiting": 10.51},
    "0.90": {"transfer": 3.460, "protocol": 28.91, "waiting": 24.49},
    "0.95": {"transfer": 2.288, "protocol": 16.62, "waiting": 32.92},
    "0.99": {"transfer": 0.794, "protocol": 3.45,  "waiting": 41.73},
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK_2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.labelsize": 8, "hatch.color": "#ffffff", "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def runs_split():
    """system -> [(target, qps, A shares, B shares)]"""
    out = {}
    for name, runs in SYS6:
        rows = []
        for target, _recall, qps, s in runs:
            extra, unprof = DA_SPLIT[target] if name == "DistributedANN*" else (0.0, s.get("other", 0.0))
            b = {"transfer": s.get("transfer", 0.0), "protocol": s.get("protocol", 0.0),
                 "waiting": s.get("waiting", 0.0), "extra": extra}
            a = {"routing": s.get("routing", 0.0), "search": s.get("search", 0.0), "merge": s.get("merge", 0.0),
                 "comm": sum(b.values()), "other": unprof}
            b["rdma"] = RDMA_GAIN[name][target]
            assert abs(sum(a.values()) - 100.0) < 0.2, (name, target, sum(a.values()))
            rows.append((target, qps, a, b))
        out[name] = rows
    # CoTra from the original breakdown (no replay)
    rows = []
    for target, _recall, _params, qps, s in dict(SYS_OLD)["CoTra"]:
        a = {"routing": s["routing"], "search": s["search"] + s["steal"] + s["remote"], "merge": s["merge"],
             "comm": s["rdma"] + s["cwait"]}
        rows.append((target, qps, a, None))
    out["CoTra"] = rows
    rows = []
    for t, q, r, se in PYRAMID_A:
        b = dict(PYRAMID_B[t], extra=0.0, rdma=RDMA_GAIN["Pyramid*"][t])
        a = {"routing": r, "search": se, "merge": 0.0, "comm": 100.0 - r - se}
        assert abs(a["comm"] - sum(v for c, v in b.items() if c != "rdma")) < 0.1, (t, a["comm"])
        rows.append((t, q, a, b))
    out["Pyramid*"] = rows
    return out


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def qps_label(qps):
    return f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000 else f"{qps:.0f}"


def axes_style(ax, n, ymax):
    ax.set_xlim(-0.6, n - 0.4)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of query-phase runtime", labelpad=2)


def stack(ax, x, shares, cats, label_min, fs=6.5):
    bottom = 0.0
    for c, (_, colour, hatch) in cats.items():
        h = shares.get(c, 0.0)
        if h <= 0:
            continue
        ax.bar(x, h, bottom=bottom, width=0.72, color=colour, hatch=hatch, edgecolor=SURFACE, linewidth=0.6)
        if h >= label_min:
            chip = dict(boxstyle="round,pad=0.1", facecolor=colour, edgecolor="none") if hatch else None
            ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=fs, color=ink_on(colour), bbox=chip)
        bottom += h
    return bottom


def panel_a(name, rows):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH / 5 - 0.06, H))
    for x, (target, qps, a, _) in enumerate(rows):
        stack(ax, x, a, A_CATS, 10)
        ax.text(x, 101.5, qps_label(qps), ha="center", va="bottom", fontsize=6, color=INK_2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    axes_style(ax, len(rows), 100)
    save(fig, f"breakdown4_system_{SLUG[name]}")


def panel_b(name, rows):
    """Zoom on the communication: one bar per component, log axis (stacked bars would hide the
    components below 1% of T, and stacking is meaningless on a log axis)."""
    cats = [c for c in B_CATS if c != "extra" or any(r[3].get(c, 0.0) > 0 for r in rows)]
    slot = 0.84 / len(cats)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH / 4 - 0.06, H))
    lo, hi = ZOOM_LOG
    for x, (target, _qps, _a, b) in enumerate(rows):
        for i, c in enumerate(cats):
            _, colour, hatch = B_CATS[c]
            xc = x - 0.42 + slot * (i + 0.5)
            v = b.get(c, 0.0)
            if v <= 0:   # measured as zero (Harmony's protocol): mark it on the axis
                ax.text(xc, lo * 1.25, "0", ha="center", va="bottom", fontsize=5.5, color=INK_2)
                continue
            edge = INK_2 if c == "rdma" else SURFACE
            ax.bar(xc, v - lo, bottom=lo, width=slot * 0.9, color=colour, hatch=hatch, edgecolor=edge,
                   linewidth=0.6 if c == "rdma" else 0.3, linestyle=(0, (1.5, 1)) if c == "rdma" else "-")
        ax.text(x, hi * 1.08, f"{sum(b[c] for c in B_CATS if c != 'rdma'):.0f}%", ha="center", va="bottom",
                fontsize=6, color=INK_2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    axes_style(ax, len(rows), None)
    ax.set_ylim(lo, hi)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    for x in range(1, len(rows)):
        ax.axvline(x - 0.5, color="#e6e5e1", linewidth=0.5)
    save(fig, f"comm_zoom_{SLUG[name]}")


def legend(name, cats):
    handles = [Patch(facecolor=c, hatch=h, edgecolor=INK_2 if c == "#c4c3bf" else SURFACE, label=l)
               for l, c, h in cats.values()]
    fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=8,
               handlelength=1.8, columnspacing=1.4)
    save(fig, name)


def main():
    data = runs_split()
    for name, rows in data.items():
        panel_a(name, rows)
        if rows[0][3] is not None:
            panel_b(name, rows)
    legend("legend_breakdown4", A_CATS)
    legend("legend_comm_zoom", B_CATS)
    print(f"wrote breakdown4_system_*, comm_zoom_* and their legends to {OUT}")


if __name__ == "__main__":
    main()
