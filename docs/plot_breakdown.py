#!/usr/bin/env python3
"""Runtime breakdown at fixed recall@10 (DEEP100M, 8 nodes, 16 threads per node).

One figure per system, one stacked bar per recall target, segments = % of the batch
query-phase runtime. Same phase -> same colour in every figure.

Usage (on the cluster):
    module load anaconda3/2023.09
    python3 plot_breakdown.py   # writes breakdown_<system>.png/.pdf next to this file

Numbers come from the breakdown jobs (see breakdown_tables.tex for the per-system notes).
To add a system, append an entry to SYSTEMS.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Phase -> (label, colour, hatch). Colours: validated reference categorical palette;
# adjacent segments pass the normal-vision (>=15) and colour-blind (>=8 dE) checks, except
# "MPI + waiting" next to magenta-free neighbours at 6-8, which carries a hatch.
PHASES = {
    "routing":   ("Routing",                       "#2a78d6", None),
    "fill":      ("Routing / pipeline fill",        "#2a78d6", None),
    "perhop":    ("Per-hop routing",                "#eb6834", None),
    "qdist":     ("Query distribution",             "#eda100", None),
    "search":    ("Local search",                   "#1baf7a", None),
    "merge":     ("Merge",                          "#008300", None),
    "mpi":       ("MPI transfer",                   "#e87ba4", None),
    "wait":      ("Waiting for slowest node",       "#6250d6", None),
    "idle":      ("Shard idle (no routed queries)", "#6250d6", "////"),
    "remote":    ("Remote tasks (search for other nodes)", "#e34948", None),
    "steal":     ("Work stealing",                  "#1baf7a", "\\\\"),
    "rdma":      ("RDMA polling",                   "#e87ba4", None),
    "cwait":     ("Idle (waiting for remote work)", "#6250d6", None),
    "mpiwait":   ("MPI + waiting",                  "#e87ba4", "////"),
    "other":     ("Other",                          "#e2e1dd", None),
}
ORDER = ["routing", "fill", "perhop", "qdist", "search", "steal", "remote", "merge", "mpi", "rdma", "idle", "wait",
         "cwait", "mpiwait", "other"]

# (row label, measured recall, parameters, queries/s, {phase: % of runtime})
SYSTEMS = [
    ("DistributedANN", [
        ("0.80†", 0.796, "BW=32",  21432, {"routing": 31.1, "perhop": 6.4, "search": 28.5, "merge": 6.2, "mpi": 11.7, "wait": 3.5, "other": 12.6}),
        ("0.90", 0.901, "BW=128", 11656, {"routing": 16.8, "perhop": 9.4, "search": 32.3, "merge": 8.0, "mpi": 8.7, "wait": 8.2, "other": 16.6}),
        ("0.93*", 0.928, "BW=512", 4705, {"routing": 7.4, "perhop": 8.0, "search": 37.5, "merge": 9.9, "mpi": 12.6, "wait": 9.9, "other": 14.7}),
    ]),
    ("GP-ANN", [
        ("0.80", 0.821, "np=1, ef=70",  36306, {"routing": 62.7, "search": 21.0, "merge": 0.3, "mpiwait": 16.0}),
        ("0.90", 0.902, "np=3, ef=50",  20793, {"routing": 38.8, "search": 23.2, "merge": 0.6, "mpiwait": 37.4}),
        ("0.95", 0.949, "np=2, ef=150", 18475, {"routing": 27.4, "search": 36.6, "merge": 0.3, "mpiwait": 35.8}),
        ("0.99", 0.992, "np=4, ef=400", 7189,  {"routing": 12.1, "search": 52.4, "merge": 0.3, "mpiwait": 35.2}),
    ]),
    ("Harmony", [
        ("0.80*", 0.788, "nprobe=6",  2995, {"routing": 22.8, "qdist": 31.4, "search": 20.4, "mpiwait": 22.8, "other": 2.6}),
        ("0.90", 0.902, "nprobe=14", 2524, {"routing": 19.3, "qdist": 26.0, "search": 31.5, "mpiwait": 20.5, "other": 2.7}),
        ("0.95", 0.954, "nprobe=28", 1634, {"routing": 12.3, "qdist": 18.1, "search": 40.3, "mpiwait": 25.2, "other": 4.1}),
        ("0.99", 0.991, "nprobe=96", 671,  {"routing": 5.4, "qdist": 7.4, "search": 49.8, "mpiwait": 31.8, "other": 5.6}),
    ]),
    # Pyramid v2 (sweep 6963795): sender, shards and receiver run as a pipeline, so per-process
    # phases overlap. Average shard over the 8 shards, as % of the query-phase wall time W
    # (receiver): pipeline fill = W - longest shard window (common to all shards); within its
    # window a shard searches (search_util x window), does MPI (thread-0 recv+send) or is idle
    # (waiting for routed queries); after its window it waits for the slowest shard.
    ("Pyramid", [
        ("0.80", 0.807, "K=3, ef=50",    13736, {"fill": 16.3, "search": 8.3,  "mpi": 1.1, "idle": 50.4, "wait": 23.9}),
        ("0.90", 0.915, "K=10, ef=100",  6871,  {"fill": 17.5, "search": 25.4, "mpi": 0.7, "idle": 39.5, "wait": 17.0}),
        ("0.95", 0.961, "K=20, ef=200",  3529,  {"fill": 12.2, "search": 44.4, "mpi": 0.3, "idle": 20.3, "wait": 22.8}),
        ("0.99", 0.994, "K=100, ef=800", 719,   {"search": 63.2, "mpi": 0.1, "idle": 3.3, "wait": 33.4}),
    ]),
    # CoTra (job 6962171, profiler pass; QPS from the non-profiled pass of the same job). Event-loop
    # time of the 16 threads summed over the 8 nodes: query intake (g_query_Q), local subqueries
    # (g_subquery_Q), work stealing (g_steal), remote tasks (g_task_Q), result merge (g_result_Q),
    # RDMA completion polling (g_poll_proc), idle remainder (f_wait_time).
    ("CoTra", [
        ("0.80*", 0.797, "ef=18",  173771, {"routing": 1.1, "search": 44.8, "steal": 13.5, "remote": 4.0, "merge": 0.2, "rdma": 2.2, "cwait": 34.2}),
        ("0.90",  0.909, "ef=50",  99927,  {"routing": 0.7, "search": 45.7, "steal": 13.1, "remote": 3.8, "merge": 0.2, "rdma": 2.4, "cwait": 34.1}),
        ("0.95",  0.956, "ef=100", 60933,  {"routing": 0.4, "search": 46.9, "steal": 13.9, "remote": 3.4, "merge": 0.2, "rdma": 2.4, "cwait": 32.7}),
        ("0.99",  0.992, "ef=400", 18529,  {"routing": 0.2, "search": 44.6, "steal": 18.1, "remote": 3.2, "merge": 0.2, "rdma": 2.8, "cwait": 30.9}),
    ]),
]

NOTES = {
    "DistributedANN": "† Measured at BW=32, recall 0.796 (the 0.80 point of the recall/QPS sweep is BW=64, recall 0.831).\n"
                      "* Highest recall reached by DistributedANN on this index (0.93).",
    "CoTra": "* Measured at ef=18, recall 0.797 (the 0.80 point of the recall/QPS sweep is ef=20).",
    "Harmony": "* Measured at nprobe=6, recall 0.788 (the 0.80 point of the recall/QPS sweep is nprobe=8, recall 0.833).",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"


def ink_on(hex_colour):
    """Dark or white label text, whichever reads better on the fill."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if lum > 0.28 else "#ffffff"


def plot_system(name, runs, out):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "hatch.color": "#ffffff",
        "hatch.linewidth": 0.8,
    })
    has_note = any(t.endswith("*") for t, *_ in runs)
    fig, ax = plt.subplots(figsize=(7.0, 0.38 * len(runs) + 1.05 + (0.15 if has_note else 0)))
    used = []
    for y, (target, recall, params, qps, shares) in enumerate(runs):
        left = 0.0
        for ph in ORDER:
            w = shares.get(ph, 0.0)
            if w <= 0:
                continue
            if ph not in used:
                used.append(ph)
            _, colour, hatch = PHASES[ph]
            ax.barh(y, w, left=left, height=0.62, color=colour, hatch=hatch,
                    edgecolor=SURFACE, linewidth=1.0)
            if w >= 6:
                # on hatched fills, a solid chip behind the number keeps it readable
                chip = dict(boxstyle="round,pad=0.15", facecolor=colour, edgecolor="none") if hatch else None
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center",
                        fontsize=7, color=ink_on(colour), bbox=chip)
            left += w
        label = f"{qps/1000:.1f}k q/s" if qps >= 1000 else f"{qps:.0f} q/s"
        ax.text(101.5, y, label, ha="left", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([t for t, *_ in runs], fontsize=8)
    ax.set_ylabel("Recall@10 target")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of query-phase runtime (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.suptitle(name, x=0.01, ha="left", fontsize=9, fontweight="bold", color=INK)
    handles = [Patch(facecolor=PHASES[p][1], hatch=PHASES[p][2], edgecolor=SURFACE, label=PHASES[p][0])
               for p in ORDER if p in used]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=(len(handles) if len(handles) <= 4 else 3 if len(handles) <= 6 else 4),
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2)
    if has_note:
        fig.text(0.01, 0.01, NOTES[name], fontsize=6.5, color=INK_2)
    fig.tight_layout(rect=(0, (0.04 * (NOTES[name].count("\n") + 1)) if has_note else 0, 1, 0.98))
    slug = name.lower().replace("-", "")
    fig.savefig(out / f"breakdown_{slug}.pdf")
    fig.savefig(out / f"breakdown_{slug}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out / f'breakdown_{slug}.png'} (+ .pdf)")


def main():
    out = Path(__file__).resolve().parent
    for name, runs in SYSTEMS:
        plot_system(name, runs, out)


if __name__ == "__main__":
    main()
