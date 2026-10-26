#!/usr/bin/env python3
"""Runtime breakdown with the communication split into transfer, protocol and waiting
(DEEP100M, 8 nodes, 16 threads per node, 10,000 queries, k = 10). One panel per system,
recall targets on x, % of the batch query-phase runtime T (average node).

Categories (bottom to top): routing, local search, merge, transfer, protocol, waiting, other.
  C_real   = T - routing - search - merge (communication time of the average node)
  C_replay = communication-only replay (local search replaced by a same-size dummy result)
  T_MPI    = transfer-only replay with the system's MPI calls (comm_replay/transport_replay)
  transfer = T_MPI, protocol = C_replay - T_MPI, waiting = C_real - C_replay, with
  protocol + transfer capped at T - max over ranks of (routing + search + merge).

Sources:
  GP-ANN*   real runs: job 6976103 (median of 5 runs, shard graphs loaded); replays: job 6975717.
            C_replay = communication of the rank that finishes routing last (17.6/19.0/16.8/19.8 ms):
            the other ranks also wait for the slowest router, which is waiting, not protocol.
            No cap needed (C_replay 6.4/5.4/4.1/1.9% < max removable 6.9/7.0/5.4/2.6%).
  Harmony   real runs: jobs 6961000 (nprobe 6) and 6960833 (nprobe 14, 28, 96), shares of
            plot_breakdown.py; routing = IVF routing + query distribution minus the replayed
            broadcast (the distribution time is the workers' query preparation); communication =
            "MPI + waiting" + heap exchange + replayed broadcast; replays: job 6975717 on 8 nodes
            (complete = transfer-only, so protocol = 0). Merge (final sort) < 0.1%.
  DA*       profiled runs of job 6975717 (barrier before every exchange, mean rank):
            routing = head search + node selection, search = scoring + response packing,
            waiting = barrier wait, transfer = T_MPI, protocol = per-round Allreduce (replay),
            other = exchange time not reproduced by the replay + time outside the profiled phases.
The RDMA gain (T_MPI - T_RDMA, at most 0.2% of T) is too small to draw; see the RDMA table.
Colours: breakdown figures' routing / search / merge; transfer / protocol / waiting / other
checked for adjacent-pair separation in this order (normal >= 15, protan/deutan/tritan >= 8 dE).

Usage:
    module load anaconda3/2023.09
    python3 plot_breakdown6.py   # writes breakdown6_by_system.png/.pdf next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"

CATS = {  # category -> (label, colour), bottom to top
    "routing":  ("Routing",      "#2a78d6"),
    "search":   ("Local search", "#1baf7a"),
    "merge":    ("Merge",        "#008300"),
    "transfer": ("Transfer",     "#7a7974"),
    "protocol": ("Protocol",     "#eda100"),
    "waiting":  ("Waiting",      "#6250d6"),
    "other":    ("Other",        "#e2e1dd"),
}

# system -> [(target label, measured recall, QPS, {category: % of T})]
SYSTEMS = [
    ("GP-ANN*", [
        ("0.80", 0.820, 36367, {"routing": 55.11, "search": 24.99, "merge": 0.29, "transfer": 0.034,
                                "protocol": 6.38, "waiting": 13.20}),
        ("0.90", 0.902, 28510, {"routing": 41.94, "search": 31.22, "merge": 0.83, "transfer": 0.078,
                                "protocol": 5.35, "waiting": 20.58}),
        ("0.95", 0.949, 24126, {"routing": 36.15, "search": 38.06, "merge": 0.44, "transfer": 0.050,
                                "protocol": 4.00, "waiting": 21.29}),
        ("0.99", 0.991, 9349,  {"routing": 13.89, "search": 53.45, "merge": 0.36, "transfer": 0.030,
                                "protocol": 1.82, "waiting": 30.45}),
    ]),
    ("Harmony", [
        ("0.80*", 0.788, 2995, {"routing": 54.12, "search": 20.40, "transfer": 1.83, "waiting": 23.65}),
        ("0.90",  0.902, 2524, {"routing": 45.22, "search": 31.50, "transfer": 3.43, "waiting": 19.85}),
        ("0.95",  0.954, 1634, {"routing": 30.34, "search": 40.30, "transfer": 4.28, "waiting": 25.08}),
        ("0.99",  0.991, 671,  {"routing": 12.75, "search": 49.80, "transfer": 5.74, "waiting": 31.71}),
    ]),
    ("DistributedANN*", [
        ("0.80",  0.821, 20070, {"routing": 35.54, "search": 31.12, "merge": 6.02, "transfer": 0.65,
                                 "protocol": 0.014, "waiting": 4.62, "other": 22.03}),
        ("0.90",  0.901, 12469, {"routing": 24.69, "search": 31.92, "merge": 8.73, "transfer": 0.87,
                                 "protocol": 0.009, "waiting": 5.49, "other": 28.30}),
        ("0.93*", 0.928, 5764,  {"routing": 17.06, "search": 35.33, "merge": 11.93, "transfer": 1.14,
                                 "protocol": 0.010, "waiting": 5.76, "other": 28.77}),
    ]),
]
LABEL_MIN = 5.0   # label a segment when it takes at least this share


def ink_on(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return INK if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.28 else "#ffffff"


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK_2,
    })
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(7.0, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [len(runs) for _, runs in SYSTEMS]})
    used = set()
    for ax, (title, runs) in zip(axes, SYSTEMS):
        for x, (target, recall, qps, shares) in enumerate(runs):
            total = sum(shares.values())
            assert abs(total - 100.0) < 0.2, (title, target, total)
            bottom = 0.0
            for c, (_, colour) in CATS.items():
                h = shares.get(c, 0.0)
                if h <= 0:
                    continue
                used.add(c)
                ax.bar(x, h, bottom=bottom, width=0.68, color=colour, edgecolor=SURFACE, linewidth=0.8)
                if h >= LABEL_MIN:
                    ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7, color=ink_on(colour))
                bottom += h
            label = (f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000 else f"{qps:.0f}")
            ax.text(x, 101.5, label, ha="center", va="bottom", fontsize=6.5, color=INK_2)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([t for t, *_ in runs], fontsize=7.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9, pad=13, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of query-phase runtime (%)")
    fig.supxlabel("Recall@10 target (queries/s above each bar)", fontsize=8.5, color=INK, y=0.035)
    handles = [Patch(facecolor=CATS[c][1], edgecolor=SURFACE, label=CATS[c][0]) for c in CATS if c in used]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=0.8)
    out = Path(__file__).resolve().parent
    fig.savefig(out / "breakdown6_by_system.pdf")
    fig.savefig(out / "breakdown6_by_system.png", dpi=250)
    print(f"wrote {out / 'breakdown6_by_system.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
