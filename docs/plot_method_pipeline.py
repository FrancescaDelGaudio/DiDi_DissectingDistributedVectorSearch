#!/usr/bin/env python3
"""Method figure: how the communication time of a distributed ANN system is split into
waiting, protocol and transfer, and how the gain of native RDMA is isolated.

  1. Real run                     routing | local search | merge | communication (C_real)
  2. Communication-only replay    same system and messages, local search replaced by a dummy
                                  result of the same size -> no imbalance, no waiting (C_replay)
  3. Transfer-only replays        the logged messages without the protocol, with the system's
                                  MPI calls (T_MPI) and with native RDMA writes (T_RDMA)
  4. Breakdown                    waiting = C_real - C_replay, protocol = C_replay - T_MPI,
                                  transfer = T_MPI, RDMA gain = T_MPI - T_RDMA
One row of four boxes (two-column figure width). Bar lengths are illustrative, not data.
Colours: the breakdown figures' routing / search / merge / "MPI + waiting"; waiting / protocol /
transfer checked for adjacent-pair separation (normal vision and protan / deutan / tritan
simulations, OKLab dE >= 15 / >= 8).

Usage:
    python3 plot_method_pipeline.py   # writes method_pipeline.png/.pdf next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"
BOX_FILL = "#f7f7f5"

ROUTING = "#2a78d6"
SEARCH = "#1baf7a"
MERGE = "#008300"
COMM = "#e87ba4"       # "MPI + waiting" of the breakdown figures (hatched)
WAIT = "#6250d6"
PROTO = "#eda100"
TRANSFER = "#7a7974"

BAR_H = 3.0
GREY_FS = 7.2    # grey description text
TEXT_FS = 7.2    # black text
MATH_FS = 9.0    # C_real, C_replay, T_MPI, T_RDMA
TITLE_FS = 8.1


def ink_on(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return INK if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.28 else "#ffffff"


def bar(ax, x, y, segments, h=BAR_H, fs=7.0):
    """segments: (width, colour, hatch, label). Returns the x of the right end."""
    for w, colour, hatch, label in segments:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, hatch=hatch, edgecolor=SURFACE, linewidth=1.2))
        if label:
            kw = dict(ha="center", va="center", fontsize=fs, color=ink_on(colour))
            if hatch:
                kw["bbox"] = dict(boxstyle="round,pad=0.12", facecolor=colour, edgecolor="none")
            ax.text(x + w / 2, y + h / 2, label, **kw)
        x += w
    return x


def box(ax, x, y, w, h, number, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=BOX_FILL, edgecolor=INK_2, linewidth=0.7))
    ax.text(x + 1.0, y + h - 1.2, f"{number}  {title}", ha="left", va="top", fontsize=TITLE_FS, color=INK,
            fontweight="bold")
    if lines:
        ax.text(x + 1.0, y + h - 4.6, "\n".join(lines), ha="left", va="top", fontsize=GREY_FS, color=INK_2,
                linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=INK_2, linewidth=0.9, shrinkA=0, shrinkB=0))


def bracket(ax, x0, x1, y, text, fs=MATH_FS):
    ax.plot([x0, x0, x1, x1], [y, y - 0.8, y - 0.8, y], color=INK_2, linewidth=0.7)
    ax.text((x0 + x1) / 2, y - 1.2, text, ha="center", va="top", fontsize=fs, color=INK)


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "hatch.color": "#ffffff", "hatch.linewidth": 0.8})
    fig = plt.figure(figsize=(7.16, 2.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 29.3)
    ax.axis("off")

    y0, h = 0.4, 28.5
    xb = [0.3, 21.8, 45.1, 69.4]       # box left edges
    wb = [19.6, 21.4, 22.4, 30.3]      # box widths
    bar_y = 13.6
    low_y = 5.6                        # black text at the bottom of boxes 1-2

    # ---- 1. real run -------------------------------------------------------------------
    x = xb[0]
    box(ax, x, y0, wb[0], h, "1", "Real run", ["Full system, every phase", "timed on every rank."])
    end = bar(ax, x + 1.0, bar_y, [(3.8, ROUTING, None, "R"), (6.2, SEARCH, None, "Search"),
                                   (1.4, MERGE, None, ""), (6.0, COMM, "////", "")])
    ax.text(x + 1.0 + 3.8 + 6.2 + 0.7, bar_y + BAR_H + 0.4, "merge", ha="center", va="bottom", fontsize=6.4,
            color=INK_2)
    bracket(ax, end - 6.0, end, bar_y - 0.3, "$C_{real}$")
    ax.text(x + 1.0, low_y, "$C_{real}$ = waiting\n+ protocol + transfer", ha="left", va="top",
            fontsize=8, color=INK, linespacing=1.3)

    # ---- 2. communication-only replay --------------------------------------------------
    x = xb[1]
    box(ax, x, y0, wb[1], h, "2", "Comm.-only replay",
        ["Same system and", "messages; local search", "returns a same-size", "dummy result."])
    end2 = bar(ax, x + 1.0, bar_y, [(3.8, ROUTING, None, "R"), (1.4, MERGE, None, ""),
                                    (3.6, PROTO, None, ""), (1.6, TRANSFER, None, "")])
    bracket(ax, end2 - 5.2, end2, bar_y - 0.3, "$C_{replay}$")
    ax.text(end2 + 1.0, bar_y + BAR_H / 2, "no search:\nno waiting", ha="left", va="center", fontsize=GREY_FS,
            color=INK_2, linespacing=1.2)
    ax.text(x + 1.0, low_y, "$C_{replay}$ = protocol\n+ transfer", ha="left", va="top", fontsize=8,
            color=INK, linespacing=1.3)

    # ---- 3. transfer-only replays ------------------------------------------------------
    x = xb[2]
    box(ax, x, y0, wb[2], h, "3", "Transfer-only replays",
        ["Logged messages, no", "protocol: the system's", "MPI calls vs RDMA."])
    ax.text(x + 1.0, 12.4, "MPI", ha="left", va="center", fontsize=TEXT_FS, color=INK)
    bar(ax, x + 6.6, 10.9, [(7.0, TRANSFER, None, "")])
    ax.text(x + 14.4, 12.4, "$T_{MPI}$", ha="left", va="center", fontsize=MATH_FS, color=INK)
    ax.text(x + 1.0, 6.6, "RDMA", ha="left", va="center", fontsize=TEXT_FS, color=INK)
    bar(ax, x + 6.6, 5.1, [(6.2, TRANSFER, "////", "")])
    ax.text(x + 14.4, 6.6, "$T_{RDMA}$", ha="left", va="center", fontsize=MATH_FS, color=INK)
    ax.text(x + 1.0, 2.0, "same messages and nodes", ha="left", va="center", fontsize=6.6, color=INK_2)

    # ---- 4. breakdown of the communication ---------------------------------------------
    x = xb[3]
    box(ax, x, y0, wb[3], h, "4", "Breakdown of $C_{real}$", None)
    widths = (11.8, 10.4, 6.6)
    bx, bby = x + 1.2, 18.6
    bar(ax, bx, bby, [(widths[0], WAIT, None, "Waiting"), (widths[1], PROTO, None, "Protocol"),
                      (widths[2], TRANSFER, None, "Transfer")], fs=6.6)
    xs = [bx, bx + widths[0], bx + widths[0] + widths[1], bx + sum(widths)]
    ax.add_patch(Rectangle((xs[3] - 0.8, bby), 0.8, BAR_H, facecolor="none", edgecolor=INK, linewidth=0.9,
                           linestyle=(0, (2, 1.2))))
    rows = [(WAIT, "Waiting", "$= C_{real} - C_{replay}$"),
            (PROTO, "Protocol", "$= C_{replay} - T_{MPI}$"),
            (TRANSFER, "Transfer", "$= T_{MPI}$"),
            (None, "RDMA gain\non transfer", "$= T_{MPI} - T_{RDMA}$")]
    ry = 14.6
    for colour, name, formula in rows:
        if colour is None:
            ax.add_patch(Rectangle((bx, ry - 0.9), 1.8, 1.8, facecolor="none", edgecolor=INK, linewidth=0.9,
                                   linestyle=(0, (2, 1.2))))
        else:
            ax.add_patch(Rectangle((bx, ry - 0.9), 1.8, 1.8, facecolor=colour, edgecolor="none"))
        ax.text(bx + 2.6, ry, name, ha="left", va="center", fontsize=TEXT_FS, color=INK, linespacing=1.0)
        ax.text(bx + 11.4, ry, formula, ha="left", va="center", fontsize=MATH_FS, color=INK)
        ry -= 3.2

    for i in range(3):
        arrow(ax, xb[i] + wb[i] + 0.15, y0 + h / 2, xb[i + 1] - 0.15, y0 + h / 2)

    out = Path(__file__).resolve().parent
    for ext in ("png", "pdf"):
        fig.savefig(out / f"method_pipeline.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {out / 'method_pipeline.png'} and .pdf")


if __name__ == "__main__":
    main()
