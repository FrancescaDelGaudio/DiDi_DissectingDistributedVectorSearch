#!/usr/bin/env python3
"""PDF report: full per-ef metrics from CoTra 100M 8p sweep logs (jobs 6911861–6911864)."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from parse_100m_8p_log import (
    EXPERIMENTS_100M_8P,
    LOG_DIR,
    LogMeta,
    EfRow,
    all_efs,
    parse_log,
)

OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_PDF = OUT_DIR / "cotra_100M_8p_sweep_jobs_6911861-6911864.pdf"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def fmt_f(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def fmt_i(x: int | None) -> str:
    if x is None:
        return "-"
    return f"{x:,}"


class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def load_all() -> tuple[list[tuple[str, str, LogMeta, dict[int, EfRow]]], list[int]]:
    exps: list[tuple[str, str, LogMeta, dict[int, EfRow]]] = []
    data: dict[str, dict[int, EfRow]] = {}
    for label, job, fname in EXPERIMENTS_100M_8P:
        meta, rows = parse_log(LOG_DIR / fname)
        exps.append((label, job, meta, rows))
        data[label] = rows
    return exps, all_efs(data)


def add_meta_block(pdf: PDF, label: str, job: str, meta: LogMeta, rows: dict[int, EfRow]):
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, label)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Slurm job {job}  |  log: {LOG_DIR.name}/scala_anns_8p-*-{job}.out", ln=True)
    pdf.cell(
        0,
        5,
        f"dataset={meta.dataset}  dist_fn={meta.dist_fn}  res_knn={meta.res_knn}  "
        f"ef points={len(rows)}  start={meta.start or '?'}  done={meta.done or '(incomplete)'}",
        ln=True,
    )
    if rows:
        recalls = [r.recall_max for r in rows.values() if r.recall_max is not None]
        if recalls:
            pdf.cell(
                0,
                5,
                f"recall@k range (max per ef): {min(recalls):.4f} .. {max(recalls):.4f}",
                ln=True,
            )
    pdf.ln(2)


def add_detail_table(pdf: PDF, rows: dict[int, EfRow], efs: list[int]):
    # Landscape A4 ~297mm; 12 columns
    col_w = [11, 16, 16, 16, 22, 22, 26, 26, 22, 22, 18, 18]
    headers = [
        "ef",
        "recall",
        "lat_us",
        "QPS",
        "writes",
        "reads",
        "write_B",
        "read_B",
        "w(h)",
        "r(h)",
        "core_sum",
        "orig/n",
    ]
    pdf.set_font("Helvetica", "B", 6)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 5, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 6)
    for ef in efs:
        if ef not in rows:
            continue
        r = rows[ef]
        cells = [
            str(ef),
            fmt_f(r.recall_max, 5),
            fmt_f(r.lat_avg, 2),
            fmt_f(r.qps_avg, 1),
            fmt_i(r.writes),
            fmt_i(r.reads),
            fmt_i(r.write_bytes),
            fmt_i(r.read_bytes),
            fmt_bytes(r.write_bytes),
            fmt_bytes(r.read_bytes),
            fmt_i(r.core_sum),
            fmt_i(r.origin_per_node),
        ]
        for i, (text, width) in enumerate(zip(cells, col_w)):
            align = "C" if i == 0 else "R"
            pdf.cell(width, 4.5, text, border=1, align=align)
        pdf.ln()


def add_core_load_page(pdf: PDF, label: str, rows: dict[int, EfRow], efs: list[int]):
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"Core load per machine (m0..m7) - {label}", ln=True)
    pdf.ln(1)
    col_w = [10] + [12] * 8 + [14]
    headers = ["ef"] + [f"m{i}" for i in range(8)] + ["sum"]
    pdf.set_font("Helvetica", "B", 6)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 5, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 6)
    for ef in efs:
        if ef not in rows or not rows[ef].core_m:
            continue
        r = rows[ef]
        cells = [str(ef)] + [str(x) for x in r.core_m] + [str(r.core_sum)]
        for i, (text, width) in enumerate(zip(cells, col_w)):
            pdf.cell(width, 4.5, text, border=1, align="C" if i == 0 else "R")
        pdf.ln()
    pdf.ln(3)


def add_summary_matrix(pdf: PDF, exps, efs: list[int]):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary @ ef=700 (or max ef if missing)", ln=True)
    pdf.ln(2)
    col_w = [58, 14, 14, 14, 22, 22, 28, 28]
    headers = ["Experiment", "ef", "recall", "QPS", "writes", "reads", "write_B", "read_B"]
    pdf.set_font("Helvetica", "B", 7)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for label, job, _meta, rows in exps:
        if not rows:
            continue
        ef_pick = 700 if 700 in rows else max(rows.keys())
        r = rows[ef_pick]
        note = "" if ef_pick == 700 else f" (max ef={ef_pick})"
        pdf.cell(col_w[0], 5, (label + note)[:52], border=1)
        pdf.cell(col_w[1], 5, str(ef_pick), border=1, align="C")
        pdf.cell(col_w[2], 5, fmt_f(r.recall_max, 5), border=1, align="R")
        pdf.cell(col_w[3], 5, fmt_f(r.qps_avg, 1), border=1, align="R")
        pdf.cell(col_w[4], 5, fmt_i(r.writes), border=1, align="R")
        pdf.cell(col_w[5], 5, fmt_i(r.reads), border=1, align="R")
        pdf.cell(col_w[6], 5, fmt_bytes(r.write_bytes), border=1, align="R")
        pdf.cell(col_w[7], 5, fmt_bytes(r.read_bytes), border=1, align="R")
        pdf.ln()


def main():
    exps, efs = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "CoTra 100M - 8 partitions query sweep (full log metrics)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "Sources: leader stdout from jobs 6911861 (DEEP k=100), 6911862 (DEEP k=10), "
        "6911863 (TTI k=100), 6911864 (TTI k=10). TTI runs use COTRA_EF_SKIP=20 (20 ef values). "
        "Job 6911864 failed during ef=700 (tcmalloc); table ends at ef=600. "
        "recall/lat/QPS: mean over repeated measurement lines per ef in the log; "
        "recall column shows max recall among those lines (typical leader report). "
        "comm_sum: cluster-total RDMA writes/reads/bytes (incl. TERMINATION write adjustment). "
        "core_load: core participations summed over 8 nodes; origin_per_node=1250 expected.",
    )
    pdf.ln(4)

    for label, job, meta, rows in exps:
        pdf.add_page()
        add_meta_block(pdf, label, job, meta, rows)
        add_detail_table(pdf, rows, efs)

    for label, _job, _meta, rows in exps:
        if not any(r.core_m for r in rows.values()):
            continue
        pdf.add_page()
        add_core_load_page(pdf, label, rows, efs)

    add_summary_matrix(pdf, exps, efs)

    pdf.output(str(OUT_PDF))
    print(OUT_PDF)


if __name__ == "__main__":
    main()
