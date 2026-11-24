#!/usr/bin/env python3
"""Build PDF tables of cluster comm_sum bytes from 10M sweep logs (job 6911247–6911250)."""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

from extract_comm_bytes import EXPERIMENTS, LOG_DIR, parse_log

OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_PDF = OUT_DIR / "cotra_10M_comm_bytes_jobs_6911247-6911250.pdf"

PAT = re.compile(
    r"^(\d+)\tcomm_sum\twrites=(\d+)\treads=(\d+)\t"
    r"write_bytes=(\d+)\tread_bytes=(\d+)\s*$"
)


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def load_all() -> tuple[list[int], dict[str, dict[int, tuple]]]:
    all_efs: set[int] = set()
    data: dict[str, dict[int, tuple]] = {}
    for label, _job, fname in EXPERIMENTS:
        rows = parse_log(LOG_DIR / fname)
        data[label] = rows
        all_efs.update(rows.keys())
    return sorted(all_efs), data


class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def add_experiment_section(pdf: PDF, title: str, job: str, rows: dict[int, tuple], efs: list[int]):
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, title)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Slurm job {job}  |  comm_sum (cluster total, incl. TERMINATION write adj.)", ln=True)
    pdf.ln(2)

    col_w = [14, 22, 22, 32, 32, 32, 32]
    headers = ["ef", "writes", "reads", "write_B", "read_B", "write (human)", "read (human)"]
    pdf.set_font("Helvetica", "B", 7)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for ef in efs:
        if ef not in rows:
            continue
        w, r, wb, rb = rows[ef]
        cells = [
            str(ef),
            f"{w:,}",
            f"{r:,}",
            f"{wb:,}",
            f"{rb:,}",
            fmt_bytes(wb),
            fmt_bytes(rb),
        ]
        for i, (text, width) in enumerate(zip(cells, col_w)):
            align = "C" if i == 0 else "R"
            pdf.cell(width, 5, text, border=1, align=align)
        pdf.ln()
    pdf.ln(4)


def add_summary_table(pdf: PDF, efs: list[int], data: dict[str, dict[int, tuple]]):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary: write_bytes @ ef=700 (comm_sum)", ln=True)
    pdf.ln(2)
    col_w = [52, 28, 28, 38, 38]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(["Experiment", "writes", "reads", "write_bytes", "read_bytes"], col_w):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for label, job, _ in EXPERIMENTS:
        rows = data[label]
        if 700 not in rows:
            ef_pick = max(rows.keys()) if rows else None
        else:
            ef_pick = 700
        if ef_pick is None:
            continue
        w, r, wb, rb = rows[ef_pick]
        pdf.cell(col_w[0], 6, label[:40], border=1)
        pdf.cell(col_w[1], 6, f"{w:,}", border=1, align="R")
        pdf.cell(col_w[2], 6, f"{r:,}", border=1, align="R")
        pdf.cell(col_w[3], 6, fmt_bytes(wb), border=1, align="R")
        pdf.cell(col_w[4], 6, fmt_bytes(rb), border=1, align="R")
        pdf.ln()


def main():
    efs, data = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "CoTra 10M - RDMA comm_sum (writes/reads/bytes per ef)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "Source: leader stdout lines 'ef\\tcomm_sum\\twrites=...\\twrite_bytes=...' "
        "from jobs 6911247 (DEEP 4p), 6911248 (TTI 4p, COTRA_EF_SKIP=20), "
        "6911249 (DEEP 6p), 6911250 (TTI 6p). "
        "Byte counts sum RDMA write/read payload sizes per node (thread-local); "
        "write_bytes include estimated TERMINATION control messages.",
    )
    pdf.ln(3)

    for label, job, _ in EXPERIMENTS:
        pdf.add_page()
        add_experiment_section(pdf, label, job, data[label], efs)

    add_summary_table(pdf, efs, data)

    pdf.output(str(OUT_PDF))
    print(OUT_PDF)


if __name__ == "__main__":
    main()
