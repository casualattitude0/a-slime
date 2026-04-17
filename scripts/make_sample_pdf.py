#!/usr/bin/env python3
"""Writes a PDF with selectable text under data/ for ingest testing."""

from pathlib import Path

from fpdf import FPDF


DEFAULT_BODY = """\
Project Alpha - internal notes

This sample PDF exists so you can run:

  python src/ingest.py

The RAG pipeline extracts text from PDFs using pypdf. Use real PDFs with
embedded text (exported from Word, Google Docs, or printed from Markdown).
Scanned pages need OCR first; image-only PDFs usually return no text.

Key facts for testing retrieval:
- Project code name: Alpha
- Owner team: Platform
- SLA target: 99.9% monthly uptime
"""


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "sample.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_margins(20, 20, 20)
    for line in DEFAULT_BODY.strip().split("\n"):
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
