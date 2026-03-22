#!/usr/bin/env python3
"""
Render docs/Report_for_writing.md to PDF using pandoc (→ HTML) + WeasyPrint + pdf_style.css.

The manuscript includes the full paper shell and the seed-42 quantitative tables (merged).

Usage (from repo root):
  ./.venv-pdf/bin/python docs/render_report_pdf.py

Output:
  docs/Report_for_writing.pdf

Requires: pandoc on PATH; Python package weasyprint (see .venv-pdf).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent


def md_to_html(md_path: Path, html_path: Path, title: str) -> None:
    subprocess.run(
        [
            "pandoc",
            str(md_path),
            "-f",
            "markdown",
            "-t",
            "html",
            "-s",
            "--metadata",
            f"title={title}",
            "-o",
            str(html_path),
        ],
        check=True,
        cwd=str(DOCS),
    )


def html_to_pdf(html_path: Path, pdf_path: Path, css_path: Path) -> None:
    from weasyprint import HTML

    HTML(filename=str(html_path), base_url=str(DOCS)).write_pdf(
        str(pdf_path),
        stylesheets=[css_path],
    )


def main() -> None:
    css = DOCS / "pdf_style.css"
    if not css.is_file():
        sys.exit(f"Missing {css}")

    md_path = DOCS / "Report_for_writing.md"
    pdf_path = DOCS / "Report_for_writing.pdf"
    if not md_path.is_file():
        sys.exit(f"Missing {md_path}")

    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, dir=str(DOCS)
    ) as tmp:
        html_path = Path(tmp.name)
    try:
        md_to_html(md_path, html_path, "Report for writing")
        html_to_pdf(html_path, pdf_path, css)
        print(f"Wrote {pdf_path}")
    finally:
        html_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
