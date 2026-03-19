from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from weasyprint import HTML


def render_one(md_path: Path, css_path: Path) -> Path:
    output_pdf = md_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_html = Path(tmpdir) / f"{md_path.stem}.html"
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-f",
                "gfm",
                "-t",
                "html5",
                "-s",
                "--toc",
                "--metadata",
                f"title={md_path.stem.replace('_', ' ')}",
                "--css",
                str(css_path),
                "-o",
                str(tmp_html),
            ],
            check=True,
        )
        HTML(filename=str(tmp_html), base_url=str(md_path.parent)).write_pdf(str(output_pdf))
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown files to high-quality PDFs.")
    parser.add_argument("markdown_files", nargs="+", type=Path, help="Markdown files to render")
    parser.add_argument(
        "--css",
        type=Path,
        default=Path("docs/pdf_style.css"),
        help="Path to CSS file for PDF styling",
    )
    args = parser.parse_args()

    css_path = args.css.resolve()
    if not css_path.exists():
        raise FileNotFoundError(f"CSS not found: {css_path}")

    for md in args.markdown_files:
        md_path = md.resolve()
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        pdf = render_one(md_path, css_path)
        print(f"Rendered: {pdf}")


if __name__ == "__main__":
    main()
