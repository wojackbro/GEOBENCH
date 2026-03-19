from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def wrap_text(text: str, c: canvas.Canvas, max_width: float, font_name: str, font_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if c.stringWidth(probe, font_name, font_size) <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    width, height = A4
    margin = 50
    max_width = width - 2 * margin
    y = height - margin

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    # Use built-in Helvetica to avoid external font dependencies.
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    mono_font = "Courier"

    line_height = 14
    header_gap = 8

    content = md_path.read_text(encoding="utf-8").splitlines()

    def new_page_if_needed(extra: int = 0) -> None:
        nonlocal y
        if y - extra < margin:
            c.showPage()
            y = height - margin

    for raw in content:
        line = raw.rstrip("\n")

        if line.startswith("```"):
            continue

        if line.startswith("# "):
            text = line[2:].strip()
            font, size = bold_font, 18
            new_page_if_needed(26)
            c.setFont(font, size)
            c.drawString(margin, y, text)
            y -= 26
            continue

        if line.startswith("## "):
            text = line[3:].strip()
            font, size = bold_font, 14
            new_page_if_needed(20)
            c.setFont(font, size)
            c.drawString(margin, y, text)
            y -= 20
            continue

        if line.startswith("### "):
            text = line[4:].strip()
            font, size = bold_font, 12
            new_page_if_needed(18)
            c.setFont(font, size)
            c.drawString(margin, y, text)
            y -= 18
            continue

        is_bullet = line.strip().startswith("- ")
        is_ordered = line.strip().startswith(tuple(f"{i}. " for i in range(1, 10)))

        if is_bullet:
            text = line.strip()[2:].strip()
            prefix = "- "
        elif is_ordered:
            parts = line.strip().split(" ", 1)
            prefix = parts[0] + " "
            text = parts[1] if len(parts) > 1 else ""
        else:
            prefix = ""
            text = line

        if not text.strip():
            y -= header_gap
            new_page_if_needed()
            continue

        if line.startswith("    ") or line.startswith("\t"):
            font, size = mono_font, 9
        else:
            font, size = body_font, 10

        c.setFont(font, size)
        wrapped = wrap_text(text, c, max_width - (14 if prefix else 0), font, size)
        for idx, w in enumerate(wrapped):
            new_page_if_needed(line_height)
            if idx == 0 and prefix:
                c.drawString(margin, y, prefix + w)
            elif prefix:
                c.drawString(margin + 14, y, w)
            else:
                c.drawString(margin, y, w)
            y -= line_height

    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert markdown file to simple PDF.")
    parser.add_argument("input", type=Path, help="Input markdown file")
    parser.add_argument("output", type=Path, help="Output PDF file")
    args = parser.parse_args()
    draw_markdown_to_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
