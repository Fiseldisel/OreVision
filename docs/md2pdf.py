# -*- coding: utf-8 -*-
"""Конвертация markdown в PDF через headless Edge.

По умолчанию собирает руководство пользователя. Можно указать другой
файл аргументом: ``python md2pdf.py VIDEO_PLAN.md``.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "РУКОВОДСТВО_ПОЛЬЗОВАТЕЛЯ.md"
if not SRC.is_absolute():
    SRC = (HERE / SRC).resolve()
DST = SRC.with_suffix(".pdf")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.45; color: #23272e;
  margin: 0;
}
h1 { font-size: 19pt; color: #23272e; border-bottom: 2px solid #c77b3f;
     padding-bottom: 6px; margin: 0 0 10px 0; }
h2 { font-size: 14pt; color: #9e5f2c; margin: 22px 0 8px 0;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #23272e; margin: 14px 0 6px 0;
     page-break-after: avoid; }
p { margin: 5px 0; }
ul, ol { margin: 5px 0; padding-left: 22px; }
li { margin: 3px 0; }
code {
  font-family: Consolas, "Courier New", monospace; font-size: 9.3pt;
  background: #f4f2ee; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f4f2ee; border: 1px solid #e3ded6; border-radius: 5px;
  padding: 9px 12px; overflow-x: hidden; white-space: pre-wrap;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 8px 0;
  page-break-inside: avoid; font-size: 10pt;
}
th, td { border: 1px solid #d8d4cc; padding: 5px 8px; text-align: left; }
th { background: #f4f2ee; }
hr { border: none; border-top: 1px solid #d8d4cc; margin: 16px 0; }
strong { color: #23272e; }
.footer-note { color: #6b7280; font-size: 9pt; margin-top: 24px; }
"""

md_text = SRC.read_text(encoding="utf-8")
m = re.search(r"^#\s+(.+)$", md_text, flags=re.MULTILINE)
title = m.group(1).strip() if m else SRC.stem
body = markdown.markdown(
    md_text, extensions=["tables", "fenced_code", "sane_lists"]
)
html = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    f"<title>{title}</title><style>{CSS}</style>"
    f"</head><body>{body}"
    "<p class='footer-note'>OreVision · github.com/Fiseldisel/OreVision</p>"
    "</body></html>"
)

tmp = Path(tempfile.mkdtemp()) / "guide.html"
tmp.write_text(html, encoding="utf-8")

subprocess.run(
    [
        EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={DST}", tmp.as_uri(),
    ],
    check=True, timeout=120,
)
print("PDF:", DST, f"{DST.stat().st_size / 1024:.0f} KB")
