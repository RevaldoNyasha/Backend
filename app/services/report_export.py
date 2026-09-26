"""Turn a finished report (Markdown) into a downloadable PDF or DOCX (§8.5).

Both use libraries already in the backend: python-docx for Word and PyMuPDF's
HTML layout (`Story`) for PDF. The Markdown is parsed with markdown-it with raw
HTML disabled, so model output can never inject markup into the files.
"""

import html
import io
import re
from dataclasses import dataclass
from datetime import datetime

import pymupdf
from docx import Document as DocxDocument
from docx.shared import Pt
from markdown_it import MarkdownIt
from markdown_it.token import Token

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MEDIA_TYPE = "application/pdf"

_markdown = MarkdownIt("commonmark", {"html": False})

PDF_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1c1c1a; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt 0; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 0; }
li { margin: 0 0 3pt 0; }
code { font-family: monospace; }
.meta { color: #6b6b66; font-size: 9pt; margin-bottom: 14pt; }
"""


@dataclass(frozen=True)
class ReportDocument:
    title: str
    content: str  # Markdown with `## ` sections
    project_name: str
    template_label: str
    created_at: datetime
    documents: list[str]

    @property
    def meta_line(self) -> str:
        count = len(self.documents)
        plural = "document" if count == 1 else "documents"
        return (
            f"{self.template_label} · {self.project_name} · "
            f"{self.created_at:%d %B %Y} · Based on {count} {plural}: "
            + ", ".join(self.documents)
        )


def filename_for(report: ReportDocument, extension: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", report.title).strip().replace(" ", "_")[:80] or "report"
    return f"{stem}.{extension}"


# ---------------------------------------------------------------- PDF


def to_pdf(report: ReportDocument) -> bytes:
    body = (
        f"<h1>{html.escape(report.title)}</h1>"
        f'<p class="meta">{html.escape(report.meta_line)}</p>'
        + _markdown.render(report.content)
    )
    story = pymupdf.Story(html=body, user_css=PDF_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page = pymupdf.paper_rect("a4")
    area = page + (56, 56, -56, -56)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


# ---------------------------------------------------------------- DOCX


def to_docx(report: ReportDocument) -> bytes:
    document = DocxDocument()
    document.styles["Normal"].font.size = Pt(10.5)
    document.add_heading(report.title, level=0)
    meta = document.add_paragraph(report.meta_line)
    for run in meta.runs:
        run.italic = True
        run.font.size = Pt(9)

    list_styles: list[str] = []
    paragraph_style: str | None = None
    tokens = _markdown.parse(report.content)
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            level = min(int(token.tag[1]), 3)
            heading = document.add_heading(level=max(1, level - 1))
            _add_inline(heading, tokens[index + 1])
        elif token.type == "bullet_list_open":
            list_styles.append("List Bullet")
        elif token.type == "ordered_list_open":
            list_styles.append("List Number")
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_styles.pop()
        elif token.type == "list_item_open":
            paragraph_style = list_styles[-1] if list_styles else None
        elif token.type == "paragraph_open":
            paragraph = document.add_paragraph(style=paragraph_style)
            paragraph_style = None if not list_styles else paragraph_style
            _add_inline(paragraph, tokens[index + 1])
        elif token.type in ("fence", "code_block"):
            paragraph = document.add_paragraph()
            run = paragraph.add_run(token.content.rstrip())
            run.font.name = "Consolas"
        elif token.type == "list_item_close":
            paragraph_style = None

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _add_inline(paragraph, inline: Token) -> None:
    """Add an inline token's text to a Word paragraph, keeping bold/italic/code."""
    bold = italic = False
    for child in inline.children or []:
        if child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type in ("text", "code_inline") and child.content:
            run = paragraph.add_run(child.content)
            run.bold = bold
            run.italic = italic
            if child.type == "code_inline":
                run.font.name = "Consolas"
        elif child.type in ("softbreak", "hardbreak"):
            paragraph.add_run(" " if child.type == "softbreak" else "\n")
