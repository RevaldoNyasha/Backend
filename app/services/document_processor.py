"""Turn an uploaded document into clean text (PDF, DOCX, TXT, CSV).

Audio and video are not handled here; they become text through transcription.
Text is returned per page (PDFs) or as one page (everything else) so later
steps can cite "page 4".
"""

import csv
import io
import re
import unicodedata
from dataclasses import dataclass

import pymupdf
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.models.document import KIND_CSV, KIND_DOCX, KIND_NOTES, KIND_PDF

EXTRACTABLE_KINDS = frozenset({KIND_PDF, KIND_DOCX, KIND_NOTES, KIND_CSV})


class ExtractionError(Exception):
    """The file could not be turned into text; the message is shown to the user."""


@dataclass(frozen=True)
class ExtractedPage:
    number: int | None  # 1-based PDF page; None for formats without pages
    text: str


@dataclass(frozen=True)
class ExtractedText:
    pages: list[ExtractedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def extract_text(kind: str, content: bytes) -> ExtractedText:
    if kind == KIND_PDF:
        pages = _extract_pdf(content)
    elif kind == KIND_DOCX:
        pages = [ExtractedPage(None, _extract_docx(content))]
    elif kind == KIND_NOTES:
        pages = [ExtractedPage(None, _decode(content))]
    elif kind == KIND_CSV:
        pages = [ExtractedPage(None, _extract_csv(content))]
    else:
        raise ExtractionError(f"Text extraction is not supported for {kind} files")

    cleaned = [ExtractedPage(page.number, clean_text(page.text)) for page in pages]
    if not any(page.text for page in cleaned):
        if kind == KIND_PDF:
            raise ExtractionError(
                "No text found in this PDF. Scanned (image-only) PDFs are not supported yet."
            )
        raise ExtractionError("No text found in this file")
    return ExtractedText(cleaned)


def _extract_pdf(content: bytes) -> list[ExtractedPage]:
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as error:
        raise ExtractionError("This file is not a readable PDF") from error
    with pdf:
        if pdf.needs_pass:
            raise ExtractionError("This PDF is password-protected")
        return [
            ExtractedPage(index + 1, page.get_text("text", sort=True))
            for index, page in enumerate(pdf)
        ]


def _extract_docx(content: bytes) -> str:
    try:
        docx = DocxDocument(io.BytesIO(content))
    except Exception as error:
        raise ExtractionError("This file is not a readable DOCX document") from error

    # Walk the body in order so tables stay where they appear in the document.
    blocks: list[str] = []
    for element in docx.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            blocks.append(Paragraph(element, docx).text)
        elif tag == "tbl":
            for row in Table(element, docx).rows:
                cells = _unique_cells(cell.text.strip() for cell in row.cells)
                if cells:
                    blocks.append(" | ".join(cells))
    return "\n\n".join(blocks)


def _unique_cells(cells) -> list[str]:
    # Merged cells repeat their text in every cell they span; keep one copy.
    result: list[str] = []
    for cell in cells:
        if cell and (not result or result[-1] != cell):
            result.append(cell)
    return result


def _extract_csv(content: bytes) -> str:
    text = _decode(content)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in row)]
    if not rows:
        return ""

    # One line per row as "Header: value", which reads well for survey answers.
    header, *records = rows
    if not records:
        return " | ".join(cell.strip() for cell in header)
    lines = []
    for record in records:
        pairs = [
            f"{name.strip() or f'Column {index + 1}'}: {value.strip()}"
            for index, (name, value) in enumerate(zip(header, record, strict=False))
            if value.strip()
        ]
        lines.append(" | ".join(pairs))
    return "\n\n".join(lines)


def _decode(content: bytes) -> str:
    # UTF-16 only with a byte-order mark: decoding arbitrary bytes as UTF-16
    # "succeeds" and produces garbage.
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")  # older Windows text files


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_SPACE = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "transpor-\ntation" -> "transportation"
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
