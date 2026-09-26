"""Split text into overlapping chunks along paragraph and sentence boundaries.

Every chunk is an exact slice of the input (`text[start:end]`), so its position is
known and quotes can later be checked against the source word for word.
"""

import re
from dataclasses import dataclass

TARGET_CHARS = 1000  # aim for chunks about this long
OVERLAP_CHARS = 150  # repeat roughly this much of the previous chunk
MAX_SEGMENT_CHARS = 1500  # split sentences longer than this at a space

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[TextChunk]:
    segments = _segments(text, target_chars)
    chunks: list[TextChunk] = []
    i = 0
    while i < len(segments):
        start = segments[i][0]
        j = i
        while j + 1 < len(segments) and segments[j + 1][1] - start <= target_chars:
            j += 1
        end = segments[j][1]
        chunks.append(TextChunk(text[start:end], start, end))
        if j == len(segments) - 1:
            break

        # Start the next chunk a few sentences back, for continuity; always move forward.
        next_i = j + 1
        for m in range(j, i, -1):
            if end - segments[m][0] <= overlap_chars:
                next_i = m
            else:
                break
        i = next_i
    return chunks


def _segments(text: str, target_chars: int) -> list[tuple[int, int]]:
    """Sentence-sized (start, end) spans with surrounding whitespace trimmed."""
    spans: list[tuple[int, int]] = []
    paragraph_start = 0
    for match in [*_PARAGRAPH_BREAK.finditer(text), None]:
        paragraph_end = match.start() if match else len(text)
        sentence_start = paragraph_start
        for sentence in _SENTENCE_END.finditer(text, paragraph_start, paragraph_end):
            spans.append((sentence_start, sentence.end()))
            sentence_start = sentence.end()
        spans.append((sentence_start, paragraph_end))
        paragraph_start = match.end() if match else len(text)

    segments: list[tuple[int, int]] = []
    for start, end in spans:
        trimmed = _trim(text, start, end)
        if trimmed is None:
            continue
        if trimmed[1] - trimmed[0] > MAX_SEGMENT_CHARS:
            segments.extend(_split_long(text, *trimmed, target_chars))
        else:
            segments.append(trimmed)
    return segments


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_long(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    while end - start > size:
        cut = text.rfind(" ", start + size // 2, start + size)
        if cut <= start:
            cut = start + size  # no space nearby: cut mid-word rather than loop forever
        trimmed = _trim(text, start, cut)
        if trimmed:
            pieces.append(trimmed)
        start = cut
    trimmed = _trim(text, start, end)
    if trimmed:
        pieces.append(trimmed)
    return pieces
