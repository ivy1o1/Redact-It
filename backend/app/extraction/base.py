"""Text extraction with precise, re-locatable positions.

Every extracted document is a list of *segments*. A segment is a contiguous unit
of searchable text (a PDF page, or a DOCX paragraph) that also carries enough
positional metadata to redact exactly the characters a detection covers:

* PDF segments carry per-word bounding boxes, so a character range maps back to
  concrete rectangles on a concrete page.
* DOCX segments carry per-run character offsets, so a range maps back to the
  specific runs whose text must be rewritten.

Character offsets are relative to the segment's own text, which keeps them valid
regardless of how documents are ordered or filtered later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FileKind = Literal["pdf", "docx"]


@dataclass(slots=True)
class WordBox:
    """A single extracted word and its rectangle on the page (PDF only)."""

    text: str
    char_start: int
    char_end: int
    page: int  # 1-based, matches human page numbering
    x0: float
    y0: float
    x1: float
    y1: float
    block: int = 0
    line: int = 0

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class RunSpan:
    """Character range owned by one DOCX run inside its paragraph."""

    index: int  # position within the paragraph's flattened run list
    char_start: int
    char_end: int


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_index: int
    text: str
    page: int | None = None
    paragraph_path: str | None = None
    words: list[WordBox] = Field(default_factory=list)
    runs: list[RunSpan] = Field(default_factory=list)
    # Character ranges of each visual line inside this segment. A PDF segment is
    # a whole page, so this is what lets an exclusion stop at the end of the
    # line it started on instead of protecting the entire page.
    line_ranges: list[tuple[int, int]] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    document_id: str
    document_name: str
    file_kind: FileKind
    size_bytes: int
    sha256: str
    page_count: int | None = None
    char_count: int = 0
    has_text_layer: bool = False
    scanned: bool = False
    encrypted: bool = False
    truncated: bool = False
    segments: list[Segment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def sample(self, max_chars: int) -> str:
        """Bounded preview of the document, for prompting the local model."""
        if max_chars <= 0:
            return ""
        chunks: list[str] = []
        used = 0
        for segment in self.segments:
            remaining = max_chars - used
            if remaining <= 0:
                break
            text = segment.text.strip()
            if not text:
                continue
            pieces = text[:remaining].splitlines()
            for piece in pieces:
                if used >= max_chars:
                    break
                chunks.append(piece)
                used += len(piece) + 1
        return "\n".join(chunks)[:max_chars]

    def words_in_range(
        self, segment_index: int, char_start: int, char_end: int
    ) -> list[WordBox]:
        """PDF helper: word boxes overlapping a character range."""
        for segment in self.segments:
            if segment.segment_index == segment_index:
                return [w for w in segment.words if w.char_end > char_start and w.char_start < char_end]
        return []
