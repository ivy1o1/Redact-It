"""PDF text extraction via PyMuPDF.

Scanned / image-only PDFs are detected and reported explicitly rather than
silently producing an empty result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from ..errors import (
    EmptyDocumentError,
    EncryptedDocumentError,
    ExtractionBudgetError,
    ExtractionError,
    ScannedDocumentError,
)
from ..logging_setup import get_logger
from .base import ExtractedDocument, Segment, WordBox

logger = get_logger("extraction.pdf")

PDF_LIMITATIONS = [
    "Text inside PDF form fields, annotations, comments and attached files is not "
    "extracted or redacted by this release.",
    "Redaction removes the text layer under each matched word. If a page is a scan, "
    "the visible pixels are only blanked where a text-layer match was found.",
    "PDF redaction works at word granularity: a match covering part of a word "
    "removes the whole word, so slightly more text than the match may disappear.",
    "No OCR is performed. Content that exists only as an image cannot be detected.",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_segment(
    page: pymupdf.Page, page_number: int, segment_index: int
) -> Segment | None:
    """Build one segment per page from word-level extraction.

    Words are joined with single spaces. Offsets are recorded per word so any
    character range can be mapped back to concrete rectangles.
    """
    try:
        raw_words = page.get_text("words")
    except Exception as exc:  # damaged page object
        logger.warning("pdf_page_word_extraction_failed", page=page_number, error=str(exc)[:200])
        return None

    if not raw_words:
        return None

    # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    ordered = sorted(raw_words, key=lambda w: (w[5], w[6], w[7]))

    pieces: list[str] = []
    boxes: list[WordBox] = []
    line_ranges: list[tuple[int, int]] = []
    cursor = 0
    line_start = 0
    current_line: tuple[int, int] | None = None

    for item in ordered:
        text = item[4]
        if not text:
            continue

        line_key = (int(item[5]), int(item[6]))
        if current_line is not None and line_key != current_line:
            line_ranges.append((line_start, cursor))
            line_start = cursor
        current_line = line_key

        start = cursor
        pieces.append(text)
        cursor = start + len(text)
        boxes.append(
            WordBox(
                text=text,
                char_start=start,
                char_end=cursor,
                page=page_number,
                x0=float(item[0]),
                y0=float(item[1]),
                x1=float(item[2]),
                y1=float(item[3]),
                block=line_key[0],
                line=line_key[1],
            )
        )
        pieces.append(" ")
        cursor += 1

    joined = "".join(pieces).rstrip()
    if not joined.strip():
        return None

    if current_line is not None:
        line_ranges.append((line_start, min(cursor, len(joined))))

    return Segment(
        segment_index=segment_index,
        text=joined,
        page=page_number,
        paragraph_path=None,
        words=boxes,
        runs=[],
        line_ranges=[(s, e) for s, e in line_ranges if e > s],
    )


def extract_pdf(
    path: Path,
    *,
    document_id: str,
    document_name: str,
    max_chars: int,
) -> ExtractedDocument:
    size_bytes = path.stat().st_size
    warnings: list[str] = []

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ExtractionError(
            f"{document_name!r} could not be opened as a PDF. The file may be "
            f"corrupt or not a real PDF. ({type(exc).__name__})",
            details={"document_name": document_name},
        ) from exc

    with document:
        if document.needs_pass or document.is_encrypted:
            raise EncryptedDocumentError(
                f"{document_name!r} is encrypted or password-protected. RedactIt "
                "does not attempt to open protected PDFs. Remove the password and "
                "re-upload.",
                details={"document_name": document_name},
            )

        page_count = document.page_count
        segments: list[Segment] = []
        image_only_pages: list[int] = []
        total_chars = 0
        truncated = False
        # Set as soon as any page yields words. Distinguishing "no page had text"
        # from "the budget cut us off before we stored anything" matters: calling
        # a text PDF a scan sends the user off to run OCR for no reason.
        saw_text = False

        for index in range(page_count):
            page = document.load_page(index)
            page_number = index + 1

            segment = _page_segment(page, page_number, len(segments))
            if segment is None:
                if page.get_images(full=True):
                    image_only_pages.append(page_number)
                continue

            saw_text = True

            if max_chars and total_chars + len(segment.text) > max_chars:
                if not segments:
                    raise ExtractionBudgetError(
                        document_name, max_chars=max_chars, needed=len(segment.text)
                    )
                truncated = True
                warnings.append(
                    f"Extraction stopped at the configured character budget "
                    f"({max_chars}); later pages were not analysed."
                )
                break

            segments.append(segment)
            total_chars += len(segment.text)

        scanned = not saw_text and page_count > 0

    if scanned:
        # Raised here so nothing downstream can mistake an image-only PDF for a
        # successfully processed document.
        raise ScannedDocumentError(
            document_name,
            details={
                "document_name": document_name,
                "page_count": page_count,
                "image_only_pages": image_only_pages[:50],
                "scanned": True,
            },
        )

    if page_count == 0:
        raise EmptyDocumentError(
            f"{document_name!r} contains no pages.",
            details={"document_name": document_name},
        )

    if image_only_pages:
        warnings.append(
            "These pages have no text layer and cannot be analysed or redacted "
            f"(no OCR): {image_only_pages[:20]}"
            + (" ..." if len(image_only_pages) > 20 else "")
        )
    if truncated:
        warnings.append(
            "Only part of the document was analysed. Increase MAX_EXTRACTED_CHARS "
            "or split the document for a complete review."
        )

    logger.info(
        "pdf_extracted",
        extra={
            "document_id": document_id,
            "pages": page_count,
            "segments": len(segments),
            "chars": total_chars,
            "image_only_pages": len(image_only_pages),
        },
    )

    return ExtractedDocument(
        document_id=document_id,
        document_name=document_name,
        file_kind="pdf",
        size_bytes=size_bytes,
        sha256=_sha256(path),
        page_count=page_count,
        char_count=total_chars,
        has_text_layer=bool(segments),
        scanned=False,
        encrypted=False,
        truncated=truncated,
        segments=segments,
        warnings=warnings,
        limitations=list(PDF_LIMITATIONS),
    )
