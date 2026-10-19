"""PDF redaction with PyMuPDF.

Redaction here means removal, not covering: ``add_redact_annot`` marks the word
rectangles and ``apply_redactions`` deletes the underlying text from the content
stream. The file is then rewritten with ``garbage=4`` and ``clean=True`` so no
orphaned object still carries the removed glyphs, and metadata is dropped.

Rectangles come from the word boxes captured during extraction, so only the
matched words are removed - surrounding text on the same line is preserved.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..extraction.base import ExtractedDocument
from ..logging_setup import get_logger
from ..matching.engine import Detection

logger = get_logger("redaction.pdf")

_IMAGE_PIXELS = getattr(pymupdf, "PDF_REDACT_IMAGE_PIXELS", 2)
_IMAGE_NONE = getattr(pymupdf, "PDF_REDACT_IMAGE_NONE", 0)


@dataclass(slots=True)
class RedactionOutcome:
    document_id: str
    source_name: str
    output_name: str
    output_path: Path
    applied: int = 0
    rectangles: int = 0
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _output_name(source_name: str) -> str:
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix or ".pdf"
    return f"{stem}.redacted{suffix}"


def redact_pdf(
    source: Path,
    destination: Path,
    detections: list[Detection],
    document: ExtractedDocument,
    *,
    redact_images: bool = True,
) -> RedactionOutcome:
    outcome = RedactionOutcome(
        document_id=document.document_id,
        source_name=document.document_name,
        output_name=_output_name(document.document_name),
        output_path=destination,
    )

    by_page: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        page = detection.location.page
        if page is None:
            outcome.skipped.append(
                f"{detection.detection_id}: no page recorded for this match"
            )
            continue
        by_page[page].append(detection)

    try:
        pdf = pymupdf.open(source)
    except Exception as exc:
        outcome.warnings.append(
            f"Could not reopen the source PDF ({type(exc).__name__}); no output was produced."
        )
        logger.error("pdf_reopen_failed", extra={"document_id": document.document_id})
        return outcome

    with pdf:
        if pdf.needs_pass or pdf.is_encrypted:
            outcome.warnings.append("Source PDF is encrypted; nothing was redacted.")
            return outcome

        image_mode = _IMAGE_PIXELS if redact_images else _IMAGE_NONE

        for page_number, page_detections in sorted(by_page.items()):
            if not 1 <= page_number <= pdf.page_count:
                outcome.warnings.append(
                    f"Page {page_number} no longer exists in the source PDF; "
                    f"{len(page_detections)} match(es) skipped."
                )
                continue

            page = pdf.load_page(page_number - 1)
            added = 0

            for detection in page_detections:
                boxes = document.words_in_range(
                    detection.location.segment_index,
                    detection.location.char_start,
                    detection.location.char_end,
                )
                if not boxes:
                    outcome.skipped.append(
                        f"{detection.detection_id}: no word geometry found on page "
                        f"{page_number}"
                    )
                    continue

                for box in boxes:
                    rect = pymupdf.Rect(*box.rect)
                    if rect.is_empty or not rect.is_valid:
                        continue
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    added += 1

                outcome.applied += 1

            if added:
                try:
                    page.apply_redactions(images=image_mode)
                except Exception as exc:
                    outcome.warnings.append(
                        f"apply_redactions failed on page {page_number} "
                        f"({type(exc).__name__}); that page may still contain text."
                    )
                    logger.error(
                        "pdf_apply_redactions_failed",
                        extra={"page": page_number, "error_type": type(exc).__name__},
                    )
                outcome.rectangles += added

        # Drop metadata: author/title/subject routinely leak the same identities
        # the document text was redacted to protect.
        try:
            pdf.set_metadata({})
        except Exception:  # pragma: no cover - unusual documents
            outcome.warnings.append("Could not clear PDF metadata.")

        scrub = getattr(pdf, "scrub", None)
        if scrub is not None:
            try:
                scrub()
            except Exception:  # pragma: no cover - older/newer signature drift
                logger.info("pdf_scrub_unavailable")

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            pdf.save(destination, garbage=4, deflate=True, clean=True)
        except Exception as exc:
            outcome.warnings.append(
                f"Saving the redacted PDF failed ({type(exc).__name__}); no output file."
            )
            logger.error("pdf_save_failed", extra={"error_type": type(exc).__name__})
            return outcome

    if outcome.applied == 0 and not outcome.warnings:
        outcome.warnings.append(
            "No redaction rectangles were produced for this PDF. The output is a "
            "copy of the original."
        )

    logger.info(
        "pdf_redacted",
        extra={
            "document_id": document.document_id,
            "applied": outcome.applied,
            "rectangles": outcome.rectangles,
            "skipped": len(outcome.skipped),
        },
    )
    return outcome
