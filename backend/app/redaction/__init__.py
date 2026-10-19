"""Redaction engines.

Both engines write a *new* file and never modify the uploaded original.
"""

from __future__ import annotations

from pathlib import Path

from ..extraction.base import ExtractedDocument
from ..matching.engine import Detection
from .docx import RedactionOutcome as DocxOutcome
from .docx import redact_docx
from .pdf import RedactionOutcome as PdfOutcome
from .pdf import redact_pdf
from .verify import VerificationReport, verify_redaction

RedactionOutcome = PdfOutcome | DocxOutcome


def redact_document(
    source: Path,
    destination: Path,
    detections: list[Detection],
    document: ExtractedDocument,
    *,
    replacement_text: str = "[REDACTED]",
    redact_images: bool = True,
) -> RedactionOutcome:
    """Apply approved detections to a copy of ``source``."""
    if document.file_kind == "pdf":
        return redact_pdf(
            source, destination, detections, document, redact_images=redact_images
        )
    return redact_docx(
        source, destination, detections, document, replacement_text=replacement_text
    )


def output_name_for(document_name: str, file_kind: str) -> str:
    suffix = Path(document_name).suffix or f".{file_kind}"
    return f"{Path(document_name).stem}.redacted{suffix}"


__all__ = [
    "DocxOutcome",
    "PdfOutcome",
    "RedactionOutcome",
    "VerificationReport",
    "output_name_for",
    "redact_document",
    "redact_docx",
    "redact_pdf",
    "verify_redaction",
]
