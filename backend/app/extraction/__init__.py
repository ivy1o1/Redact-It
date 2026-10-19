"""Document text extraction."""

from __future__ import annotations

from pathlib import Path

from ..errors import UnsupportedFileTypeError
from .base import ExtractedDocument, RunSpan, Segment, WordBox
from .docx import DOCX_LIMITATIONS, extract_docx, iter_paragraphs, paragraph_runs
from .pdf import PDF_LIMITATIONS, extract_pdf

__all__ = [
    "DOCX_LIMITATIONS",
    "PDF_LIMITATIONS",
    "ExtractedDocument",
    "RunSpan",
    "Segment",
    "WordBox",
    "extract_document",
    "extract_docx",
    "extract_pdf",
    "iter_paragraphs",
    "paragraph_runs",
]


def extract_document(
    path: Path,
    *,
    file_kind: str,
    document_id: str,
    document_name: str,
    max_chars: int,
) -> ExtractedDocument:
    if file_kind == "pdf":
        return extract_pdf(
            path, document_id=document_id, document_name=document_name, max_chars=max_chars
        )
    if file_kind == "docx":
        return extract_docx(
            path, document_id=document_id, document_name=document_name, max_chars=max_chars
        )
    raise UnsupportedFileTypeError(
        f"Unsupported file kind {file_kind!r}. RedactIt accepts .pdf and .docx only.",
        details={"file_kind": file_kind},
    )
