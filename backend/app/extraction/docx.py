"""DOCX text extraction via python-docx.

Paragraphs are addressed by a stable *path* (for example
``body:p3`` or ``body:tbl0/r1c2:p0``) so the redaction pass can re-locate the
exact paragraph and rewrite only the runs a detection covers. Body text, tables
(including nested tables) and section headers/footers are walked.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import docx
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..errors import EmptyDocumentError, ExtractionBudgetError, ExtractionError
from ..logging_setup import get_logger
from .base import ExtractedDocument, RunSpan, Segment

logger = get_logger("extraction.docx")

DOCX_LIMITATIONS = [
    "Text inside floating text boxes, Word comments, footnotes, endnotes and "
    "tracked changes is reported as a warning but is not redacted by this release.",
    "Field codes are redacted by their cached display text only; the field "
    "instruction itself is not rewritten.",
    "Redaction rewrites run text in place, so formatting of untouched text is "
    "preserved but the original characters are not recoverable.",
]

_HEADER_FOOTER_ATTRS = (
    "header",
    "footer",
    "first_page_header",
    "first_page_footer",
    "even_page_header",
    "even_page_footer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paragraph_runs(paragraph: Paragraph) -> list[Any]:
    """Flattened runs of a paragraph, including runs nested in hyperlinks.

    ``paragraph.runs`` omits hyperlink runs, which would leave redactable text
    behind, so inner content is walked explicitly.
    """
    runs: list[Any] = []
    try:
        items = list(paragraph.iter_inner_content())
    except AttributeError:  # pragma: no cover - older python-docx
        return list(paragraph.runs)

    for item in items:
        nested = getattr(item, "runs", None)
        if nested is not None and not hasattr(item, "text"):
            runs.extend(nested)
        elif hasattr(item, "text"):
            runs.append(item)
    return runs


def _walk_container(
    container: Any, prefix: str, seen: set[Any]
) -> Iterator[tuple[str, Paragraph]]:
    for index, paragraph in enumerate(container.paragraphs):
        yield f"{prefix}:p{index}", paragraph

    for table_index, table in enumerate(container.tables):
        yield from _walk_table(table, f"{prefix}:tbl{table_index}", seen)


def _walk_table(table: Table, prefix: str, seen: set[Any]) -> Iterator[tuple[str, Paragraph]]:
    for row_index, row in enumerate(table.rows):
        for cell_index, cell in enumerate(row.cells):
            # Merged cells are returned repeatedly by python-docx; visiting the
            # same underlying element twice would produce stale offsets later.
            # The element itself is the key: holding it in `seen` keeps the lxml
            # proxy alive, whereas id() would be reused once a proxy is collected.
            element = cell._tc
            if element in seen:
                continue
            seen.add(element)
            yield from _walk_container(cell, f"{prefix}/r{row_index}c{cell_index}", seen)


def iter_paragraphs(document: DocumentObject) -> Iterator[tuple[str, Paragraph]]:
    """Yield ``(path, paragraph)`` for every addressable paragraph."""
    seen: set[Any] = set()
    yield from _walk_container(document, "body", seen)

    for section_index, section in enumerate(document.sections):
        for attr in _HEADER_FOOTER_ATTRS:
            part = getattr(section, attr, None)
            if part is None:
                continue
            try:
                yield from _walk_container(part, f"sec{section_index}.{attr}", seen)
            except Exception:  # pragma: no cover - malformed section part
                logger.warning("docx_section_walk_failed", section=section_index, part=attr)


def _uncovered_text_markers(document: DocumentObject) -> list[str]:
    """Detect Word constructs whose text this release cannot redact."""
    found: list[str] = []
    try:
        body_xml = document.element.body.xml
    except Exception:  # pragma: no cover
        return found

    # Matched with a regex rather than a literal because a start tag is followed
    # by either a space (Word always writes w:id/w:author on w:ins) or by ">", and
    # missing the second form would silently under-report.
    markers = {
        "floating text boxes": re.compile(r"txbxContent"),
        "tracked-change insertions": re.compile(r"<w:ins[\s>/]"),
        "tracked-change deletions": re.compile(r"<w:del[\s>/]"),
        "field codes": re.compile(r"fldSimple|w:instrText"),
    }
    for label, needle in markers.items():
        if needle.search(body_xml):
            found.append(label)

    package = getattr(document.part, "package", None)
    if package is not None:
        for part in getattr(package, "iter_parts", lambda: [])():
            name = str(getattr(part, "partname", ""))
            if name.endswith("/footnotes.xml") and "footnotes" not in found:
                found.append("footnotes")
            elif name.endswith("/endnotes.xml") and "endnotes" not in found:
                found.append("endnotes")
            elif name.endswith("/comments.xml") and "comments" not in found:
                found.append("comments")
    return found


def extract_docx(
    path: Path,
    *,
    document_id: str,
    document_name: str,
    max_chars: int,
) -> ExtractedDocument:
    size_bytes = path.stat().st_size
    warnings: list[str] = []

    if not zipfile.is_zipfile(path):
        raise ExtractionError(
            f"{document_name!r} is not a valid .docx package.",
            details={"document_name": document_name},
        )

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(
            f"{document_name!r} could not be opened as a DOCX document. "
            f"The file may be corrupt. ({type(exc).__name__})",
            details={"document_name": document_name},
        ) from exc

    segments: list[Segment] = []
    total_chars = 0
    truncated = False
    mismatched: list[str] = []

    for paragraph_path, paragraph in iter_paragraphs(document):
        runs = paragraph_runs(paragraph)
        pieces: list[str] = []
        run_spans: list[RunSpan] = []
        cursor = 0
        for run_index, run in enumerate(runs):
            try:
                text = run.text or ""
            except Exception:  # pragma: no cover
                continue
            if not text:
                continue
            pieces.append(text)
            run_spans.append(
                RunSpan(index=run_index, char_start=cursor, char_end=cursor + len(text))
            )
            cursor += len(text)

        text = "".join(pieces)
        if not text.strip():
            continue

        if text != paragraph.text:
            mismatched.append(paragraph_path)

        if max_chars and total_chars + len(text) > max_chars:
            if not segments:
                raise ExtractionBudgetError(
                    document_name, max_chars=max_chars, needed=len(text)
                )
            truncated = True
            warnings.append(
                f"Extraction stopped at the configured character budget "
                f"({max_chars}); the rest of the document was not analysed."
            )
            break

        segments.append(
            Segment(
                segment_index=len(segments),
                text=text,
                page=None,
                paragraph_path=paragraph_path,
                words=[],
                runs=run_spans,
                line_ranges=[(0, len(text))],
            )
        )
        total_chars += len(text)

    if not segments:
        raise EmptyDocumentError(
            f"{document_name!r} contains no readable paragraph text. If the "
            "content lives only in text boxes or images, RedactIt cannot analyse it.",
            details={"document_name": document_name},
        )

    if mismatched:
        warnings.append(
            "Some paragraph text is stored outside normal runs (field codes, "
            "tracked changes or smart tags) and may not be fully redacted: "
            f"{mismatched[:10]}" + (" ..." if len(mismatched) > 10 else "")
        )

    uncovered = _uncovered_text_markers(document)
    if uncovered:
        warnings.append(
            "This document contains " + ", ".join(uncovered) + ". Text there is "
            "not redacted by this release - review the output manually."
        )
    if truncated:
        warnings.append(
            "Only part of the document was analysed. Increase MAX_EXTRACTED_CHARS "
            "or split the document for a complete review."
        )

    logger.info(
        "docx_extracted",
        extra={
            "document_id": document_id,
            "segments": len(segments),
            "chars": total_chars,
            "uncovered_constructs": len(uncovered),
        },
    )

    return ExtractedDocument(
        document_id=document_id,
        document_name=document_name,
        file_kind="docx",
        size_bytes=size_bytes,
        sha256=_sha256(path),
        page_count=None,
        char_count=total_chars,
        has_text_layer=True,
        scanned=False,
        encrypted=False,
        truncated=truncated,
        segments=segments,
        warnings=warnings,
        limitations=list(DOCX_LIMITATIONS),
    )
