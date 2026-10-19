"""DOCX redaction with python-docx.

Text is rewritten inside the runs that own it, so the characters are gone from
``word/document.xml`` rather than merely hidden. Runs outside a matched range are
untouched, which preserves the surrounding formatting. Every occurrence in every
addressed paragraph is handled, because each detection carries its own range.

The replacement marker is inserted once per detection, at the first run the
detection touches; the remaining runs simply lose those characters.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph

from ..extraction.base import ExtractedDocument
from ..extraction.docx import iter_paragraphs, paragraph_runs
from ..logging_setup import get_logger
from ..matching.engine import Detection

logger = get_logger("redaction.docx")


@dataclass(slots=True)
class RedactionOutcome:
    document_id: str
    source_name: str
    output_name: str
    output_path: Path
    applied: int = 0
    paragraphs_touched: int = 0
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _output_name(source_name: str) -> str:
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix or ".docx"
    return f"{stem}.redacted{suffix}"


def _rewrite_paragraph(
    paragraph: Paragraph,
    ranges: list[tuple[int, int, str]],
    replacement: str,
) -> int:
    """Blank out ``ranges`` inside one paragraph. Returns detections applied."""
    runs = paragraph_runs(paragraph)
    if not runs:
        return 0

    texts = [run.text or "" for run in runs]
    full = "".join(texts)
    if not full:
        return 0

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        bounds.append((cursor, cursor + len(text)))
        cursor += len(text)

    # owner[k] = index of the detection covering character k, else -1
    owner = [-1] * len(full)
    for index, (start, end, _) in enumerate(ranges):
        start = max(0, min(start, len(full)))
        end = max(0, min(end, len(full)))
        for position in range(start, end):
            owner[position] = index

    if not any(o >= 0 for o in owner):
        return 0

    emitted: set[int] = set()
    for run, (run_start, run_end) in zip(runs, bounds):
        if run_start >= run_end:
            continue
        pieces: list[str] = []
        position = run_start
        while position < run_end:
            owner_index = owner[position]
            if owner_index < 0:
                pieces.append(full[position])
                position += 1
                continue
            if owner_index not in emitted:
                pieces.append(replacement)
                emitted.add(owner_index)
            while position < run_end and owner[position] == owner_index:
                position += 1

        new_text = "".join(pieces)
        if new_text != (run.text or ""):
            run.text = new_text

    return len(emitted)


def redact_docx(
    source: Path,
    destination: Path,
    detections: list[Detection],
    document: ExtractedDocument,
    *,
    replacement_text: str = "[REDACTED]",
) -> RedactionOutcome:
    outcome = RedactionOutcome(
        document_id=document.document_id,
        source_name=document.document_name,
        output_name=_output_name(document.document_name),
        output_path=destination,
    )

    by_path: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for detection in detections:
        path = detection.location.paragraph_path
        if path is None:
            outcome.skipped.append(
                f"{detection.detection_id}: no paragraph location recorded"
            )
            continue
        by_path[path].append(
            (
                detection.location.char_start,
                detection.location.char_end,
                detection.detection_id,
            )
        )

    try:
        doc = docx.Document(str(source))
    except Exception as exc:
        outcome.warnings.append(
            f"Could not reopen the source DOCX ({type(exc).__name__}); no output produced."
        )
        logger.error("docx_reopen_failed", extra={"document_id": document.document_id})
        return outcome

    available = dict(iter_paragraphs(doc))
    missing = [path for path in by_path if path not in available]
    if missing:
        outcome.warnings.append(
            f"{len(missing)} paragraph location(s) could not be re-found in the "
            "source document and were skipped."
        )
        for path in missing:
            for start, end, detection_id in by_path.pop(path):
                outcome.skipped.append(
                    f"{detection_id}: paragraph {path} no longer exists"
                )

    for path, ranges in sorted(by_path.items()):
        paragraph = available[path]
        ranges.sort(key=lambda r: (r[0], -r[1]))
        applied = _rewrite_paragraph(paragraph, ranges, replacement_text)
        if applied:
            outcome.applied += applied
            outcome.paragraphs_touched += 1
        else:
            outcome.warnings.append(
                f"No editable runs at {path}; text there may live in a construct "
                "this release cannot rewrite."
            )

    # Core properties frequently repeat the author name the document was redacted
    # to protect.
    try:
        core = doc.core_properties
        core.author = ""
        core.last_modified_by = ""
        core.title = ""
        core.subject = ""
        core.comments = ""
        core.keywords = ""
        core.category = ""
    except Exception:  # pragma: no cover
        outcome.warnings.append("Could not clear DOCX document properties.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(str(destination))
    except Exception as exc:
        outcome.warnings.append(
            f"Saving the redacted DOCX failed ({type(exc).__name__}); no output file."
        )
        logger.error("docx_save_failed", extra={"error_type": type(exc).__name__})
        return outcome

    logger.info(
        "docx_redacted",
        extra={
            "document_id": document.document_id,
            "applied": outcome.applied,
            "paragraphs": outcome.paragraphs_touched,
            "skipped": len(outcome.skipped),
        },
    )
    return outcome
