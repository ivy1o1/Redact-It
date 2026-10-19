"""Post-redaction verification.

This is a *residual-text check*, not a guarantee. It re-extracts the generated
file with the same extractor used for the source and searches the text layer for
each string the user approved for removal.

Three outcomes are possible per string, and the distinction matters:

* ``removed`` - the string no longer occurs anywhere in the output text layer.
* ``partially_removed`` - the string still occurs, but only as often as the user
  left occurrences unapproved. That is the user's choice, not a failure.
* ``failed`` - the string occurs more often than the unapproved count explains,
  so the redaction pass did not remove everything it claimed to.

Anything that cannot be read back (a scanned output, a corrupt file) is reported
as ``not_verifiable``. Verification never fails silently and never claims to have
checked content it cannot see.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..errors import RedactItError
from ..extraction.base import ExtractedDocument
from ..extraction.docx import extract_docx
from ..extraction.pdf import extract_pdf
from ..logging_setup import get_logger
from ..matching.engine import Detection

logger = get_logger("redaction.verify")

VerificationStatus = Literal["removed", "partially_removed", "failed", "not_verifiable"]

VERIFICATION_LIMITATIONS = [
    "Only the text layer that RedactIt can extract is checked. Text inside PDF "
    "form fields, annotations, comments and embedded files, or inside DOCX text "
    "boxes, comments, footnotes, endnotes and tracked changes, is not verified.",
    "Verification searches for the exact approved strings. The same information "
    "written differently - a paraphrase, a table cell, a chart label or a scanned "
    "image - will not be found.",
    "No OCR is performed, so nothing rendered only as pixels can be verified.",
    "A string may still appear in the output because that particular occurrence "
    "was not approved. Those cases are reported as 'retained by choice', not as a "
    "redaction failure.",
    "RedactIt does not guarantee that all sensitive data was found or removed, and "
    "is not a legal-compliance or certification tool.",
]


class VerifiedText(BaseModel):
    text: str
    status: VerificationStatus
    approved_count: int = 0
    retained_by_choice: int = 0
    occurrences_in_output: int = 0
    output_locations: list[str] = Field(default_factory=list)
    detail: str = ""


class VerificationReport(BaseModel):
    output_name: str
    coverage: Literal["text-layer-only", "none"] = "text-layer-only"
    residual_text_found: bool = False
    checked: int = 0
    results: list[VerifiedText] = Field(default_factory=list)
    not_verifiable_reason: str | None = None
    limitations: list[str] = Field(
        default_factory=lambda: list(VERIFICATION_LIMITATIONS)
    )


@dataclass(slots=True)
class _OutputRead:
    document: ExtractedDocument | None = None
    empty: bool = False
    reason: str | None = None


def _normalise(text: str) -> str:
    """Collapse whitespace so reflowed output still matches the original span."""
    return " ".join(text.split())


def _read_output(
    output_path: Path, *, source: ExtractedDocument, max_chars: int
) -> _OutputRead:
    if not output_path.exists():
        return _OutputRead(reason="The redacted file was not produced.")

    kwargs = {
        "document_id": f"{source.document_id}:output",
        "document_name": output_path.name,
        "max_chars": max_chars,
    }
    extractor = extract_pdf if source.file_kind == "pdf" else extract_docx
    try:
        return _OutputRead(document=extractor(output_path, **kwargs))
    except RedactItError as exc:
        code = getattr(exc, "code", "")
        if code == "EMPTY_DOCUMENT":
            # No extractable text at all: every approved string is necessarily
            # gone from the text layer.
            return _OutputRead(empty=True)
        return _OutputRead(
            reason=f"The redacted file could not be re-read ({code}): {exc.message}"
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("verify_reextract_failed", extra={"error_type": type(exc).__name__})
        return _OutputRead(
            reason=f"The redacted file could not be re-read ({type(exc).__name__})."
        )


def _occurrences_in_output(
    document: ExtractedDocument, needle: str
) -> tuple[int, list[str]]:
    total = 0
    locations: list[str] = []
    for segment in document.segments:
        haystack = _normalise(segment.text)
        count = haystack.count(needle) if needle else 0
        if not count:
            continue
        total += count
        where = f"page {segment.page}" if segment.page else (
            segment.paragraph_path or f"segment {segment.segment_index}"
        )
        locations.append(f"{where} (x{count})")
    return total, locations[:20]


def verify_redaction(
    output_path: Path,
    *,
    source: ExtractedDocument,
    approved: Sequence[Detection],
    retained: Sequence[Detection] = (),
    max_chars: int = 200_000,
) -> VerificationReport:
    """Check ``output_path`` for the strings in ``approved``.

    ``retained`` are the detections the user deliberately did not approve; they
    explain occurrences that are legitimately still present.
    """
    report = VerificationReport(output_name=output_path.name)

    approved_counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for detection in approved:
        needle = _normalise(detection.matched_text)
        if not needle:
            continue
        approved_counts[needle] += 1
        display.setdefault(needle, detection.matched_text)

    retained_counts: Counter[str] = Counter()
    for detection in retained:
        needle = _normalise(detection.matched_text)
        if needle:
            retained_counts[needle] += 1

    if not approved_counts:
        report.coverage = "none"
        report.not_verifiable_reason = (
            "No approved redactions carried text to search for, so nothing was "
            "verified."
        )
        return report

    report.checked = len(approved_counts)
    read = _read_output(output_path, source=source, max_chars=max_chars)

    if read.reason is not None:
        report.coverage = "none"
        report.not_verifiable_reason = read.reason
        for needle, count in approved_counts.items():
            report.results.append(
                VerifiedText(
                    text=display[needle],
                    status="not_verifiable",
                    approved_count=count,
                    detail=read.reason,
                )
            )
        logger.warning(
            "verification_not_possible",
            extra={"document_id": source.document_id},
        )
        return report

    if read.empty or read.document is None:
        # Output has no extractable text layer at all.
        for needle, count in approved_counts.items():
            report.results.append(
                VerifiedText(
                    text=display[needle],
                    status="removed",
                    approved_count=count,
                    detail="The output contains no extractable text at all.",
                )
            )
        report.residual_text_found = False
        report.limitations.append(
            "The redacted file contains no extractable text layer, so verification "
            "could only confirm that the searched strings are absent from text. "
            "Anything rendered as an image was not checked."
        )
        return report

    document = read.document
    if document.truncated:
        report.limitations.append(
            "The output exceeded the character budget for verification, so only "
            "part of it was searched."
        )

    for needle, count in approved_counts.items():
        expected_retained = retained_counts.get(needle, 0)
        found, locations = _occurrences_in_output(document, needle)

        if found == 0:
            status: VerificationStatus = "removed"
            detail = f"All {count} approved occurrence(s) are gone from the text layer."
        elif found <= expected_retained:
            status = "partially_removed"
            detail = (
                f"{found} occurrence(s) remain, matching the {expected_retained} "
                "you did not approve. Approved occurrences were removed."
            )
        else:
            status = "failed"
            detail = (
                f"{found} occurrence(s) remain but only {expected_retained} were "
                "left unapproved. Redaction did not remove everything it reported."
            )
            report.residual_text_found = True

        report.results.append(
            VerifiedText(
                text=display[needle],
                status=status,
                approved_count=count,
                retained_by_choice=expected_retained,
                occurrences_in_output=found,
                output_locations=locations,
                detail=detail,
            )
        )

    logger.info(
        "verification_complete",
        extra={
            "document_id": source.document_id,
            "checked": report.checked,
            "failed": sum(1 for r in report.results if r.status == "failed"),
        },
    )
    return report


__all__ = [
    "VERIFICATION_LIMITATIONS",
    "VerificationReport",
    "VerificationStatus",
    "VerifiedText",
    "verify_redaction",
]
