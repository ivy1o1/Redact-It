"""Tests for post-redaction verification.

Verification is a residual-text check, never a guarantee. The contract under
test: it honestly reports whether each approved string is still present, absent,
or something it technically cannot see (a scan), and it never dresses up an
incomplete check as a clean bill of health.

Status literals below were confirmed by reading ``verify.py`` and probing real
output files: ``removed`` / ``partially_removed`` / ``failed`` / ``not_verifiable``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.base import ExtractedDocument
from app.matching.engine import MatchPlan, Target, detect
from app.redaction.verify import (
    VERIFICATION_LIMITATIONS,
    VerificationReport,
    verify_redaction,
)

from .conftest import build_docx, build_pdf, build_scanned_pdf, document_with_segments

SRC_LINES = [
    "Student: Alice Johnson",
    "Phone: (555) 123-4567",
    "Course: Algebra II Grade: A",
]
SRC_TEXT = "\n".join(SRC_LINES)


def detections_for(text: str, *terms: str):
    """Run exact matching over ``text`` so we get real Detection objects."""
    document = document_with_segments([text], name="report.pdf")
    plan = MatchPlan(targets=[Target(kind="exact", term=t, category="named text") for t in terms])
    return detect([document], plan).detections


def pdf_source(name: str = "report.pdf") -> ExtractedDocument:
    # verify only uses the source's file_kind (to pick an extractor) and id.
    return document_with_segments([SRC_TEXT], name=name)


def write_pdf(tmp: Path, lines: list[str], name: str = "redacted.pdf") -> Path:
    path = tmp / name
    path.write_bytes(build_pdf(lines=lines))
    return path


# ---------------------------------------------------------------- status matrix
class TestStatusMatrix:
    def test_removed_when_the_string_is_absent_from_the_output(self, tmp_path: Path) -> None:
        output = write_pdf(tmp_path, ["Student: [REDACTED]", "Course: Algebra II Grade: A"])
        report = verify_redaction(
            output, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        assert [r.status for r in report.results] == ["removed"]
        assert report.results[0].occurrences_in_output == 0
        assert report.residual_text_found is False
        assert report.results[0].detail == "All 1 approved occurrence(s) are gone from the text layer."

    def test_failed_when_the_string_is_still_present_unexplained(self, tmp_path: Path) -> None:
        output = write_pdf(tmp_path, ["Student: Alice Johnson", "Course: Algebra II Grade: A"])
        report = verify_redaction(
            output, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        assert [r.status for r in report.results] == ["failed"]
        assert report.results[0].occurrences_in_output == 1
        assert report.results[0].retained_by_choice == 0
        assert report.residual_text_found is True

    def test_partially_removed_when_only_unapproved_occurrences_remain(
        self, tmp_path: Path
    ) -> None:
        output = write_pdf(tmp_path, ["Student: Alice Johnson"])
        report = verify_redaction(
            output,
            source=pdf_source(),
            approved=detections_for(SRC_TEXT, "Alice Johnson"),
            retained=detections_for(SRC_TEXT, "Alice Johnson"),
        )
        assert [r.status for r in report.results] == ["partially_removed"]
        assert report.results[0].retained_by_choice == 1
        # A legitimate user choice is not a redaction failure.
        assert report.residual_text_found is False

    def test_not_verifiable_when_the_output_could_not_be_read(self, tmp_path: Path) -> None:
        missing = tmp_path / "never-produced.pdf"
        report = verify_redaction(
            missing, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        assert [r.status for r in report.results] == ["not_verifiable"]
        assert report.not_verifiable_reason == "The redacted file was not produced."


# ------------------------------------------------------- presence reporting
class TestPresenceReporting:
    def test_string_present_is_reported_as_present_and_absent_as_absent(
        self, tmp_path: Path
    ) -> None:
        present = write_pdf(tmp_path, ["Student: Alice Johnson"], name="present.pdf")
        absent = write_pdf(tmp_path, ["Student: [REDACTED]"], name="absent.pdf")
        approved = detections_for(SRC_TEXT, "Alice Johnson")

        assert verify_redaction(present, source=pdf_source(), approved=approved).results[0].status == "failed"
        assert verify_redaction(absent, source=pdf_source(), approved=approved).results[0].status == "removed"

    def test_checked_counts_the_distinct_strings(self, tmp_path: Path) -> None:
        output = write_pdf(tmp_path, ["Student: [REDACTED]", "Phone: (555) 000-0000"])
        report = verify_redaction(
            output,
            source=pdf_source(),
            approved=detections_for(SRC_TEXT, "Alice Johnson", "(555) 123-4567"),
        )
        assert report.checked == 2
        assert {r.status for r in report.results} == {"removed"}


# ------------------------------------------------------------------- not-verifiable
class TestCannotVerifyIsNotClaimedAsClean:
    def test_image_only_output_is_not_reported_as_verified_clean(self, tmp_path: Path) -> None:
        scanned = tmp_path / "scan.pdf"
        scanned.write_bytes(build_scanned_pdf())
        report = verify_redaction(
            scanned, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        # A scan has no text layer to read, so this must be 'not_verifiable'
        # rather than the 'removed' the naive absence of text would suggest.
        assert [r.status for r in report.results] == ["not_verifiable"]
        assert report.coverage == "none"
        assert "SCANNED_DOCUMENT_UNSUPPORTED" in report.not_verifiable_reason

    def test_no_approved_text_produces_no_claims(self, tmp_path: Path) -> None:
        output = write_pdf(tmp_path, ["Student: Alice Johnson"])
        report = verify_redaction(output, source=pdf_source(), approved=[])
        assert report.coverage == "none"
        assert report.results == []
        assert "nothing was verified" in report.not_verifiable_reason


# ----------------------------------------------------------------- honest hedging
class TestHonestHedging:
    def test_limitations_are_surfaced_on_the_report(self, tmp_path: Path) -> None:
        output = write_pdf(tmp_path, ["Student: [REDACTED]"])
        report = verify_redaction(
            output, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        for limitation in VERIFICATION_LIMITATIONS:
            assert limitation in report.limitations

    def test_the_no_guarantee_and_no_compliance_disclaimer_is_present(
        self, tmp_path: Path
    ) -> None:
        output = write_pdf(tmp_path, ["Student: [REDACTED]"])
        report = verify_redaction(
            output, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        joined = " ".join(report.limitations)
        assert "does not guarantee that all sensitive data was found or removed" in joined
        assert "not a legal-compliance or certification tool" in joined

    def test_a_clean_report_still_admits_it_only_checked_the_text_layer(
        self, tmp_path: Path
    ) -> None:
        output = write_pdf(tmp_path, ["Student: [REDACTED]"])
        report = verify_redaction(
            output, source=pdf_source(), approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        assert report.coverage == "text-layer-only"
        assert "text layer" in " ".join(report.limitations).lower()


# --------------------------------------------------------------- repeat occurrences
class TestRepeatOccurrences:
    def test_one_removed_one_retained_differs_from_total_failure(self, tmp_path: Path) -> None:
        text = "Alice Johnson signed. Alice Johnson left."
        approved = detections_for(text, "Alice Johnson")  # two occurrences
        assert len(approved) == 2

        output = write_pdf(tmp_path, ["Alice Johnson signed. [REDACTED] left."])

        partial = verify_redaction(
            output, source=pdf_source(), approved=approved, retained=approved[:1]
        )
        assert [r.status for r in partial.results] == ["partially_removed"]
        assert partial.results[0].approved_count == 2
        assert partial.results[0].retained_by_choice == 1
        assert partial.results[0].occurrences_in_output == 1
        assert partial.residual_text_found is False

        failed = verify_redaction(output, source=pdf_source(), approved=approved, retained=[])
        assert [r.status for r in failed.results] == ["failed"]
        assert failed.results[0].retained_by_choice == 0
        assert failed.residual_text_found is True


# ----------------------------------------------------------------------- docx path
class TestDocxSource:
    def test_docx_output_is_checked_with_the_docx_extractor(self, tmp_path: Path) -> None:
        source = document_with_segments([SRC_TEXT], name="report.docx", file_kind="docx")
        output = tmp_path / "redacted.docx"
        output.write_bytes(build_docx(paragraphs=["Student: [REDACTED]", "Course: Algebra II Grade: A"]))
        report = verify_redaction(
            output, source=source, approved=detections_for(SRC_TEXT, "Alice Johnson")
        )
        assert isinstance(report, VerificationReport)
        assert [r.status for r in report.results] == ["removed"]
