"""Tests for the deterministic matching engine.

The engine's contract: it only ever reports spans that literally exist in the
extracted text, it honours what the user asked to keep, and it numbers repeated
occurrences so a human reviewer can see how many there are.
"""

from __future__ import annotations

import pytest

from app.matching.engine import (
    Exclusion,
    MatchPlan,
    Target,
    detect,
    detection_id,
)

from .conftest import REPORT_LINES, document_with_segments, one_segment_document

REPORT_TEXT = "\n".join(REPORT_LINES)


def phone_plan() -> MatchPlan:
    return MatchPlan(
        targets=[Target(kind="pattern", term="phone numbers", pattern="phone", category="Phone number")]
    )


def name_plan() -> MatchPlan:
    return MatchPlan(
        targets=[Target(kind="pattern", term="student names", pattern="person_name", category="Student name")]
    )


def exact_plan(term: str) -> MatchPlan:
    return MatchPlan(targets=[Target(kind="exact", term=term, category="named text")])


class TestLocations:
    def test_offsets_point_at_the_real_characters(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        result = detect([document], phone_plan())
        assert result.detections
        for detection in result.detections:
            located = document.segments[0].text[
                detection.location.char_start : detection.location.char_end
            ]
            assert located == detection.matched_text

    def test_pdf_detections_carry_a_one_based_page(self) -> None:
        document = document_with_segments(
            ["Call 555-111-2222 now.", "Also 555-333-4444 here.", "And 555-555-6666."]
        )
        result = detect([document], phone_plan())
        assert [d.location.page for d in result.detections] == [1, 2, 3]
        assert [d.location.segment_index for d in result.detections] == [0, 1, 2]
        assert all(d.location.paragraph_path is None for d in result.detections)

    def test_docx_detections_carry_a_paragraph_path(self) -> None:
        document = document_with_segments(
            ["Contact Alice Johnson"], name="report.docx", file_kind="docx"
        )
        result = detect([document], name_plan())
        assert len(result.detections) == 1
        location = result.detections[0].location
        assert location.file_kind == "docx"
        assert location.paragraph_path == "body:p0"
        assert location.page is None

    def test_document_identity_is_copied_onto_every_detection(self) -> None:
        document = document_with_segments([REPORT_TEXT], document_id="doc_xyz", name="report.pdf")
        result = detect([document], phone_plan())
        assert all(d.location.document_id == "doc_xyz" for d in result.detections)
        assert all(d.location.document_name == "report.pdf" for d in result.detections)

    def test_context_surrounds_the_match_without_repeating_it(self) -> None:
        result = detect([one_segment_document(REPORT_TEXT)], exact_plan("Alice Johnson"))
        detection = result.detections[0]
        assert detection.context_before.endswith("Student: ")
        assert detection.context_after.startswith("\nPhone:")
        assert "Alice Johnson" not in detection.context_before
        assert "Alice Johnson" not in detection.context_after


class TestRepeatedOccurrences:
    def test_every_occurrence_of_a_pattern_is_reported(self) -> None:
        result = detect([one_segment_document(REPORT_TEXT)], phone_plan())
        assert [d.matched_text for d in result.detections] == ["(555) 123-4567", "555-987-6543"]
        assert [d.occurrence for d in result.detections] == [1, 2]
        assert result.per_pattern_counts == {"phone": 2}

    def test_occurrence_numbering_continues_across_pages(self) -> None:
        document = document_with_segments(
            ["Call 555-111-2222 now.", "Also 555-333-4444 here.", "And 555-555-6666."]
        )
        result = detect([document], phone_plan())
        assert [d.occurrence for d in result.detections] == [1, 2, 3]

    def test_occurrence_numbering_restarts_for_a_different_document(self) -> None:
        first = document_with_segments(["Phone: 555-111-2222"], document_id="doc_1", name="one.pdf")
        second = document_with_segments(["Phone: 555-333-4444"], document_id="doc_2", name="two.pdf")
        result = detect([first, second], phone_plan())
        assert [(d.location.document_name, d.occurrence) for d in result.detections] == [
            ("one.pdf", 1),
            ("two.pdf", 1),
        ]

    def test_case_variants_of_one_term_are_occurrences_of_that_term(self) -> None:
        document = one_segment_document("Alice Johnson and alice johnson and ALICE JOHNSON")
        result = detect([document], exact_plan("Alice Johnson"))
        assert [d.matched_text for d in result.detections] == [
            "Alice Johnson",
            "alice johnson",
            "ALICE JOHNSON",
        ]
        assert [d.occurrence for d in result.detections] == [1, 2, 3]
        # Only the verbatim spelling is certain; the rest are flagged for review.
        assert [d.fuzzy for d in result.detections] == [False, True, True]
        assert [round(d.confidence, 2) for d in result.detections] == [0.95, 0.7, 0.7]

    def test_a_term_broken_across_a_line_is_still_found(self) -> None:
        # PDF text extraction routinely splits a printed phrase over two lines.
        document = one_segment_document("See Project\n   Falcon below.")
        result = detect([document], exact_plan("Project Falcon"))
        assert len(result.detections) == 1
        assert result.detections[0].fuzzy is True
        assert result.unmatched_exact_terms == []

    def test_line_broken_and_verbatim_occurrences_are_numbered_together(self) -> None:
        document = one_segment_document("Project\n   Falcon here and Project Falcon there")
        result = detect([document], exact_plan("Project Falcon"))
        assert len(result.detections) == 2
        assert [d.occurrence for d in result.detections] == [1, 2]
        assert result.unmatched_exact_terms == []

    def test_tabs_and_repeated_spaces_count_as_separators(self) -> None:
        document = one_segment_document("Code\tX-99 and Code X-99 again")
        result = detect([document], exact_plan("Code X-99"))
        assert len(result.detections) == 2

    def test_adjacent_occurrences_do_not_overlap(self) -> None:
        document = one_segment_document("555-111-2222 555-111-2222 555-111-2222")
        result = detect([document], phone_plan())
        spans = [(d.location.char_start, d.location.char_end) for d in result.detections]
        assert len(spans) == 3
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            assert previous_end <= next_start


class TestExclusions:
    def test_a_kept_course_name_is_not_proposed_for_redaction(self) -> None:
        # The loose name heuristic would otherwise flag "English Literature" on the
        # "Course: English Literature" line.
        document = one_segment_document(REPORT_TEXT)
        result = detect(
            [document],
            MatchPlan(
                targets=name_plan().targets,
                exclusions=[Exclusion(kind="exact", term="course names")],
            ),
        )
        matched = [d.matched_text for d in result.detections]
        assert "Alice Johnson" in matched
        assert "English Literature" not in matched
        assert result.suppressed_by_exclusion >= 1

    def test_an_unrelated_exclusion_suppresses_nothing(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        baseline = detect([document], name_plan())
        result = detect(
            [document],
            MatchPlan(targets=name_plan().targets, exclusions=[Exclusion(kind="exact", term="grades")]),
        )
        assert len(result.detections) == len(baseline.detections)
        assert result.suppressed_by_exclusion == 0

    def test_exclusions_are_case_insensitive(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        result = detect(
            [document],
            MatchPlan(
                targets=name_plan().targets,
                exclusions=[Exclusion(kind="exact", term="ALICE JOHNSON")],
            ),
        )
        assert "Alice Johnson" not in [d.matched_text for d in result.detections]
        assert result.suppressed_by_exclusion >= 1

    def test_a_pattern_can_be_excluded(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        result = detect(
            [document],
            MatchPlan(
                targets=[
                    Target(kind="pattern", term="phone numbers", pattern="phone"),
                    Target(kind="pattern", term="email addresses", pattern="email"),
                ],
                exclusions=[Exclusion(kind="pattern", pattern="phone")],
            ),
        )
        matched = [d.matched_text for d in result.detections]
        assert not any("555" in text for text in matched)
        assert "a.johnson@example-school.test" in matched

    def test_protection_stops_at_the_next_label(self) -> None:
        # "Alice Johnson" is protected, but that must not swallow the phone number
        # that follows it on the same line.
        document = one_segment_document("Name: Alice Johnson Phone: 555-111-2222")
        result = detect(
            [document],
            MatchPlan(
                targets=phone_plan().targets,
                exclusions=[Exclusion(kind="exact", term="Alice Johnson")],
            ),
        )
        assert [d.matched_text for d in result.detections] == ["555-111-2222"]
        assert result.suppressed_by_exclusion == 0

    def test_protection_stops_at_the_end_of_the_line(self) -> None:
        document = one_segment_document("Alice Johnson\n555-111-2222")
        result = detect(
            [document],
            MatchPlan(
                targets=phone_plan().targets,
                exclusions=[Exclusion(kind="exact", term="Alice Johnson")],
            ),
        )
        assert [d.matched_text for d in result.detections] == ["555-111-2222"]

    def test_protection_is_bounded_even_in_one_very_long_line(self) -> None:
        tail = "keep this " + "word " * 60
        document = one_segment_document(f"Name: Alice Johnson {tail} Phone: 555-111-2222")
        result = detect(
            [document],
            MatchPlan(
                targets=[
                    Target(kind="pattern", term="names", pattern="person_name"),
                    Target(kind="pattern", term="phones", pattern="phone"),
                ],
                exclusions=[Exclusion(kind="exact", term="Name")],
            ),
        )
        # The exclusion protects its own clause only; the phone number 400+ chars
        # later is still proposed.
        assert "555-111-2222" in [d.matched_text for d in result.detections]

    def test_protect_clause_can_be_switched_off(self) -> None:
        document = one_segment_document("Name: Alice Johnson Phone: 555-111-2222")
        result = detect(
            [document],
            MatchPlan(
                targets=name_plan().targets,
                exclusions=[Exclusion(kind="exact", term="Alice Johnson", protect_clause=False)],
            ),
        )
        assert "Alice Johnson" not in [d.matched_text for d in result.detections]

    def test_an_exclusion_with_no_term_protects_nothing(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        baseline = detect([document], phone_plan())
        result = detect(
            [document],
            MatchPlan(targets=phone_plan().targets, exclusions=[Exclusion(kind="exact", term="   ")]),
        )
        assert len(result.detections) == len(baseline.detections)
        assert result.suppressed_by_exclusion == 0

    def test_an_exclusion_for_a_different_document_is_not_applied(self) -> None:
        # "course names" resolves against each document separately, so a phrase
        # that only exists in one file cannot change what is protected in another.
        first = document_with_segments(["Course: Algebra II"], document_id="doc_1")
        second = document_with_segments(["Student: Alice Johnson"], document_id="doc_2")
        result = detect(
            [first, second],
            MatchPlan(
                targets=name_plan().targets,
                exclusions=[Exclusion(kind="exact", term="course names")],
            ),
        )
        assert [d.matched_text for d in result.detections] == ["Alice Johnson"]


class TestResolvedExclusions:
    def test_a_verbatim_phrase_is_used_as_is(self) -> None:
        document = one_segment_document("Course: Algebra II Grade: A")
        result = detect(
            [document], MatchPlan(exclusions=[Exclusion(kind="exact", term="Algebra II")])
        )
        resolved = result.resolved_exclusions[document.document_id]
        assert [e.match_terms for e in resolved] == [["Algebra II"]]

    def test_a_phrase_that_never_appears_falls_back_to_its_tokens(self) -> None:
        document = one_segment_document("Course: Algebra II Grade: A")
        result = detect(
            [document], MatchPlan(exclusions=[Exclusion(kind="exact", term="course names")])
        )
        resolved = result.resolved_exclusions[document.document_id]
        # Only the token that actually occurs is used, so protection does not
        # spread further than the user intended.
        assert [e.match_terms for e in resolved] == [["course"]]

    def test_resolved_exclusions_are_keyed_by_document(self) -> None:
        first = document_with_segments(["Course: Algebra II"], document_id="doc_1")
        second = document_with_segments(["Nothing relevant"], document_id="doc_2")
        result = detect(
            [first, second],
            MatchPlan(exclusions=[Exclusion(kind="exact", term="course names")]),
        )
        assert set(result.resolved_exclusions) == {"doc_1", "doc_2"}
        assert result.resolved_exclusions["doc_1"][0].match_terms == ["course"]
        # Nothing in the second document matched, so the original phrase is kept
        # and reported back to the user unchanged.
        assert result.resolved_exclusions["doc_2"][0].match_terms == ["course names"]

    def test_pattern_exclusions_are_not_rewritten(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        result = detect(
            [document], MatchPlan(exclusions=[Exclusion(kind="pattern", pattern="phone")])
        )
        resolved = result.resolved_exclusions[document.document_id]
        assert resolved[0].match_terms == []
        assert resolved[0].effective_terms() == []


class TestNothingIsInvented:
    def test_a_term_that_appears_nowhere_is_reported_not_fabricated(self) -> None:
        result = detect([one_segment_document(REPORT_TEXT)], exact_plan("Not In Here"))
        assert result.detections == []
        assert result.unmatched_exact_terms == ["Not In Here"]
        assert any("nothing was invented" in warning for warning in result.warnings)

    def test_a_term_only_in_protected_text_explains_itself(self) -> None:
        result = detect(
            [one_segment_document(REPORT_TEXT)],
            MatchPlan(
                targets=[Target(kind="exact", term="Algebra II", category="named text")],
                exclusions=[Exclusion(kind="exact", term="Algebra II")],
            ),
        )
        assert result.detections == []
        assert result.unmatched_exact_terms == []
        assert any("falls inside text you asked to keep" in w for w in result.warnings)

    def test_an_unsupported_detector_name_is_ignored_with_a_warning(self) -> None:
        result = detect(
            [one_segment_document(REPORT_TEXT)],
            MatchPlan(targets=[Target(kind="pattern", term="x", pattern="passport_number")]),
        )
        assert result.detections == []
        assert any("unsupported detector name" in w for w in result.warnings)

    def test_a_supported_detector_next_to_an_unsupported_one_still_runs(self) -> None:
        result = detect(
            [one_segment_document(REPORT_TEXT)],
            MatchPlan(
                targets=[
                    Target(kind="pattern", term="x", pattern="retina_scan"),
                    Target(kind="pattern", term="phones", pattern="phone"),
                ]
            ),
        )
        assert result.per_pattern_counts == {"phone": 2}
        assert any("retina_scan" in w for w in result.warnings)

    def test_blank_terms_and_empty_plans_produce_nothing(self) -> None:
        document = one_segment_document(REPORT_TEXT)
        assert detect([document], MatchPlan()).detections == []
        assert detect(
            [document], MatchPlan(targets=[Target(kind="exact", term="   ", category="x")])
        ).detections == []

    def test_a_document_without_a_text_layer_yields_no_detections(self) -> None:
        document = document_with_segments([REPORT_TEXT], has_text_layer=False)
        assert detect([document], phone_plan()).detections == []

    def test_heuristic_matches_are_flagged_for_review(self) -> None:
        result = detect([one_segment_document(REPORT_TEXT)], name_plan())
        loose = [d for d in result.detections if d.pattern_name == "person_name" and d.confidence < 0.5]
        assert loose, "the loose name heuristic should have proposed something"
        assert all(d.heuristic for d in loose)
        assert any("least reliable detector" in w for w in result.warnings)


class TestDocumentScope:
    def test_scope_limits_detection_to_the_named_documents(self) -> None:
        first = document_with_segments(["Phone: 555-111-2222"], document_id="doc_1", name="one.pdf")
        second = document_with_segments(["Phone: 555-333-4444"], document_id="doc_2", name="two.pdf")
        result = detect([first, second], phone_plan(), document_scope=["doc_2"])
        assert [d.location.document_name for d in result.detections] == ["two.pdf"]
        assert list(result.resolved_exclusions) == ["doc_2"]

    def test_a_scope_matching_nothing_yields_nothing(self) -> None:
        document = document_with_segments(["Phone: 555-111-2222"], document_id="doc_1")
        assert detect([document], phone_plan(), document_scope=["doc_missing"]).detections == []

    def test_no_scope_means_every_document(self) -> None:
        first = document_with_segments(["Phone: 555-111-2222"], document_id="doc_1", name="one.pdf")
        second = document_with_segments(["Phone: 555-333-4444"], document_id="doc_2", name="two.pdf")
        result = detect([first, second], phone_plan())
        assert len(result.detections) == 2


class TestOverlapResolution:
    def test_an_exact_target_and_a_detector_do_not_double_report_a_span(self) -> None:
        document = one_segment_document("Student: Alice Johnson")
        result = detect(
            [document],
            MatchPlan(
                targets=[
                    Target(kind="exact", term="Alice Johnson", category="named text"),
                    Target(kind="pattern", term="names", pattern="person_name"),
                ]
            ),
        )
        assert len(result.detections) == 1
        assert result.detections[0].strategy == "exact"
        assert result.detections[0].confidence == 0.95

    def test_a_longer_match_at_the_same_start_wins(self) -> None:
        document = one_segment_document("Student: Alice Johnson")
        result = detect(
            [document],
            MatchPlan(
                targets=[
                    Target(kind="exact", term="Alice", category="named text"),
                    Target(kind="pattern", term="names", pattern="person_name"),
                ]
            ),
        )
        assert [d.matched_text for d in result.detections] == ["Alice Johnson"]

    def test_detection_ids_are_unique_within_one_analysis(self) -> None:
        document = document_with_segments([REPORT_TEXT, REPORT_TEXT])
        result = detect([document], MatchPlan(targets=name_plan().targets + phone_plan().targets))
        ids = [d.detection_id for d in result.detections]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("det_") and len(i) == 28 for i in ids)

    def test_the_same_span_in_two_documents_gets_different_ids(self) -> None:
        first = document_with_segments(["Phone: 555-111-2222"], document_id="doc_1")
        second = document_with_segments(["Phone: 555-111-2222"], document_id="doc_2")
        result = detect([first, second], phone_plan())
        ids = [d.detection_id for d in result.detections]
        assert len(ids) == 2
        assert ids[0] != ids[1]


class TestDetectionId:
    def test_the_id_is_a_pure_function_of_the_span(self) -> None:
        kwargs = {
            "document_id": "doc_a",
            "segment_index": 3,
            "char_start": 10,
            "char_end": 22,
            "matched_text": "Alice Johnson",
            "strategy": "exact",
            "pattern_name": None,
        }
        assert detection_id(**kwargs) == detection_id(**kwargs)

    @pytest.mark.parametrize(
        "field",
        ["document_id", "segment_index", "char_start", "char_end", "matched_text", "strategy", "pattern_name"],
    )
    def test_every_component_changes_the_id(self, field: str) -> None:
        base = {
            "document_id": "doc_a",
            "segment_index": 3,
            "char_start": 10,
            "char_end": 22,
            "matched_text": "Alice Johnson",
            "strategy": "exact",
            "pattern_name": None,
        }
        changed = dict(base)
        changed[field] = "different" if field in {"document_id", "matched_text", "strategy", "pattern_name"} else 99
        assert detection_id(**base) != detection_id(**changed)


class TestDeterminism:
    def test_the_same_input_yields_the_same_detections(self) -> None:
        plan = MatchPlan(targets=name_plan().targets + phone_plan().targets)
        document = one_segment_document(REPORT_TEXT)
        first = detect([document], plan)
        second = detect([document], plan)
        assert [d.model_dump() for d in first.detections] == [d.model_dump() for d in second.detections]
        assert first.warnings == second.warnings
        assert first.per_pattern_counts == second.per_pattern_counts

    def test_detection_order_follows_the_document(self) -> None:
        result = detect(
            [one_segment_document(REPORT_TEXT)],
            MatchPlan(targets=name_plan().targets + phone_plan().targets),
        )
        offsets = [d.location.char_start for d in result.detections]
        assert offsets == sorted(offsets)
