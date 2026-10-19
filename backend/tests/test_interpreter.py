"""Tests for instruction interpretation: the model proposes, the code validates.

The property under test throughout is that a 1B model cannot widen what gets
redacted on its own. Every proposal is checked, and every rejection is reported
to the caller rather than dropped.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.errors import InstructionAmbiguousError, InstructionUnusableError
from app.llm.interpreter import (
    _label_matches_pattern,
    _looks_like_literal_phrase,
    build_plan_schema,
    interpret_instruction,
)
from app.matching.patterns import PATTERN_REGISTRY, supported_pattern_names

from .conftest import make_settings

# The interpreter never touches the filesystem; the settings object just needs a
# scratch root that is not the repository.
_SCRATCH = Path(tempfile.mkdtemp(prefix="redactit-interpreter-tests-"))


def build_settings(**overrides: Any):
    return make_settings(_SCRATCH, **overrides)

SAMPLE = (
    "Student: Alice Johnson\n"
    "Phone: (555) 123-4567\n"
    "Home Address: 42 Maple Street, Springfield\n"
    "Course: Algebra II Grade: A\n"
    "Emergency contact: Brian O'Connor\n"
    "Project Falcon launch date\n"
)

CLEAR = "Remove all student names, phone numbers and home addresses."


class FakeClient:
    """Stands in for LlamaCppClient and records exactly what was asked."""

    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.output = output if output is not None else {}
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.output

    @property
    def user_prompt(self) -> str:
        return self.calls[-1]["user_prompt"]

    @property
    def schema(self) -> dict[str, Any] | None:
        return self.calls[-1]["schema"]


def target(
    term: str, kind: str = "pattern", pattern: str | None = None, category: str = "c"
) -> dict[str, Any]:
    return {"term": term, "kind": kind, "pattern": pattern, "category": category}


def interpret(
    instruction: str = CLEAR,
    model_output: dict[str, Any] | None = None,
    *,
    client: FakeClient | None | str = "auto",
    sample: str = SAMPLE,
    mode: str = "model",
    names: list[str] | None = None,
    settings: Any = None,
):
    if client == "auto":
        client = FakeClient(model_output)
    settings = settings or build_settings()
    result = asyncio.run(
        interpret_instruction(
            instruction=instruction,
            document_sample=sample,
            client=client,  # type: ignore[arg-type]
            settings=settings,
            mode=mode,  # type: ignore[arg-type]
            document_names=names,
        )
    )
    return result, client


def patterns_of(interpretation) -> set[str | None]:
    return {t.pattern for t in interpretation.plan.targets}


def kinds_of(interpretation) -> set[str]:
    return {t.kind for t in interpretation.plan.targets}


class TestDeterministicBasePlan:
    """The parser alone must produce a usable plan - the model is an addition."""

    def test_named_data_types_become_pattern_targets(self) -> None:
        result, _ = interpret(CLEAR, mode="deterministic")
        assert patterns_of(result) == {"person_name", "phone", "address"}
        assert all(t.source == "instruction" for t in result.plan.targets)

    def test_categories_come_from_the_closed_registry(self) -> None:
        result, _ = interpret("Remove every email address and IP address.", mode="deterministic")
        for detection_target in result.plan.targets:
            assert detection_target.category == PATTERN_REGISTRY[detection_target.pattern].label

    def test_a_quoted_phrase_becomes_an_exact_target(self) -> None:
        result, _ = interpret(
            'Remove the phone numbers and "Project Falcon".', mode="deterministic"
        )
        assert {"exact", "pattern"} == kinds_of(result)
        assert "Project Falcon" in {t.term for t in result.plan.targets}

    def test_keep_clauses_become_exclusions_not_targets(self) -> None:
        result, _ = interpret(
            "Remove all student names, but keep the grades unchanged.", mode="deterministic"
        )
        assert [e.term for e in result.plan.exclusions] == ["grades"]
        assert "person_name" in patterns_of(result)
        assert "date" not in patterns_of(result)

    def test_the_parse_is_reported_back(self) -> None:
        result, _ = interpret(CLEAR, mode="deterministic")
        assert set(result.parsed) == {
            "remove_clauses", "keep_clauses", "exact_targets", "pattern_targets", "exclusion_terms"
        }
        assert result.parsed["pattern_targets"]

    def test_model_is_not_consulted_and_says_so(self) -> None:
        result, client = interpret(CLEAR, mode="deterministic")
        assert client.calls == []
        assert result.source == "deterministic"
        assert result.model_used is False
        assert result.model_raw is None
        assert any("INTERPRETATION_MODE=deterministic" in w for w in result.warnings)

    def test_document_names_are_carried_through_untouched(self) -> None:
        result, _ = interpret(CLEAR, mode="deterministic", names=["a.pdf", "b.docx"])
        assert result.document_names == ["a.pdf", "b.docx"]
        assert result.document_scope == "all"


class TestModelIsRequiredWhenRequested:
    def test_a_missing_client_is_an_error_not_a_silent_fallback(self) -> None:
        with pytest.raises(InstructionUnusableError, match="no local model client"):
            interpret(CLEAR, client=None)

    def test_model_output_is_recorded_for_the_review_screen(self) -> None:
        output = {"targets": [target("email addresses", pattern="email")]}
        result, _ = interpret("Remove the phone numbers.", output)
        assert result.source == "model+deterministic"
        assert result.model_used is True
        assert result.model_raw == output


class TestRejectionsAreNeverSilent:
    """Regression: pydantic copies list fields, so these two reports once vanished."""

    def test_a_rejected_proposal_reaches_the_returned_interpretation(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("grades", pattern="credit_card")]})
        assert result.rejected_model_proposals == [
            "detector 'credit_card' for 'grades': label does not match that detector"
        ]

    def test_a_warning_reaches_the_returned_interpretation(self) -> None:
        result, _ = interpret(CLEAR, {"document_scope": "named", "document_names": []})
        assert any("named none" in w for w in result.warnings)

    def test_rejections_accumulate_in_order(self) -> None:
        output = {
            "targets": [
                target("grades", pattern="credit_card"),
                target("whatever", pattern="not_a_detector"),
                target("", "exact"),
            ]
        }
        result, _ = interpret("Remove the phone numbers.", output)
        assert len(result.rejected_model_proposals) == 3
        assert "not_a_detector" in result.rejected_model_proposals[1]


class TestTargetValidation:
    def test_an_unknown_detector_is_refused(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("income", pattern="salary")]})
        assert patterns_of(result) == {"phone"}
        assert "unknown detector 'salary'" in result.rejected_model_proposals

    def test_a_label_must_match_its_own_detector(self) -> None:
        # The classic 1B failure: mapping "grades" onto credit_card.
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("grades", pattern="credit_card")]})
        assert "credit_card" not in patterns_of(result)

    def test_a_consistent_label_and_detector_is_accepted(self) -> None:
        result, _ = interpret("Remove the addresses.", {"targets": [target("mobile numbers", pattern="phone")]})
        assert {"address", "phone"} <= patterns_of(result)
        added = [t for t in result.plan.targets if t.source == "model"]
        assert [t.term for t in added] == ["mobile numbers"]

    def test_a_detector_the_instruction_already_named_is_not_duplicated(self) -> None:
        result, _ = interpret(CLEAR, {"targets": [target("mobile numbers", pattern="phone")]})
        phone = [t for t in result.plan.targets if t.pattern == "phone"]
        assert len(phone) == 1
        # The instruction's own wording is authoritative, not the model's.
        assert phone[0].source == "instruction"
        assert phone[0].term == "phone numbers"

    def test_an_inconsistent_label_that_reads_as_literal_text_is_salvaged(self) -> None:
        result, _ = interpret(
            "Remove the phone numbers.", {"targets": [target("Project Falcon", pattern="ssn")]}
        )
        assert "ssn" not in patterns_of(result)
        assert "Project Falcon" in {t.term for t in result.plan.targets if t.kind == "exact"}
        assert any("re-read as literal text" in r for r in result.rejected_model_proposals)

    def test_a_salvaged_literal_must_exist_in_the_document(self) -> None:
        result, _ = interpret(
            "Remove the phone numbers.", {"targets": [target("Blue Harbor Contract", pattern="ssn")]}
        )
        assert kinds_of(result) == {"pattern"}
        assert any("label does not match" in r for r in result.rejected_model_proposals)

    def test_an_exact_target_is_accepted_as_given(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("Project Falcon", "exact")]})
        assert "Project Falcon" in {t.term for t in result.plan.targets}

    def test_a_duplicate_exact_target_is_not_added_twice(self) -> None:
        output = {"targets": [target("Project Falcon", "exact"), target("Project Falcon", "exact")]}
        result, _ = interpret("Remove the phone numbers.", output)
        assert [t.term for t in result.plan.targets].count("Project Falcon") == 1

    def test_an_empty_exact_term_is_refused(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("", "exact")]})
        assert "exact target with empty term" in result.rejected_model_proposals

    def test_an_unsupported_kind_is_refused(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("x", "regex", "phone")]})
        assert any("unsupported kind" in r for r in result.rejected_model_proposals)

    def test_non_object_entries_are_ignored(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": ["phone", None, 7]})
        assert patterns_of(result) == {"phone"}
        assert result.rejected_model_proposals == []

    def test_a_blank_category_falls_back_to_unspecified(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"targets": [target("Project Falcon", "exact", category="  ")]})
        exact = [t for t in result.plan.targets if t.kind == "exact"]
        assert exact[0].category == "unspecified"

    def test_the_model_cannot_introduce_a_pattern_it_was_never_offered(self) -> None:
        # Even a self-consistent label cannot name a detector outside the registry.
        output = {"targets": [target("medical records", pattern="mrn")]}
        result, _ = interpret("Remove the phone numbers.", output)
        assert all(t.pattern in PATTERN_REGISTRY for t in result.plan.targets)


class TestKeepBeatsModel:
    def test_a_target_the_user_asked_to_keep_is_dropped(self) -> None:
        output = {"targets": [target("Alice Johnson", "exact")]}
        result, _ = interpret(
            "Remove the phone numbers but keep Alice Johnson unchanged.", output
        )
        assert "Alice Johnson" not in {t.term for t in result.plan.targets}
        assert any("you asked to keep this" in r for r in result.rejected_model_proposals)
        assert any("says to keep it" in w for w in result.warnings)

    def test_the_keep_exclusion_itself_survives(self) -> None:
        result, _ = interpret("Remove the phone numbers but keep Alice Johnson unchanged.")
        assert "Alice Johnson" in [e.term for e in result.plan.exclusions]

    def test_a_pattern_target_for_kept_data_is_still_dropped(self) -> None:
        output = {"targets": [target("student names", pattern="person_name")]}
        result, _ = interpret("Remove the phone numbers but keep the student names.", output)
        assert "person_name" not in patterns_of(result)


class TestExclusionValidation:
    def test_an_exclusion_only_counts_if_the_text_is_really_there(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"exclusions": [{"term": "student_names", "kind": "exact"}]})
        assert result.plan.exclusions == []
        assert any("not present in the document" in r for r in result.rejected_model_proposals)

    def test_a_literal_name_found_in_the_document_is_protected(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"exclusions": [{"term": "Brian O'Connor", "kind": "exact"}]})
        assert [("Brian O'Connor", "model")] == [(e.term, e.source) for e in result.plan.exclusions]

    def test_protection_matches_across_case(self) -> None:
        result, _ = interpret("Remove the phone numbers.", {"exclusions": [{"term": "brian o'connor", "kind": "exact"}]})
        assert result.plan.exclusions[0].term == "brian o'connor"

    def test_an_exclusion_the_parser_already_made_is_not_duplicated(self) -> None:
        result, _ = interpret("Remove the phone numbers but keep Brian O'Connor.", {"exclusions": [{"term": "Brian O'Connor", "kind": "exact"}]})
        assert [e.term for e in result.plan.exclusions].count("Brian O'Connor") == 1

    def test_blank_and_non_object_exclusions_are_ignored(self) -> None:
        output = {"exclusions": [{"term": "   ", "kind": "exact"}, "Project Falcon", 3]}
        result, _ = interpret("Remove the phone numbers.", output)
        assert result.plan.exclusions == []

    def test_the_model_cannot_turn_an_exclusion_into_a_redaction(self) -> None:
        output = {"exclusions": [{"term": "Project Falcon", "kind": "exact"}]}
        result, _ = interpret("Remove the phone numbers.", output)
        assert "Project Falcon" not in {t.term for t in result.plan.targets}


class TestDocumentScope:
    def test_scope_defaults_to_everything(self) -> None:
        result, _ = interpret(CLEAR)
        assert (result.document_scope, result.document_names) == ("all", [])

    def test_a_named_scope_is_honoured(self) -> None:
        output = {"document_scope": "named", "document_names": ["report.pdf"]}
        result, _ = interpret(CLEAR, output)
        assert (result.document_scope, result.document_names) == ("named", ["report.pdf"])

    def test_a_named_scope_without_names_widens_to_all_and_warns(self) -> None:
        result, _ = interpret(CLEAR, {"document_scope": "named", "document_names": []})
        assert result.document_scope == "all"
        assert any("named none" in w for w in result.warnings)

    def test_an_unrecognised_scope_value_is_treated_as_all(self) -> None:
        result, _ = interpret(CLEAR, {"document_scope": "only_the_second_file"})
        assert result.document_scope == "all"

    def test_names_are_trimmed_and_blank_ones_dropped(self) -> None:
        output = {"document_scope": "named", "document_names": ["  a.pdf ", "", "b.docx"]}
        result, _ = interpret(CLEAR, output)
        assert result.document_names == ["a.pdf", "b.docx"]

    def test_the_name_list_is_capped(self) -> None:
        output = {"document_scope": "named", "document_names": [f"f{i}.pdf" for i in range(60)]}
        result, _ = interpret(CLEAR, output)
        assert len(result.document_names) == 20


class TestAmbiguityIsOwnedByTheParser:
    def test_a_small_model_flagging_everything_does_not_block_work(self) -> None:
        result, _ = interpret(CLEAR, {"ambiguous": True, "clarification_needed": "which names?"})
        assert result.ambiguous is False
        assert result.clarification_needed is None

    def test_an_instruction_with_no_targets_raises_an_actionable_error(self) -> None:
        with pytest.raises(InstructionAmbiguousError) as exc:
            interpret("redact stuff")
        assert exc.value.details["supported_detectors"] == supported_pattern_names()
        assert "exclusions_recognised" in exc.value.details
        message = str(exc.value)
        assert "Remove all student names" in message  # concrete example, not a shrug

    def test_the_model_clarification_is_surfaced_inside_the_error(self) -> None:
        with pytest.raises(InstructionAmbiguousError) as exc:
            interpret("redact stuff", {"ambiguous": True, "clarification_needed":  "  which fields? "})
        assert "which fields?" in str(exc.value)
        assert exc.value.details["clarification_needed"] == "which fields?"

    def test_a_non_string_clarification_is_ignored(self) -> None:
        with pytest.raises(InstructionAmbiguousError) as exc:
            interpret("redact stuff", {"ambiguous": True, "clarification_needed": 42})
        assert exc.value.details["clarification_needed"] is None

    def test_exclusions_recognised_are_reported_even_when_nothing_can_be_redacted(self) -> None:
        with pytest.raises(InstructionAmbiguousError) as exc:
            interpret("Keep Alice Johnson and every grade exactly as they are.")
        assert set(exc.value.details["exclusions_recognised"]) >= {"Alice Johnson"}

    def test_a_model_target_rescues_an_otherwise_ambiguous_instruction(self) -> None:
        result, _ = interpret("redact stuff", {"targets": [target("email addresses", pattern="email")]})
        assert "email" in patterns_of(result)
        assert result.ambiguous is False


class TestPromptIsBounded:
    def test_only_the_allowed_prefix_of_the_document_is_shown(self) -> None:
        settings = build_settings(llm_max_prompt_chars=40)
        result, client = interpret(CLEAR, {}, settings=settings)
        # The cut lands inside the phone number, so the boundary is observed
        # rather than assumed: the head of the document arrives and everything
        # from the rest of the file - address, later names - does not.
        assert "Student: Alice Johnson\nPhone: (555) 123-" in client.user_prompt
        assert "4567" not in client.user_prompt
        assert "42 Maple Street" not in client.user_prompt
        assert "Brian O'Connor" not in client.user_prompt
        assert any("Only the first 40 characters" in w for w in result.warnings)

    def test_an_unbounded_sample_is_not_warned_about(self) -> None:
        result, client = interpret(CLEAR, {})
        assert SAMPLE in client.user_prompt
        assert not any("Only the first" in w for w in result.warnings)

    def test_a_zero_budget_shows_no_document_text(self) -> None:
        settings = build_settings(llm_max_prompt_chars=0)
        result, client = interpret(CLEAR, {}, settings=settings)
        assert "42 Maple Street" not in client.user_prompt
        assert "Student: Alice Johnson" not in client.user_prompt

    def test_the_instruction_and_file_names_frame_the_sample(self) -> None:
        result, client = interpret(CLEAR, {}, names=["term-report.pdf"])
        assert CLEAR in client.user_prompt
        assert "term-report.pdf" in client.user_prompt
        assert "Output the JSON plan" in client.user_prompt

    def test_the_plan_schema_is_sent_for_constrained_decoding(self) -> None:
        result, client = interpret(CLEAR, {})
        assert client.schema == build_plan_schema()
        assert client.calls[0]["system_prompt"]

    def test_parser_notes_are_kept_beside_model_warnings(self) -> None:
        result, _ = interpret("Remove the phone numbers and keep grades.", {})
        assert isinstance(result.warnings, list)


class TestPlanSchema:
    def test_the_schema_requires_every_top_level_field(self) -> None:
        schema = build_plan_schema()
        assert sorted(schema["required"]) == [
            "ambiguous", "clarification_needed", "document_names", "document_scope",
            "exclusions", "targets",
        ]

    def test_the_detector_enum_is_the_closed_registry(self) -> None:
        enum = build_plan_schema()["properties"]["targets"]["items"]["properties"]["pattern"]["enum"]
        assert set(enum) == set(supported_pattern_names()) | {None}
        assert enum[-1] is None

    def test_no_extra_properties_anywhere(self) -> None:
        schema = build_plan_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["targets"]["items"]["additionalProperties"] is False
        assert schema["properties"]["exclusions"]["items"]["additionalProperties"] is False

    def test_an_exclusion_can_only_be_literal_text(self) -> None:
        exclusions = build_plan_schema()["properties"]["exclusions"]["items"]
        assert exclusions["properties"]["kind"]["enum"] == ["exact"]
        assert exclusions["required"] == ["term", "kind"]

    def test_scope_is_a_closed_choice(self) -> None:
        assert build_plan_schema()["properties"]["document_scope"]["enum"] == ["all", "named"]


class TestLabelConsistencyCheck:
    @pytest.mark.parametrize(
        ("label", "pattern", "expected"),
        [
            ("mobile numbers", "phone", True),
            ("phone", "phone", True),
            ("email addresses", "email", True),
            ("Web addresses", "url", True),
            ("ssn", "ssn", True),
            ("home addresses", "address", True),
            ("grades", "credit_card", False),
            ("student", "credit_card", False),
            ("Project Falcon", "ssn", False),
            ("", "phone", False),
            ("Alice", "person_name", False),
        ],
    )
    def test_a_label_only_supports_a_detector_it_mentions(
        self, label: str, pattern: str, expected: bool
    ) -> None:
        assert _label_matches_pattern(label, pattern) is expected

    def test_the_check_is_case_insensitive(self) -> None:
        assert _label_matches_pattern("MOBILE NUMBER", "phone") is True


class TestLiteralPhraseHeuristic:
    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            ("Project Falcon", True),
            ("42 Maple Street", True),
            ("ZIP", True),
            ("student_names", False),  # a model's invented label, not quoted text
            ("ab", False),  # too short to be worth a review row
            ("x" * 100, False),  # too long to be a phrase
            ("   ", False),
        ],
    )
    def test_only_plausible_literal_text_qualifies(self, term: str, expected: bool) -> None:
        assert _looks_like_literal_phrase(term) is expected

    def test_an_all_lowercase_single_word_is_not_a_literal_phrase(self) -> None:
        assert _looks_like_literal_phrase("grades") is False
