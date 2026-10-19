"""Tests for the deterministic instruction parser.

This module is the authority on what the user asked to *preserve*, so the tests
focus on the asymmetric guarantee: a "keep" clause must never lose to a "remove"
clause. Under-redacting is recoverable, destroying data is not.
"""

from __future__ import annotations

import pytest

from app.llm.instruction_parser import (
    KEEP_MARKERS,
    REMOVE_MARKERS,
    conflicts_with_keep,
    exclusion_variants,
    parse_instruction,
)

FULL_INSTRUCTION = (
    "Remove all student names, phone numbers and home addresses, "
    "but keep grades and course names unchanged."
)


class TestMarkerCoverage:
    """Every marker in the public lists must actually be recognised."""

    @pytest.mark.parametrize("marker", sorted(set(REMOVE_MARKERS)))
    def test_remove_markers_produce_a_remove_clause(self, marker: str) -> None:
        parsed = parse_instruction(f"please {marker} the phone numbers")
        assert parsed.keep_clauses == []
        assert any("phone" in clause.lower() for clause in parsed.remove_clauses)

    @pytest.mark.parametrize("marker", sorted(set(KEEP_MARKERS)))
    def test_keep_markers_produce_a_keep_clause(self, marker: str) -> None:
        parsed = parse_instruction(f"remove the phone numbers and {marker} the grades")
        assert any("grade" in clause.lower() for clause in parsed.keep_clauses)

    def test_keep_marker_wins_over_the_remove_prefix_it_contains(self) -> None:
        # "do not remove" contains "remove"; the longer marker must match first.
        parsed = parse_instruction("do not remove the course names")
        assert parsed.pattern_targets == {}
        assert parsed.exclusion_terms == ["course names"]

    def test_parsing_is_case_insensitive(self) -> None:
        parsed = parse_instruction("REMOVE ALL EMAIL ADDRESSES AND PHONE NUMBERS")
        assert set(parsed.pattern_targets) == {"email", "phone"}

    def test_whitespace_is_normalised_before_splitting(self) -> None:
        parsed = parse_instruction("remove\n\nall\tphone   numbers")
        assert parsed.pattern_targets == {"phone": "phone numbers"}
        assert all("\n" not in clause and "\t" not in clause for clause in parsed.remove_clauses)


class TestClauseSplitting:
    def test_the_reference_instruction(self) -> None:
        parsed = parse_instruction(FULL_INSTRUCTION)
        assert parsed.actionable
        assert not parsed.ambiguous
        assert set(parsed.pattern_targets) == {"person_name", "phone", "address"}
        assert parsed.exclusion_terms == ["grades", "course names"]

    def test_text_before_the_first_marker_is_treated_as_a_removal(self) -> None:
        parsed = parse_instruction("the attachment below, redact every phone number")
        assert parsed.remove_clauses
        assert "phone" in parsed.pattern_targets

    def test_several_markers_split_into_several_clauses(self) -> None:
        parsed = parse_instruction(
            "black out the phone numbers. Leave the course names intact."
        )
        assert parsed.pattern_targets == {"phone": "phone numbers"}
        assert parsed.exclusion_terms == ["course names"]

    def test_a_keep_clause_with_no_removal_says_so(self) -> None:
        parsed = parse_instruction("keep the grades and the course names")
        assert parsed.pattern_targets == {}
        assert parsed.exclusion_terms == ["grades", "course names"]
        assert any("only says what to keep" in note for note in parsed.notes)

    def test_two_remove_verbs_yield_both_targets(self) -> None:
        parsed = parse_instruction("remove names remove phone numbers")
        assert set(parsed.pattern_targets) == {"person_name", "phone"}

    def test_a_later_sentence_can_reopen_a_removal(self) -> None:
        parsed = parse_instruction(
            "Please do not remove the reference codes. Remove the phone numbers."
        )
        assert parsed.pattern_targets == {"phone": "phone numbers"}
        assert parsed.exclusion_terms == ["reference codes"]
        assert parsed.actionable


class TestPatternVocabulary:
    def test_email_address_does_not_also_enable_street_address_redaction(self) -> None:
        # The longest keyword must win outright. If "address" survived alongside
        # "email address", an instruction about email would silently delete
        # home addresses.
        parsed = parse_instruction("remove email addresses and home address")
        assert parsed.pattern_targets == {
            "email": "email addresses",
            "address": "home address",
        }

    def test_a_lone_email_keyword_still_maps_only_to_email(self) -> None:
        parsed = parse_instruction("remove every email address")
        assert parsed.pattern_targets == {"email": "email address"}

    def test_hyphenated_and_spelled_out_variants_agree(self) -> None:
        parsed = parse_instruction("remove the credit-card number and the credit card numbers")
        assert list(parsed.pattern_targets) == ["credit_card"]

    def test_plural_and_compound_vocabulary(self) -> None:
        parsed = parse_instruction("remove the names of students and the addresses")
        assert set(parsed.pattern_targets) == {"person_name", "address"}

    def test_date_of_birth_and_ip_vocabulary(self) -> None:
        parsed = parse_instruction("remove all dates of birth and ip addresses")
        assert set(parsed.pattern_targets) == {"date", "ip_address"}

    def test_unsupported_detector_names_are_never_invented(self) -> None:
        parsed = parse_instruction("remove all the biometric signatures")
        assert parsed.pattern_targets == {}
        assert parsed.ambiguous


class TestExactTargets:
    def test_double_quoted_text_is_a_literal_target(self) -> None:
        parsed = parse_instruction('redact "Project Falcon" from the summary')
        assert parsed.exact_targets == ["Project Falcon"]

    def test_curly_quoted_text_is_a_literal_target(self) -> None:
        parsed = parse_instruction("Remove \u201cProject Falcon\u201d and \u201cOperation Kestrel\u201d")
        assert parsed.exact_targets == ["Project Falcon", "Operation Kestrel"]

    def test_a_named_phrase_stops_at_the_next_conjunction(self) -> None:
        parsed = parse_instruction("Remove the phrase Project Falcon and every email address")
        assert parsed.exact_targets == ["Project Falcon"]
        assert "email" in parsed.pattern_targets

    def test_labelled_terms_other_than_phrase(self) -> None:
        parsed = parse_instruction("remove the text Alice Johnson and the code X-99")
        assert parsed.exact_targets == ["Alice Johnson"]

    def test_single_quoted_named_phrase(self) -> None:
        parsed = parse_instruction("Remove the phrase 'Project Falcon' from every page")
        assert parsed.exact_targets == ["Project Falcon"]

    def test_repeated_quoted_terms_are_deduplicated_preserving_case(self) -> None:
        parsed = parse_instruction('redact "Alpha" and also "alpha" again')
        assert parsed.exact_targets == ["Alpha"]

    def test_an_unquoted_proper_noun_is_not_guessed(self) -> None:
        # The parser deliberately refuses to decide on its own where one name ends
        # and the next begins; it asks the user to quote the text instead.
        parsed = parse_instruction("Please remove Alice Johnson and Bob Smith.")
        assert parsed.exact_targets == []
        assert not parsed.actionable
        assert parsed.ambiguous
        assert any("quote the exact text" in note for note in parsed.notes)


class TestKeepBeatsRemove:
    def test_a_colliding_detector_is_blocked_and_the_user_is_told(self) -> None:
        parsed = parse_instruction("Remove all phone numbers but keep the office phone number")
        assert parsed.pattern_targets == {}
        assert parsed.exclusion_terms == ["office phone number"]
        assert any(
            "Ignored 'phone numbers' as a redaction target" in note for note in parsed.notes
        )
        # Nothing to redact is left, so the result must be reported as ambiguous
        # rather than silently producing an empty plan.
        assert not parsed.actionable
        assert parsed.ambiguous

    def test_a_colliding_exact_target_is_dropped(self) -> None:
        parsed = parse_instruction('redact "Springfield High" but do not remove Springfield High')
        assert parsed.exact_targets == []
        assert parsed.exclusion_terms == ["Springfield High"]

    def test_a_non_colliding_keep_clause_leaves_the_target_alone(self) -> None:
        parsed = parse_instruction("Remove phone numbers, keep the grades")
        assert parsed.pattern_targets == {"phone": "phone numbers"}
        assert parsed.exclusion_terms == ["grades"]

    def test_conflicts_use_significant_tokens(self) -> None:
        # "student names" must not collide with "keep course names".
        assert conflicts_with_keep("student names", ["course names"]) is False
        # A bare "names" must collide with it.
        assert conflicts_with_keep("names", ["course names"]) is True

    def test_conflicts_handle_singular_and_plural(self) -> None:
        assert conflicts_with_keep("email address", ["email addresses"]) is True

    def test_conflicts_are_false_for_unrelated_terms(self) -> None:
        assert conflicts_with_keep("phone", ["grades"]) is False
        assert conflicts_with_keep("grades", []) is False
        assert conflicts_with_keep("", ["grades"]) is False
        assert conflicts_with_keep("Algebra II", ["course names"]) is False

    def test_conflicts_match_an_identical_phrase(self) -> None:
        assert conflicts_with_keep("course names", ["course names"]) is True
        assert conflicts_with_keep("Algebra II", ["Algebra II"]) is True


class TestAmbiguity:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_blank_instruction(self, text: str) -> None:
        parsed = parse_instruction(text)
        assert not parsed.actionable
        assert parsed.ambiguous
        assert parsed.notes == ["The instruction is empty."]

    def test_instruction_with_no_verb_is_read_as_a_removal(self) -> None:
        parsed = parse_instruction("all phone numbers in the attachment")
        assert parsed.remove_clauses == ["all phone numbers in the attachment"]
        assert "phone" in parsed.pattern_targets

    def test_polite_but_unactionable_instruction(self) -> None:
        parsed = parse_instruction("Make the document look nicer please.")
        assert not parsed.actionable
        assert parsed.ambiguous
        assert any("Name the data explicitly" in note for note in parsed.notes)

    def test_ambiguous_note_shows_a_usable_example(self) -> None:
        parsed = parse_instruction("clean this up")
        assert parsed.ambiguous
        assert any("remove all phone numbers and email addresses" in note for note in parsed.notes)

    def test_remove_everything_else_is_not_actionable(self) -> None:
        # "everything else" cannot be turned into spans, so the parser refuses
        # rather than deciding on the user's behalf.
        parsed = parse_instruction("Keep grades visible but remove everything else")
        assert not parsed.actionable
        assert parsed.ambiguous
        assert parsed.exclusion_terms == ["grades"]


class TestExclusionVariants:
    def test_plural_and_singular_tokens(self) -> None:
        assert exclusion_variants("course names") == ["course", "courses"]

    def test_already_plural_tokens_lose_the_suffix(self) -> None:
        assert exclusion_variants("grades") == ["grades", "grade"]

    def test_filler_words_are_dropped(self) -> None:
        assert exclusion_variants("the phone number") == ["phone", "phones"]

    def test_two_char_tokens_are_ignored(self) -> None:
        assert exclusion_variants("Algebra II") == ["Algebra", "Algebras"]

    def test_labelled_phrase_keeps_only_its_words(self) -> None:
        assert exclusion_variants("Course: Algebra II") == [
            "Course", "Courses", "Algebra", "Algebras",
        ]

    @pytest.mark.parametrize("term", ["", "a", "the", "of the"])
    def test_terms_with_no_significant_token(self, term: str) -> None:
        assert exclusion_variants(term) == []
