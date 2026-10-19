"""Detector registry: what each pattern finds, and what it deliberately misses."""

from __future__ import annotations

import pytest

from app.matching.patterns import PATTERN_REGISTRY, supported_pattern_names
from .conftest import detect_pattern


# --------------------------------------------------------------------- registry
def test_the_registry_is_closed_and_named():
    names = supported_pattern_names()
    assert names == list(PATTERN_REGISTRY)
    assert {"phone", "email", "ssn", "credit_card", "date", "ip_address", "url",
            "zip_code", "address", "person_name"} <= set(names)
    for spec in PATTERN_REGISTRY.values():
        assert spec.label
        assert spec.keywords, spec.name


@pytest.mark.parametrize("name", supported_pattern_names())
def test_every_detector_can_be_invoked_by_name(name):
    assert detect_pattern(name, "Nothing relevant at all in this sentence.") == []


# ---------------------------------------------------------------- person names
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Student: Alice Johnson", ["Alice Johnson"]),
        ("Name: Dr. Maria del Carmen Rodriguez", ["Maria del Carmen Rodriguez"]),
        ("Brian O'Connor signed the form", ["Brian O'Connor"]),
        ("Jean-Luc Picard commanded the vessel", ["Jean-Luc Picard"]),
        ("Contact Alice B. Johnson now", ["Alice B. Johnson"]),
        ("Vincent van Gogh painted it", ["Vincent van Gogh"]),
        ("He Wei and Li Na attended", ["He Wei", "Li Na"]),
    ],
)
def test_names_are_found(text, expected):
    assert detect_pattern("person_name", text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # A greedy match that over-runs must be narrowed, never discarded: the
        # real name still has to be proposed for review.
        ("Call Alice Johnson today", ["Alice Johnson"]),
        ("Also call Alice Johnson today", ["Alice Johnson"]),
        ("Alice Johnson Grade: A", ["Alice Johnson"]),
        ("Please ask Mary Ann Smith about it", ["Mary Ann Smith"]),
    ],
)
def test_names_survive_neighbouring_words(text, expected):
    assert detect_pattern("person_name", text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Course: Algebra II Grade: A",
        "Home Address: 42 Maple Street",
        "The Springfield High Report Card was issued.",
        "Introduction Conclusion References Appendix",
        "Phone: 555-123-4567",
        "This document is confidential and internal.",
    ],
)
def test_document_furniture_is_not_a_name(text):
    assert detect_pattern("person_name", text) == []


# ---------------------------------------------------------------------- phones
def test_phone_formats():
    found = detect_pattern(
        "phone", "Call (555) 123-4567 or 555-987-6543 or +1 555 222 3333 now"
    )
    assert found == ["(555) 123-4567", "555-987-6543", "+1 555 222 3333"]


def test_bare_digit_runs_are_not_phones():
    # A 10 digit run with no separators is far more often an order or account id.
    assert detect_pattern("phone", "Order 12345 of 5551234567") == []


# ----------------------------------------------------------------------- email
def test_email_addresses():
    assert detect_pattern(
        "email", "write a.johnson@example-school.test or bob+tag@sub.domain.co.uk"
    ) == ["a.johnson@example-school.test", "bob+tag@sub.domain.co.uk"]


# ----------------------------------------------------------------- identifiers
def test_ssn_rejects_invalid_groups():
    assert detect_pattern("ssn", "SSN 123-45-6789") == ["123-45-6789"]
    # 000 and 666 areas are never issued; 9xx is not an SSN.
    assert detect_pattern("ssn", "000-12-3456 666-12-3456 912-34-5678") == []
    assert detect_pattern("ssn", "123-00-3456 123-45-0000") == []


def test_credit_card_requires_a_valid_checksum():
    assert detect_pattern("credit_card", "4111 1111 1111 1111") == ["4111 1111 1111 1111"]
    assert detect_pattern("credit_card", "4111 1111 1111 1112") == []


def test_dates():
    assert detect_pattern("date", "Born 12 March 1990") == ["12 March 1990"]
    assert detect_pattern("date", "1990-03-12") == ["1990-03-12"]
    assert detect_pattern("date", "03/12/1990") == ["03/12/1990"]


def test_ip_addresses():
    assert detect_pattern("ip_address", "host 192.168.0.14") == ["192.168.0.14"]
    assert detect_pattern("ip_address", "999.1.1.1 is not an address") == []


def test_urls():
    assert detect_pattern(
        "url", "see https://example.com/a?b=1 and www.example.org and example.com"
    ) == ["https://example.com/a?b=1", "www.example.org", "example.com"]


def test_zip_codes_need_context_to_avoid_plain_numbers():
    assert detect_pattern("zip_code", "Springfield, IL 62704") == ["IL 62704"]
    assert detect_pattern("zip_code", "62704-1234") == ["62704-1234"]
    # A bare 5 digit number is an amount, an id or a quantity far more often.
    assert detect_pattern("zip_code", "Total 12345 items") == []


def test_addresses():
    assert detect_pattern("address", "42 Maple Street, Springfield, IL 62704") == [
        "42 Maple Street, Springfield, IL 62704"
    ]
    assert detect_pattern("address", "1 Infinite Loop") == ["1 Infinite Loop"]
    # No street number, no confident address match.
    assert detect_pattern("address", "Maple Street is quiet") == []
