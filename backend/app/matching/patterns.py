"""Closed registry of supported detectors.

The local model may only ever select a pattern *by name* from this registry; it
cannot inject a regular expression. That keeps matching deterministic and
reviewable, and makes ReDoS from model output impossible.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# --------------------------------------------------------------- definitions
_STREET_SUFFIX = (
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
    r"Way|Place|Pl|Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Square|Sq|"
    r"Trail|Trl|Loop|Path|Alley|Aly|Row)"
)
_UNIT = r"(?:\s*(?:Apt|Apartment|Unit|Suite|Ste|Floor|Flm|Rm|Room|#)\.?\s*[A-Za-z0-9\-]+)?"
_CITY_STATE_ZIP = r"(?:,\s*[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_NAME_LABELS = (
    r"(?:Name|Student|Patient|Employee|Applicant|Candidate|Contact|Parent|"
    r"Guardian|Author|Owner|Recipient|Sender|From|To|Attn|Signed|Prepared by|"
    r"Mr|Mrs|Ms|Miss|Dr|Prof|Doctor|Rev)"
)

_TITLE_CASE_WORD = r"[A-Z][a-z'’\-]{1,15}"

# A name token must be a whole word and must not be a "Label:" that follows the
# name - without these guards "Student: Alice Johnson Phone: ..." yields
# "Alice Johnson Phone".
_NAME_WORD = (
    r"[A-Z](?:'[A-Z][a-z’\-]{0,14}|[a-z’]+(?:-[a-z’]+)*)(?:-[A-Z][a-z’]+(?:-[a-z’]+)*)*"
)
_NAME_PARTICLE = r"(?:de|van|von|del|la|le|di|da|dos|der|den|ter|bin|al)"
_NAME_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Doctor|Rev|Hon|Sr|Jr)"
_NAME_TOKEN = (
    rf"(?:{_NAME_WORD}(?![a-z'’\-])(?!\s*:)"
    rf"|[A-Z]\.(?!\s*:)"
    rf"|{_NAME_PARTICLE}(?![a-z]))"
)

PHONE_RE = re.compile(
    r"(?<![\d\w])"
    r"(?:\+?1[\s.\-]{0,2})?"
    r"(?:\(\d{3}\)[\s.\-]{0,2}|\d{3}[\s.\-])"
    r"\d{3}[\s.\-]?\d{4}"
    r"(?![\d])"
    r"(?:\s*(?:ext\.?|extension|x|#)\s*\d{1,5})?",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9])"
)

SSN_RE = re.compile(
    r"(?<![\d\-])"
    r"(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}"
    r"(?![\d\-])"
)

CREDIT_CARD_RE = re.compile(r"(?<![\d])(?:\d[ \-]?){12,18}\d(?![\d])")

DATE_RE = re.compile(
    rf"(?<![\w/])"
    rf"(?:"
    rf"\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf"|\d{{1,2}}\s+{_MONTHS}\.?\s+\d{{4}}"
    rf"|{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|{_MONTHS}\.?,?\s+\d{{4}}"
    rf")"
    rf"(?![\w/-])",
    re.IGNORECASE,
)

IP_RE = re.compile(
    r"(?<![\d.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?![\d.])"
)

URL_RE = re.compile(
    r"(?<![A-Za-z0-9/])"
    r"(?:https?://[^\s<>\"')\]}]+|www\.[^\s<>\"')\]}]+)"
    r"|(?:[A-Za-z0-9\-]+\.)+(?:com|org|net|edu|gov|io|co|us|uk|ca|au|de|fr|info|biz)\b"
    r"(?:/[^\s<>\"')\]}]*)?",
    re.IGNORECASE,
)

ZIP_RE = re.compile(
    r"(?<![\d\-])\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"
    r"|(?<![\d\-])\b\d{5}-\d{4}\b(?![\d\-])"
)

ADDRESS_RE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"\d{{1,6}}\s+(?:{_TITLE_CASE_WORD}\s+){{0,3}}?{_STREET_SUFFIX}\.?"
    rf"{_UNIT}{_CITY_STATE_ZIP}"
    rf"(?![A-Za-z0-9])"
)

PERSON_NAME_LABELLED_RE = re.compile(
    rf"\b{_NAME_LABELS}\b\s*[:.\-]?\s+"
    rf"(?:{_NAME_TITLE}\.?\s+)?"
    rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})"
)

PERSON_NAME_LOOSE_RE = re.compile(
    rf"(?<![A-Za-z])"
    rf"({_NAME_TOKEN}"
    rf"(?:\s+{_NAME_TOKEN}){{1,2}})"
    rf"(?![a-z])"
)

# Words that commonly appear capitalised in documents but are not people.
NON_NAME_WORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for from with without
    January February March April May June July August September October November December
    Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday
    Mon Tue Wed Thu Fri Sat Sun
    North South East West Northeast Northwest Southeast Southwest
    Street Road Avenue Boulevard Lane Drive Court Way Place Terrace Circle Parkway
    Highway Square Trail Loop Path Alley Row Apt Unit Suite Floor Room
    Course Class Grade Semester Term Section Chapter Unit Lesson Module
    Student Patient Employee Applicant Candidate Contact Parent Guardian Author
    Owner Recipient Sender Name Address Phone Email Fax Mobile Date Birth
    School College University Academy Institute Department Faculty District
    Report Card Transcript Record Records File Document Page Pages Total Summary
    Introduction Conclusion References Appendix Table Figure List Notes Note
    Confidential Internal Private Draft Final Approved Pending Reviewed
    Project Program Initiative Task Item Order Invoice Account Number Id
    Mr Mrs Ms Miss Dr Prof Doctor Rev Sir Madam
    City State Country Zip Postal Code Region County Province
    Monday Morning Afternoon Evening Night
    Grade Point Average Gpa Ssn Idn Tax Social Security
    Home Work Business Personal Emergency Primary Secondary
    January Springfield
    Also Additionally However Therefore Furthermore Moreover Meanwhile Although
    Because Since While Whereas Thus Hence Otherwise Instead Likewise Similarly
    This That These Those Here There Where When What Which Who Whom Whose Why
    All Any Each Every Some Both Either Neither Many Much Few Several Various Such
    Our Your Their His Her Its We You They Them Us Me
    Is Are Was Were Be Been Being Am Do Does Did Done Have Has Had Having
    Can Could Shall Should Would Must Might Ought Need
    Let Make Made Take Took Give Gave Given Get Got Put Keep Kept Send Sent
    Receive Received Provide Provided Include Included Exclude Excluded
    Remove Removed Delete Deleted Submit Submitted Reply Replied
    Confirm Confirmed Ensure Verify Verified Update Updated Sign
    Attach Attached Enclose Enclosed Refer Referred Notify Contact Call Called
    Please Dear Hello Hi Regards Sincerely Thank Thanks Welcome
    Below Above Following Previous Current
    """.split()
)

# Instruction vocabulary that maps to a detector. Ordered longest-first so that
# "phone number" wins over "number".
PATTERN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "person_name": (
        "person name", "personal name", "people name", "full name", "names", "name",
        "student", "students", "patient", "patients", "employee", "employees",
        "applicant", "candidate", "individual", "people", "person", "who",
        "guardian", "parent", "author", "signatory", "contact person",
    ),
    "phone": (
        "phone number", "phone numbers", "telephone", "phone", "mobile", "cell",
        "fax number", "fax", "contact number", "msisdn",
    ),
    "email": ("email address", "email addresses", "e-mail", "email", "emails", "mail address"),
    "address": (
        "home address", "home addresses", "street address", "postal address",
        "mailing address", "residential address", "physical address", "address",
        "addresses", "location of residence",
    ),
    "ssn": (
        "social security number", "social security numbers", "ssn", "ssns",
        "national insurance number", "tax id", "taxpayer id",
    ),
    "credit_card": (
        "credit card", "credit cards", "card number", "card numbers", "debit card",
        "payment card", "pan", "credit-card",
    ),
    "date": ("date", "dates", "date of birth", "dob", "birth date", "birthday", "timestamp"),
    "ip_address": ("ip address", "ip addresses", "ip", "ipv4", "network address"),
    "url": ("url", "urls", "website", "websites", "web address", "link", "links", "domain"),
    "zip_code": ("zip code", "zip codes", "zipcode", "postal code", "postal codes", "postcode"),
}


# (accepted text, its offset inside the raw match) or None to reject the match.
RefineResult = tuple[str, int] | None


@dataclass(frozen=True, slots=True)
class RegexRule:
    """One regular expression plus the confidence and refiner to apply to it.

    A refiner returns ``(text, offset_within_the_match)`` to accept a possibly
    narrowed span, or ``None`` to reject the match outright. Narrowing matters:
    a greedy name match that over-runs into a neighbouring word must yield the
    name rather than be discarded, otherwise the real value is silently missed.
    """

    regex: re.Pattern[str]
    group: int
    confidence: float
    refine: Callable[[str], RefineResult] | None = None


@dataclass(frozen=True, slots=True)
class PatternSpec:
    name: str
    label: str
    description: str
    rules: tuple[RegexRule, ...]
    keywords: tuple[str, ...]
    caveat: str | None = None

    @property
    def min_confidence(self) -> float:
        return min(rule.confidence for rule in self.rules)


def _luhn_ok(candidate: str) -> bool:
    digits = re.sub(r"[\s\-]", "", candidate)
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, ch in enumerate(reversed(digits)):
        value = int(ch)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _refine_luhn(candidate: str) -> RefineResult:
    return (candidate, 0) if _luhn_ok(candidate) else None


_NON_NAME_LOWER = frozenset(word.lower() for word in NON_NAME_WORDS)

_TOKEN_RE = re.compile(r"\S+")


def _is_non_name_token(token: str) -> bool:
    cleaned = token.strip(".,;:'’\"-")
    return bool(cleaned) and cleaned.lower() in _NON_NAME_LOWER


def _longest_name_run(candidate: str, *, min_tokens: int) -> RefineResult:
    """Widest run of consecutive tokens that all look like name parts.

    Greedy name regexes over-run in both directions: "Call Alice Johnson" picks
    up the sentence-initial verb, and "Alice Johnson Grade" picks up the label
    that follows. Rejecting the whole match would silently drop a real name, so
    the match is narrowed to the plausible part instead.
    """
    spans = [match.span() for match in _TOKEN_RE.finditer(candidate)]
    if len(spans) < min_tokens:
        return None

    blocked = [_is_non_name_token(candidate[a:b]) for a, b in spans]
    best: tuple[int, int] | None = None
    best_length = 0
    index = 0
    while index < len(blocked):
        if blocked[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(blocked) and not blocked[end + 1]:
            end += 1
        length = end - index + 1
        if length >= min_tokens and length > best_length:
            best, best_length = (index, end), length
        index = end + 1

    if best is None:
        return None
    start = spans[best[0]][0]
    return candidate[start : spans[best[1]][1]], start


def _refine_loose_name(candidate: str) -> RefineResult:
    """The loose heuristic needs at least a first and a last name."""
    return _longest_name_run(candidate, min_tokens=2)


def _refine_labelled_name(candidate: str) -> RefineResult:
    """A label such as "Name:" already vouches for the value, so one token is enough."""
    refined = _longest_name_run(candidate, min_tokens=2)
    if refined is not None:
        return refined
    for match in _TOKEN_RE.finditer(candidate):
        if not _is_non_name_token(match.group(0)):
            return match.group(0), match.start()
    return None


PATTERN_REGISTRY: dict[str, PatternSpec] = {
    spec.name: spec
    for spec in (
        PatternSpec(
            name="phone",
            label="Phone number",
            description="North-American style landline, mobile and fax numbers, with optional country code and extension.",
            rules=(RegexRule(PHONE_RE, 0, 0.9),),
            keywords=PATTERN_KEYWORDS["phone"],
        ),
        PatternSpec(
            name="email",
            label="Email address",
            description="RFC-ish email addresses.",
            rules=(RegexRule(EMAIL_RE, 0, 0.95),),
            keywords=PATTERN_KEYWORDS["email"],
        ),
        PatternSpec(
            name="ssn",
            label="Social security number",
            description="US SSN in dashed form (NNN-NN-NNNN), rejecting invalid area/group/serial groups.",
            rules=(RegexRule(SSN_RE, 0, 0.92),),
            keywords=PATTERN_KEYWORDS["ssn"],
            caveat="Only the dashed form is detected. Spaced or unseparated SSNs are missed.",
        ),
        PatternSpec(
            name="credit_card",
            label="Payment card number",
            description="13-19 digit card numbers that pass the Luhn checksum.",
            rules=(RegexRule(CREDIT_CARD_RE, 0, 0.8, _refine_luhn),),
            keywords=PATTERN_KEYWORDS["credit_card"],
        ),
        PatternSpec(
            name="date",
            label="Date",
            description="Numeric and month-name dates in common Western formats.",
            rules=(RegexRule(DATE_RE, 0, 0.7),),
            keywords=PATTERN_KEYWORDS["date"],
            caveat="Date detection is broad; course codes and other numeric pairs can look like dates.",
        ),
        PatternSpec(
            name="ip_address",
            label="IPv4 address",
            description="Dotted-quad IPv4 addresses with valid octets.",
            rules=(RegexRule(IP_RE, 0, 0.75),),
            keywords=PATTERN_KEYWORDS["ip_address"],
            caveat="Software version numbers such as 1.2.3.4 also match.",
        ),
        PatternSpec(
            name="url",
            label="URL or domain",
            description="http(s) URLs, www. hosts and bare common-TLD domains.",
            rules=(RegexRule(URL_RE, 0, 0.7),),
            keywords=PATTERN_KEYWORDS["url"],
        ),
        PatternSpec(
            name="zip_code",
            label="Postal / ZIP code",
            description="US ZIP or ZIP+4, only when preceded by a state abbreviation or in ZIP+4 form.",
            rules=(RegexRule(ZIP_RE, 0, 0.7),),
            keywords=PATTERN_KEYWORDS["zip_code"],
            caveat="Bare 5-digit numbers are not treated as ZIP codes, to limit false positives.",
        ),
        PatternSpec(
            name="address",
            label="Street address",
            description="Street number plus street name and suffix, optionally followed by unit, city, state and ZIP.",
            rules=(RegexRule(ADDRESS_RE, 0, 0.65),),
            keywords=PATTERN_KEYWORDS["address"],
            caveat=(
                "Address detection is heuristic. It requires a numeric street number, so "
                "addresses written without one are missed, and phrases like '2 Dr' can false-positive."
            ),
        ),
        PatternSpec(
            name="person_name",
            label="Person name",
            description=(
                "Names introduced by a label (Name:, Student:, Dr, ...) at higher confidence, "
                "plus a looser two-to-three capitalised-word heuristic at lower confidence."
            ),
            rules=(
                RegexRule(PERSON_NAME_LABELLED_RE, 1, 0.75, _refine_labelled_name),
                RegexRule(PERSON_NAME_LOOSE_RE, 1, 0.35, _refine_loose_name),
            ),
            keywords=PATTERN_KEYWORDS["person_name"],
            caveat=(
                "Name detection is the least reliable detector here. The loose heuristic "
                "proposes any capitalised word sequence, so review these candidates carefully."
            ),
        ),
    )
}


def supported_pattern_names() -> list[str]:
    return list(PATTERN_REGISTRY)

