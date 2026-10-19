"""Deterministic parsing of the plain-English instruction.

The local Gemma model is small enough to produce plausible-looking but wrong
plans (it will happily map "grades" onto a credit-card detector, or turn a
"keep" clause into a redaction target). This module is therefore the authority
on the one thing that must never get wrong in the destructive direction: what
the user asked to *preserve*. It also reliably recognises common data-type
vocabulary and explicitly quoted phrases.

The interpreter merges this with the model's output; see
:mod:`app.llm.interpreter`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..matching.patterns import PATTERN_KEYWORDS

# Longest first so "do not remove" wins over "remove".
KEEP_MARKERS = (
    "do not remove", "don't remove", "do not redact", "don't redact",
    "do not delete", "don't delete", "do not touch", "don't touch",
    "do not change", "don't change", "do not alter", "don't alter",
    "leave unchanged", "leave untouched", "must remain", "remain visible",
    "other than", "apart from", "excluding", "as well as", "but not",
    "unchanged", "intact", "visible", "preserve", "maintain", "retain",
    "except", "leave", "keep", "skip", "as is",
)

REMOVE_MARKERS = (
    "black out", "blank out", "cross out", "take out", "cut out", "redaction",
    "removal", "sanitise", "sanitize", "obscure", "censor", "remove", "redact",
    "delete", "erase", "mask", "scrub", "strip", "hide", "omit", "clear",
)

# Words that carry no identifying meaning on their own, so they never become a
# standalone exclusion token.
_FILLER_TOKENS = frozenset(
    """
    name names number numbers no nos num info information details detail data
    value values text field fields entry entries item items all any the of and
    person persons people personal full first last middle given family surname
    other another everything something anything nothing some none
    """.split()
)

_LEADING_FILLER = frozenset(
    """
    the a an all any every each some just only please also both of our their
    his her its my your this that these those remaining rest still should must
    can could would will be been being are is am do does did to for with in on
    at by from out up down over under again further once here there and but or
    keep keeps keeping keep-it leave leaves leaving preserve preserves
    preserved retain retains retained maintain maintains maintained unchanged
    intact visible unmodified untouched same
    """.split()
)

_TRAILING_FILLER = frozenset(
    """
    unchanged intact visible unmodified untouched alone same should must be are
    is remain remains remaining stays stay please too only kept left preserved
    retained maintained as in on at the a an and or but do does did will would
    can could shall may might from with without visible-to-anyone
    """.split()
)

_QUOTED = re.compile(
    r"\u201c([^\u201d]{2,90})\u201d"  # curly double quotes
    r'|"([^"]{2,90})"'  # straight double quotes
)

_NAMED_PHRASE = re.compile(
    r"\b(?:phrase|term|string|word|text|code)\s+"
    r"[\u201c\"']?([A-Za-z0-9][^\n,;.\u201d\"']{1,80})",
    re.IGNORECASE,
)

_SPLIT_ON = re.compile(r"\s*(?:,|;|\band\b|\bor\b|&|\bplus\b|\balso\b|\bas well as\b)\s*", re.IGNORECASE)
_WORD_SPLIT = re.compile(r"[\s]+")


def _build_marker_regex() -> re.Pattern[str]:
    markers = sorted(
        {(m, "keep") for m in KEEP_MARKERS} | {(m, "remove") for m in REMOVE_MARKERS},
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    alternation = "|".join(re.escape(marker) for marker, _ in markers)
    return re.compile(rf"(?<![a-z])({alternation})(?![a-z])", re.IGNORECASE)


_MARKER_RE = _build_marker_regex()
_MARKER_KIND: dict[str, str] = {}
for _marker in KEEP_MARKERS:
    _MARKER_KIND[_marker.lower()] = "keep"
for _marker in REMOVE_MARKERS:
    # "keep" markers were registered first and several share no text with the
    # remove list, so a plain overwrite is safe here.
    _MARKER_KIND.setdefault(_marker.lower(), "remove")


@dataclass(slots=True)
class ParsedInstruction:
    """What the instruction says, decided without a model."""

    exact_targets: list[str] = field(default_factory=list)
    pattern_targets: dict[str, str] = field(default_factory=dict)  # name -> user's words
    exclusion_terms: list[str] = field(default_factory=list)
    remove_clauses: list[str] = field(default_factory=list)
    keep_clauses: list[str] = field(default_factory=list)
    ambiguous: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return bool(self.exact_targets or self.pattern_targets)


def _strip_filler(piece: str) -> str:
    text = piece.strip().strip(".,;:!?").strip()
    if not text:
        return ""
    tokens = _WORD_SPLIT.split(text)
    while tokens and tokens[0].lower().strip(".,;:!?") in _LEADING_FILLER:
        tokens.pop(0)
    while tokens and tokens[-1].lower().strip(".,;:!?") in _TRAILING_FILLER:
        tokens.pop()
    cleaned = " ".join(tokens).strip().strip(".,;:!?").strip()
    return cleaned if len(cleaned) >= 2 else ""


def _split_terms(clause: str) -> list[str]:
    terms: list[str] = []
    for piece in _SPLIT_ON.split(clause):
        cleaned = _strip_filler(piece)
        if cleaned and cleaned.lower() not in _FILLER_TOKENS:
            terms.append(cleaned)
    return terms


def _find_patterns(clause: str) -> dict[str, str]:
    """Map data-type vocabulary in a clause onto registry detector names.

    Longer keywords win: "email address" must yield only ``email``, not also
    ``address``, otherwise an instruction about email would silently enable
    street-address redaction.
    """
    lowered = clause.lower()

    spans: list[tuple[int, int, str, str]] = []
    for name, keywords in PATTERN_KEYWORDS.items():
        for keyword in keywords:
            for match in re.finditer(
                rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", lowered
            ):
                spans.append((match.start(), match.end(), name, keyword))

    survivors: list[tuple[int, int, str, str]] = []
    for span in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        swallowed = any(
            kept[0] <= span[0]
            and span[1] <= kept[1]
            and kept[2] != span[2]
            and (kept[1] - kept[0]) > (span[1] - span[0])
            for kept in survivors
        )
        if not swallowed:
            survivors.append(span)

    found: dict[str, str] = {}
    for start, end, name, keyword in sorted(survivors, key=lambda s: s[1] - s[0], reverse=True):
        if name not in found or (end - start) > len(found[name]):
            found[name] = keyword
    return found


def _find_quoted(clause: str) -> list[str]:
    terms: list[str] = []
    for match in _QUOTED.finditer(clause):
        text = next((g for g in match.groups() if g), "").strip()
        if len(text) >= 2:
            terms.append(text)
    return terms


def _find_named(clause: str) -> list[str]:
    terms: list[str] = []
    for match in _NAMED_PHRASE.finditer(clause):
        raw = match.group(1).strip()
        # "the phrase Project Falcon and every email" -> keep only the phrase.
        first = re.split(r"\s+(?:and|or|but|then|,)\s+", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = _strip_filler(first)
        if cleaned and len(cleaned) >= 3 and cleaned.lower() not in _FILLER_TOKENS:
            terms.append(cleaned)
    return terms


def _dedupe(values: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        seen.setdefault(value.lower().strip(), value.strip())
    return list(seen.values())


def parse_instruction(instruction: str) -> ParsedInstruction:
    """Split an instruction into remove-clauses and keep-clauses, then mine both."""
    parsed = ParsedInstruction()
    text = re.sub(r"\s+", " ", instruction or "").strip()
    if not text:
        parsed.ambiguous = True
        parsed.notes.append("The instruction is empty.")
        return parsed

    markers = list(_MARKER_RE.finditer(text))

    if not markers:
        # No verb at all: treat the whole thing as a description of what to remove.
        parsed.remove_clauses = [text]
        parsed.keep_clauses = []
    else:
        # Text before the first marker defaults to "remove" intent.
        preamble = text[: markers[0].start()].strip()
        if preamble and len(preamble.split()) >= 2:
            parsed.remove_clauses.append(preamble)

        for index, match in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            clause = text[match.end() : end].strip().strip(",;:")
            kind = _MARKER_KIND.get(match.group(1).lower(), "remove")
            if not clause:
                continue
            if kind == "keep":
                parsed.keep_clauses.append(clause)
            else:
                parsed.remove_clauses.append(clause)

    for clause in parsed.remove_clauses:
        parsed.pattern_targets.update(_find_patterns(clause))
        parsed.exact_targets.extend(_find_quoted(clause))
        parsed.exact_targets.extend(_find_named(clause))

    for clause in parsed.keep_clauses:
        parsed.exclusion_terms.extend(_split_terms(clause))

    # A term the user asked to keep always wins over the same term asked to
    # remove; under-redacting is recoverable, destroying data is not.
    parsed.exact_targets = [
        target
        for target in _dedupe(parsed.exact_targets)
        if not conflicts_with_keep(target, parsed.exclusion_terms)
    ]
    blocked_patterns = {
        name: label
        for name, label in parsed.pattern_targets.items()
        if conflicts_with_keep(label, parsed.exclusion_terms)
    }
    for name, label in blocked_patterns.items():
        parsed.pattern_targets.pop(name, None)
        parsed.notes.append(
            f"Ignored {label!r} as a redaction target because the instruction also "
            "asks to keep it."
        )

    parsed.exclusion_terms = _dedupe(parsed.exclusion_terms)

    if not parsed.actionable:
        parsed.ambiguous = True
        parsed.notes.append(
            "No redaction target could be determined from this wording. Name the data "
            "explicitly, for example 'remove all phone numbers and email addresses', "
            'or quote the exact text you want removed.'
        )

    if parsed.exclusion_terms and not parsed.remove_clauses:
        parsed.notes.append(
            "The instruction only says what to keep. Add what should be removed."
        )

    return parsed


def exclusion_variants(term: str) -> list[str]:
    """Search variants for an exclusion term.

    "course names" must protect a document line reading "Course: Algebra II", so
    the significant tokens are also used, in singular and plural form. The full
    phrase is always tried first by the caller; these are the fallback.
    """
    variants: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", term):
        lowered = token.lower()
        if lowered in _FILLER_TOKENS or lowered in _LEADING_FILLER:
            continue
        variants.append(token)
        if lowered.endswith("s") and len(lowered) > 3:
            variants.append(token[:-1])
        elif not lowered.endswith("s"):
            variants.append(token + "s")
    return _dedupe(variants)


def _significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{3,}", text.lower())
        if token not in _FILLER_TOKENS
    }


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def conflicts_with_keep(term: str, kept_terms: list[str] | set[str]) -> bool:
    """True when ``term`` asks to touch something the user asked to preserve.

    Uses significant tokens so "student names" does not collide with a
    "keep course names" clause, while a bare "names" or "course names" does.
    """
    if not term.strip():
        return False
    significant = _significant_tokens(term)
    raw = _raw_tokens(term)

    for kept in kept_terms:
        kept = kept.strip()
        if not kept:
            continue
        if term.lower() == kept.lower():
            return True
        kept_significant = _significant_tokens(kept)
        if significant and kept_significant:
            if significant & kept_significant:
                return True
            continue
        kept_raw = _raw_tokens(kept)
        if raw and kept_raw and (raw <= kept_raw or kept_raw <= raw):
            return True
    return False
