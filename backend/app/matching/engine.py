"""Deterministic detection engine.

Everything here is reproducible: the same document plus the same validated plan
always yields the same detections and the same ids. The model never supplies
text to redact directly - it can only name detectors from the closed registry, or
propose literal terms that this engine must actually find in the document.
Terms it proposes that do not occur are dropped and reported back.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from ..extraction.base import ExtractedDocument, FileKind, Segment
from ..logging_setup import get_logger
from .patterns import PATTERN_REGISTRY

logger = get_logger("matching")

CONTEXT_CHARS = 48
MAX_SEGMENT_CHARS = 2_000_000

# How far past the matched term an exclusion may extend its protection. Bounds
# the damage when an exclusion term is a common word: it can protect the rest of
# its clause, but not an arbitrary amount of surrounding text.
MAX_CLAUSE_EXTENSION = 120

# A new "Label:" starts a fresh clause, which bounds how far an exclusion may
# extend its protection.
_LABEL_BOUNDARY_RE = re.compile(r"(?:^|(?<=\s))[A-Z][A-Za-z]{2,24}\s*:\s")


# --------------------------------------------------------------------- models
class Target(BaseModel):
    kind: Literal["exact", "pattern"]
    term: str = ""
    pattern: str | None = None
    category: str = "unspecified"
    source: Literal["instruction", "model"] = "instruction"

    @property
    def identity(self) -> str:
        return f"{self.kind}:{self.term.lower().strip()}:{self.pattern or ''}"


class Exclusion(BaseModel):
    kind: Literal["exact", "pattern"]
    term: str = ""
    pattern: str | None = None
    category: str = "user_protected"
    source: Literal["instruction", "model"] = "instruction"
    protect_clause: bool = True
    # Resolved by `detect()`: the full phrase when it occurs verbatim, otherwise
    # its significant tokens so "course names" still protects "Course: Algebra II".
    match_terms: list[str] = Field(default_factory=list)

    def effective_terms(self) -> list[str]:
        if self.match_terms:
            return self.match_terms
        return [self.term] if self.term.strip() else []


class MatchPlan(BaseModel):
    targets: list[Target] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)


class DetectionLocation(BaseModel):
    document_id: str
    document_name: str
    file_kind: FileKind
    segment_index: int
    page: int | None = None
    paragraph_path: str | None = None
    char_start: int
    char_end: int


class Detection(BaseModel):
    detection_id: str
    matched_text: str
    category: str
    strategy: Literal["exact", "pattern"]
    pattern_name: str | None = None
    source: Literal["instruction", "model", "both"] = "instruction"
    confidence: float = Field(ge=0.0, le=1.0)
    heuristic: bool = False
    fuzzy: bool = False
    occurrence: int = 1
    location: DetectionLocation
    context_before: str = ""
    context_after: str = ""


class MatchResult(BaseModel):
    detections: list[Detection] = Field(default_factory=list)
    unmatched_exact_terms: list[str] = Field(default_factory=list)
    suppressed_by_exclusion: int = 0
    per_pattern_counts: dict[str, int] = Field(default_factory=dict)
    # document_id -> the exclusions with their resolved search terms, so the API
    # can show the user exactly what text each "keep" request protected.
    resolved_exclusions: dict[str, list[Exclusion]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------- helpers
def detection_id(
    *,
    document_id: str,
    segment_index: int,
    char_start: int,
    char_end: int,
    matched_text: str,
    strategy: str,
    pattern_name: str | None,
) -> str:
    """Stable id: a pure function of the located span, not of the session."""
    payload = "\x1f".join(
        (
            document_id,
            str(segment_index),
            str(char_start),
            str(char_end),
            matched_text,
            strategy,
            pattern_name or "",
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"det_{digest}"


def _normalise_term(term: str) -> str:
    """Case- and whitespace-insensitive form, used to group occurrences.

    The same requested phrase can be matched verbatim, in different case, or
    across a line break; those are occurrences of one thing and must be numbered
    as such rather than each starting again at 1.
    """
    return " ".join(term.split()).lower()


def _literal_ranges(segment_text: str, term: str) -> list[tuple[int, int]]:
    """Case-sensitive literal occurrences."""
    if not term:
        return []
    ranges: list[tuple[int, int]] = []
    start = segment_text.find(term)
    while start != -1:
        ranges.append((start, start + len(term)))
        start = segment_text.find(term, start + len(term))
    return ranges


def _case_insensitive_ranges(segment_text: str, term: str) -> list[tuple[int, int]]:
    if not term:
        return []
    lowered = segment_text.lower()
    needle = term.lower()
    ranges: list[tuple[int, int]] = []
    start = lowered.find(needle)
    while start != -1:
        ranges.append((start, start + len(needle)))
        start = lowered.find(needle, start + len(needle))
    return ranges


def _whitespace_flexible_regex(term: str) -> re.Pattern[str] | None:
    """Regex tolerating arbitrary whitespace between the term's words.

    PDF extraction joins words with single spaces, so a term quoted from a
    layout with line breaks or wide gaps would otherwise never match.
    """
    tokens = term.split()
    if len(tokens) < 2:
        return None
    body = r"\s+".join(re.escape(token) for token in tokens)
    return re.compile(body, re.IGNORECASE)


def _clause_end(segment: Segment, position: int) -> int:
    """End of the clause containing ``position``.

    Bounded by the next ``Label:``, by the end of the extracted line the position
    sits on (PDF segments are whole pages), and by a hard character cap.
    """
    limit = min(len(segment.text), position + 1 + MAX_CLAUSE_EXTENSION)

    boundary = _LABEL_BOUNDARY_RE.search(segment.text, position)
    if boundary and position < boundary.start() < limit:
        limit = boundary.start()

    for start, end in segment.line_ranges:
        if start <= position < end:
            limit = min(limit, end)
            break

    return limit


class _Candidate:
    __slots__ = (
        "start", "end", "text", "strategy", "pattern_name", "category",
        "confidence", "source", "heuristic", "fuzzy", "term_key",
    )

    def __init__(
        self,
        start: int,
        end: int,
        text: str,
        *,
        strategy: str,
        pattern_name: str | None,
        category: str,
        confidence: float,
        source: str,
        heuristic: bool = False,
        fuzzy: bool = False,
        term_key: str | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.strategy = strategy
        self.pattern_name = pattern_name
        self.category = category
        self.confidence = confidence
        self.source = source
        self.heuristic = heuristic
        self.fuzzy = fuzzy
        # Groups occurrences of the same requested thing so numbering survives a
        # match that differs from the term by case or by internal whitespace.
        self.term_key = term_key


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(r_start < end and start < r_end for r_start, r_end in ranges)


def _exclusion_ranges(segment: Segment, exclusions: list[Exclusion]) -> list[tuple[int, int]]:
    """Protected character ranges inside one segment."""
    protected: list[tuple[int, int]] = []
    for exclusion in exclusions:
        hits: list[tuple[int, int]] = []

        if exclusion.kind == "exact":
            for term in exclusion.effective_terms():
                term = term.strip()
                if not term:
                    continue
                # Cumulative on purpose: protecting text that turns out not to be
                # sensitive costs the user a review click, while failing to protect
                # a line-broken occurrence of a "keep" phrase destroys data.
                found = _literal_ranges(segment.text, term)
                found += [
                    span
                    for span in _case_insensitive_ranges(segment.text, term)
                    if span not in found
                ]
                flexible = _whitespace_flexible_regex(term)
                if flexible:
                    found += [
                        (match.start(), match.end()) for match in flexible.finditer(segment.text)
                    ]
                hits.extend(found)
        else:
            spec = PATTERN_REGISTRY.get(exclusion.pattern or "")
            if spec is not None:
                for rule in spec.rules:
                    for match in rule.regex.finditer(segment.text):
                        if match.group(rule.group).strip():
                            hits.append(
                                (match.start(rule.group), match.end(rule.group))
                            )

        for start, end in hits:
            if exclusion.protect_clause:
                end = max(end, _clause_end(segment, end - 1 if end > start else start))
            protected.append((start, end))

    protected.sort()
    return protected


def _phrase_occurs(documents: list[ExtractedDocument], term: str, scope: set[str] | None) -> bool:
    needle = term.strip().lower()
    if not needle:
        return False
    for document in documents:
        if scope and document.document_id not in scope:
            continue
        for segment in document.segments:
            if needle in segment.text.lower():
                return True
    return False


def _resolve_exclusions(
    documents: list[ExtractedDocument],
    exclusions: list[Exclusion],
    scope: set[str] | None,
) -> list[Exclusion]:
    """Pick the narrowest search terms that still protect what the user named.

    The verbatim phrase is preferred. Token-level variants are only used when the
    phrase itself appears nowhere, which keeps protection from spreading further
    than the user intended.
    """
    from ..llm.instruction_parser import exclusion_variants

    resolved: list[Exclusion] = []
    for exclusion in exclusions:
        if exclusion.kind != "exact" or exclusion.match_terms:
            resolved.append(exclusion)
            continue
        term = exclusion.term.strip()
        if not term:
            continue
        if _phrase_occurs(documents, term, scope):
            resolved.append(exclusion.model_copy(update={"match_terms": [term]}))
            continue
        variants = exclusion_variants(term)
        usable = [v for v in variants if _phrase_occurs(documents, v, scope)]
        resolved.append(
            exclusion.model_copy(update={"match_terms": usable or [term]})
        )
    return resolved


def _collect_pattern_candidates(
    segment: Segment, target: Target
) -> list[_Candidate]:
    spec = PATTERN_REGISTRY.get(target.pattern or "")
    if spec is None:
        return []

    found: list[_Candidate] = []
    for rule in spec.rules:
        for match in rule.regex.finditer(segment.text):
            try:
                text = match.group(rule.group)
            except IndexError:  # pragma: no cover - defensive
                continue
            if not text or not text.strip():
                continue

            offset = 0
            if rule.refine is not None:
                refined = rule.refine(text)
                if refined is None:
                    continue
                text, offset = refined
                if not text.strip():
                    continue

            start = match.start(rule.group) + offset
            end = start + len(text)
            found.append(
                _Candidate(
                    start,
                    end,
                    text,
                    strategy="pattern",
                    pattern_name=spec.name,
                    category=target.category or spec.label,
                    confidence=rule.confidence,
                    source=target.source,
                    heuristic=rule.confidence < 0.5,
                    term_key=f"pattern|{spec.name}",
                )
            )
    return found


def _collect_exact_candidates(
    segment: Segment, target: Target, *, case_sensitive: bool
) -> list[_Candidate]:
    term = target.term.strip()
    if not term:
        return []
    hits = (
        _literal_ranges(segment.text, term)
        if case_sensitive
        else _case_insensitive_ranges(segment.text, term)
    )
    return [
        _Candidate(
            start,
            end,
            segment.text[start:end],
            strategy="exact",
            pattern_name=None,
            category=target.category,
            confidence=0.95 if case_sensitive else 0.7,
            source=target.source,
            term_key=f"exact|{_normalise_term(term)}",
        )
        for start, end in hits
    ]


def _exact_candidates(segment: Segment, target: Target) -> list[_Candidate]:
    """Every occurrence of a literal term, most certain first.

    The three passes are cumulative, not a fallback chain. A phrase the user
    quoted can appear both on one line and split across a line break in the same
    PDF, and in both cases and cases that differ only by capitalisation; stopping
    at the first pass that found anything would silently miss the rest.
    """
    found = _collect_exact_candidates(segment, target, case_sensitive=True)
    occupied = [(candidate.start, candidate.end) for candidate in found]

    for candidate in _collect_exact_candidates(segment, target, case_sensitive=False):
        candidate.fuzzy = True
        if _overlaps(candidate.start, candidate.end, occupied):
            continue
        found.append(candidate)
        occupied.append((candidate.start, candidate.end))

    flexible = _whitespace_flexible_regex(target.term.strip())
    if flexible is None:
        return found

    key = f"exact|{_normalise_term(target.term)}"
    for match in flexible.finditer(segment.text):
        start, end = match.start(), match.end()
        if _overlaps(start, end, occupied):
            continue
        found.append(
            _Candidate(
                start,
                end,
                segment.text[start:end],
                strategy="exact",
                pattern_name=None,
                category=target.category,
                confidence=0.7,
                source=target.source,
                fuzzy=True,
                term_key=key,
            )
        )
        occupied.append((start, end))
    return found


# ---------------------------------------------------------------- main entry
def detect(
    documents: list[ExtractedDocument],
    plan: MatchPlan,
    *,
    document_scope: list[str] | None = None,
) -> MatchResult:
    """Locate every candidate span described by ``plan`` inside ``documents``."""
    warnings: list[str] = []
    detections: list[Detection] = []
    unmatched: list[str] = []
    suppressed = 0
    per_pattern: dict[str, int] = defaultdict(int)

    scope = set(document_scope) if document_scope else None
    exact_targets = [t for t in plan.targets if t.kind == "exact" and t.term.strip()]
    pattern_targets = [t for t in plan.targets if t.kind == "pattern"]

    for spec_name in {t.pattern for t in pattern_targets}:
        if spec_name not in PATTERN_REGISTRY:
            warnings.append(f"Ignored unsupported detector name {spec_name!r}.")

    pattern_targets = [t for t in pattern_targets if t.pattern in PATTERN_REGISTRY]

    resolved_by_document: dict[str, list[Exclusion]] = {}

    for document in documents:
        if scope and document.document_id not in scope:
            continue
        if not document.has_text_layer:
            continue

        exclusions = _resolve_exclusions([document], plan.exclusions, None)
        resolved_by_document[document.document_id] = exclusions

        # Numbered per document, not per segment: "occurrence 2 of 3" has to mean
        # the second time this appears anywhere in the file, including other pages.
        occurrence: dict[str, int] = defaultdict(int)

        for segment in document.segments:
            if len(segment.text) > MAX_SEGMENT_CHARS:  # pragma: no cover - guarded upstream
                warnings.append(
                    f"Skipped an oversized text block in {document.document_name}."
                )
                continue

            protected = _exclusion_ranges(segment, exclusions)
            candidates: list[_Candidate] = []

            for target in pattern_targets:
                candidates.extend(_collect_pattern_candidates(segment, target))

            for target in exact_targets:
                candidates.extend(_exact_candidates(segment, target))

            # Longest-first at each start position, then highest confidence.
            candidates.sort(key=lambda c: (c.start, -(c.end - c.start), -c.confidence))

            accepted: list[_Candidate] = []
            max_end = -1

            for candidate in candidates:
                if candidate.end <= candidate.start:
                    continue
                if _overlaps(candidate.start, candidate.end, protected):
                    suppressed += 1
                    continue
                if candidate.start < max_end:
                    # Nested inside or overlapping an already accepted, longer or
                    # more confident candidate.
                    continue
                accepted.append(candidate)
                max_end = candidate.end

            for candidate in accepted:
                key = candidate.term_key or (
                    f"{candidate.strategy}|{candidate.pattern_name or candidate.text.lower()}"
                )
                occurrence[key] += 1
                start, end = candidate.start, candidate.end
                detections.append(
                    Detection(
                        detection_id=detection_id(
                            document_id=document.document_id,
                            segment_index=segment.segment_index,
                            char_start=start,
                            char_end=end,
                            matched_text=candidate.text,
                            strategy=candidate.strategy,
                            pattern_name=candidate.pattern_name,
                        ),
                        matched_text=candidate.text,
                        category=candidate.category,
                        strategy=candidate.strategy,
                        pattern_name=candidate.pattern_name,
                        source=candidate.source,  # type: ignore[arg-type]
                        confidence=candidate.confidence,
                        heuristic=candidate.heuristic,
                        fuzzy=candidate.fuzzy,
                        occurrence=occurrence[key],
                        location=DetectionLocation(
                            document_id=document.document_id,
                            document_name=document.document_name,
                            file_kind=document.file_kind,
                            segment_index=segment.segment_index,
                            page=segment.page,
                            paragraph_path=segment.paragraph_path,
                            char_start=start,
                            char_end=end,
                        ),
                        context_before=segment.text[max(0, start - CONTEXT_CHARS) : start],
                        context_after=segment.text[end : end + CONTEXT_CHARS],
                    )
                )
                if candidate.pattern_name:
                    per_pattern[candidate.pattern_name] += 1

    # Report model-proposed literal terms that appear nowhere in the documents.
    for target in exact_targets:
        term = target.term.strip()
        if not term:
            continue
        # Compared with whitespace folded because a term the user quoted from a
        # printed layout is often matched here across a line break.
        wanted = _normalise_term(term)
        found_anywhere = any(
            d.strategy == "exact" and _normalise_term(d.matched_text) == wanted
            for d in detections
        )
        if not found_anywhere:
            if _term_only_in_protected_regions(
                documents, term, resolved_by_document, scope
            ):
                warnings.append(
                    f"{term!r} was not proposed for redaction because it falls inside "
                    "text you asked to keep."
                )
            else:
                unmatched.append(term)

    if unmatched:
        warnings.append(
            "These requested terms were not found in any uploaded document and were "
            f"ignored (nothing was invented): {unmatched[:10]}"
        )

    for spec_name, count in sorted(per_pattern.items()):
        spec = PATTERN_REGISTRY.get(spec_name)
        if spec and spec.caveat and count:
            warnings.append(f"{spec.label}: {spec.caveat}")

    logger.info(
        "matching_complete",
        extra={
            "documents": len(documents),
            "detections": len(detections),
            "suppressed_by_exclusion": suppressed,
            "unmatched_terms": len(unmatched),
        },
    )

    return MatchResult(
        detections=detections,
        unmatched_exact_terms=unmatched,
        suppressed_by_exclusion=suppressed,
        per_pattern_counts=dict(per_pattern),
        resolved_exclusions=resolved_by_document,
        warnings=warnings,
    )


def _term_only_in_protected_regions(
    documents: list[ExtractedDocument],
    term: str,
    resolved_by_document: dict[str, list[Exclusion]],
    scope: set[str] | None,
) -> bool:
    """True when the term exists but every occurrence sits in protected text."""
    seen = False
    for document in documents:
        if scope and document.document_id not in scope:
            continue
        exclusions = resolved_by_document.get(document.document_id, [])
        for segment in document.segments:
            hits = _literal_ranges(segment.text, term)
            hits += [
                (start, end)
                for start, end in _case_insensitive_ranges(segment.text, term)
                if (start, end) not in hits
            ]
            flexible = _whitespace_flexible_regex(term)
            if flexible is not None:
                hits += [(m.start(), m.end()) for m in flexible.finditer(segment.text)]
            if not hits:
                continue
            seen = True
            protected = _exclusion_ranges(segment, exclusions)
            if not all(_overlaps(s, e, protected) for s, e in hits):
                return False
    return seen
