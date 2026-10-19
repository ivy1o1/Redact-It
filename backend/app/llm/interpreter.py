"""Instruction interpretation: local Gemma proposal merged with deterministic parsing.

Division of responsibility
---------------------------
The deterministic parser (:mod:`app.llm.instruction_parser`) owns what the user
asked to *keep*, and recognises common data-type vocabulary and quoted phrases.
That direction must never be wrong, because acting on a false "redact this"
destroys data.

The local model contributes what a keyword parser cannot: literal phrases named
in unusual wording, and a natural-language clarification when the instruction is
vague. Every model proposal is validated before it can influence a detection:

* a detector name must exist in the closed registry;
* a detector must be consistent with the model's own label for it - this is what
  stops a 1B model from mapping "grades" onto ``credit_card``;
* an exclusion is only accepted if the term literally occurs in the document;
* a target that the user asked to keep is always dropped.

Nothing is dropped silently: every rejection is reported in ``warnings``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import Settings
from ..errors import InstructionAmbiguousError, InstructionUnusableError
from ..logging_setup import get_logger
from ..matching.engine import Exclusion, MatchPlan, Target
from ..matching.patterns import PATTERN_KEYWORDS, PATTERN_REGISTRY, supported_pattern_names
from .adapter import LlamaCppClient
from .instruction_parser import (
    ParsedInstruction,
    conflicts_with_keep,
    parse_instruction,
)

logger = get_logger("llm.interpreter")

_SYSTEM_PROMPT = """You are the instruction parser for RedactIt, a document redaction tool.
Turn the user's redaction instruction into a JSON plan.

Rules:
1. "targets" = data to redact. "exclusions" = data that MUST be kept.
2. Text the user says to KEEP / LEAVE / PRESERVE / RETAIN / leave "unchanged" / "do not remove" goes ONLY in "exclusions", never in "targets".
3. For a data type, use kind="pattern", term=the user's own words for that data, and pattern from the allowed enum only.
   phone = phone/fax/mobile numbers. address = home/postal/street addresses. person_name = people's names.
   email = email addresses. ssn = social security numbers. date = dates. credit_card = payment card numbers.
   zip_code = postal/ZIP codes. ip_address = IP addresses. url = web addresses.
4. For specific literal text the user names or quotes, use kind="exact", term=that literal text, pattern=null.
5. "document_scope" is "all" unless the instruction names particular files, in which case use "named" and list the filenames in "document_names".
6. If the instruction is too vague to act on, set ambiguous=true, emit no targets, and say in clarification_needed exactly what the user must specify."""


def build_plan_schema() -> dict[str, Any]:
    """JSON Schema used for grammar-constrained decoding on llama-server."""
    patterns = supported_pattern_names()
    return {
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "kind": {"type": "string", "enum": ["exact", "pattern"]},
                        "pattern": {"type": ["string", "null"], "enum": patterns + [None]},
                        "category": {"type": "string"},
                    },
                    "required": ["term", "kind", "pattern", "category"],
                    "additionalProperties": False,
                },
            },
            "exclusions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "kind": {"type": "string", "enum": ["exact"]},
                    },
                    "required": ["term", "kind"],
                    "additionalProperties": False,
                },
            },
            "document_scope": {"type": "string", "enum": ["all", "named"]},
            "document_names": {"type": "array", "items": {"type": "string"}},
            "ambiguous": {"type": "boolean"},
            "clarification_needed": {"type": ["string", "null"]},
        },
        "required": [
            "targets",
            "exclusions",
            "document_scope",
            "document_names",
            "ambiguous",
            "clarification_needed",
        ],
        "additionalProperties": False,
    }


class Interpretation(BaseModel):
    source: Literal["deterministic", "model+deterministic"] = "deterministic"
    plan: MatchPlan = Field(default_factory=MatchPlan)
    document_scope: Literal["all", "named"] = "all"
    document_names: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    clarification_needed: str | None = None
    warnings: list[str] = Field(default_factory=list)
    rejected_model_proposals: list[str] = Field(default_factory=list)
    model_used: bool = False
    model_raw: dict[str, Any] | None = None
    parsed: dict[str, Any] = Field(default_factory=dict)


def _label_matches_pattern(label: str, pattern_name: str) -> bool:
    """Is the model's own wording for this target consistent with the detector?"""
    if not label:
        return False
    lowered = label.lower()
    keywords = PATTERN_KEYWORDS.get(pattern_name, ())
    for keyword in keywords:
        if re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", lowered):
            return True
    # Allow single-word overlap as a weaker signal: "mobile" for phone, "ssn".
    tokens = {t for t in re.findall(r"[a-z]{3,}", lowered)}
    for keyword in keywords:
        if tokens & set(keyword.split()):
            return True
    return False


def _looks_like_literal_phrase(term: str) -> bool:
    cleaned = term.strip()
    if len(cleaned) < 3 or len(cleaned) > 90:
        return False
    if "_" in cleaned or cleaned.islower() and " " not in cleaned:
        return False
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def _base_plan(parsed: ParsedInstruction) -> MatchPlan:
    targets: list[Target] = []
    for name, label in sorted(parsed.pattern_targets.items()):
        targets.append(
            Target(
                kind="pattern",
                term=label,
                pattern=name,
                category=PATTERN_REGISTRY[name].label,
                source="instruction",
            )
        )
    for term in parsed.exact_targets:
        targets.append(
            Target(kind="exact", term=term, pattern=None, category="named text", source="instruction")
        )

    exclusions: list[Exclusion] = []
    for term in parsed.exclusion_terms:
        exclusions.append(Exclusion(kind="exact", term=term, source="instruction"))

    return MatchPlan(targets=targets, exclusions=exclusions)


def _merge_model_output(
    plan: MatchPlan,
    model: dict[str, Any],
    parsed: ParsedInstruction,
    document_sample: str,
    warnings: list[str],
    rejected: list[str],
) -> tuple[Literal["all", "named"], list[str], bool, str | None]:
    kept_terms = list(parsed.exclusion_terms)
    sample_lower = document_sample.lower()
    existing_targets = {target.identity for target in plan.targets}
    existing_exclusions = {(e.kind, e.term.lower(), e.pattern) for e in plan.exclusions}

    for raw in model.get("targets") or []:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind")
        term = str(raw.get("term") or "").strip()
        pattern = raw.get("pattern")
        category = str(raw.get("category") or "").strip() or "unspecified"

        if conflicts_with_keep(term, kept_terms):
            rejected.append(f"target {term!r}: you asked to keep this")
            warnings.append(
                f"Ignored a proposed redaction of {term!r} because your instruction "
                "says to keep it."
            )
            continue

        if kind == "exact":
            if not term:
                rejected.append("exact target with empty term")
                continue
            candidate = Target(
                kind="exact", term=term, pattern=None, category=category, source="model"
            )
            if candidate.identity in existing_targets:
                continue
            plan.targets.append(candidate)
            existing_targets.add(candidate.identity)
            continue

        if kind == "pattern":
            if pattern not in PATTERN_REGISTRY:
                rejected.append(f"unknown detector {pattern!r}")
                continue
            if _label_matches_pattern(term, pattern):
                candidate = Target(
                    kind="pattern",
                    term=term,
                    pattern=pattern,
                    category=PATTERN_REGISTRY[pattern].label,
                    source="model",
                )
                if candidate.identity in existing_targets:
                    continue
                # Same detector already added from the instruction: keep the
                # instruction-sourced one, it is the authoritative label.
                if any(
                    t.kind == "pattern" and t.pattern == pattern for t in plan.targets
                ):
                    continue
                plan.targets.append(candidate)
                existing_targets.add(candidate.identity)
                continue

            # Inconsistent label. If it reads like literal text, salvage it as an
            # exact term; the matching engine discards it when it is absent from
            # the documents, so nothing can be invented.
            if _looks_like_literal_phrase(term) and term.lower() in sample_lower:
                candidate = Target(
                    kind="exact",
                    term=term,
                    pattern=None,
                    category=category,
                    source="model",
                )
                if candidate.identity not in existing_targets:
                    plan.targets.append(candidate)
                    existing_targets.add(candidate.identity)
                rejected.append(
                    f"detector {pattern!r} for {term!r}: label did not match the "
                    "detector, re-read as literal text"
                )
            else:
                rejected.append(
                    f"detector {pattern!r} for {term!r}: label does not match that detector"
                )
            continue

        rejected.append(f"target with unsupported kind {kind!r}")

    for raw in model.get("exclusions") or []:
        if not isinstance(raw, dict):
            continue
        term = str(raw.get("term") or "").strip()
        if not term:
            continue
        # Only protect text that actually exists - a small model happily invents
        # exclusion labels like "student_names", which would protect nothing and
        # only add noise to the review.
        if term.lower() not in sample_lower:
            rejected.append(f"exclusion {term!r}: not present in the document")
            continue
        key = ("exact", term.lower(), None)
        if key in existing_exclusions:
            continue
        plan.exclusions.append(Exclusion(kind="exact", term=term, source="model"))
        existing_exclusions.add(key)

    scope_raw = str(model.get("document_scope") or "all").lower()
    scope: Literal["all", "named"] = "named" if scope_raw == "named" else "all"
    names = [
        str(name).strip()
        for name in (model.get("document_names") or [])
        if isinstance(name, (str, int)) and str(name).strip()
    ]
    if scope == "named" and not names:
        scope = "all"
        warnings.append(
            "The model asked to limit the scope to named files but named none, so "
            "all uploaded documents are in scope."
        )

    ambiguous = bool(model.get("ambiguous"))
    clarification = model.get("clarification_needed")
    if not isinstance(clarification, str):
        clarification = None
    return scope, names[:20], ambiguous, clarification.strip() or None if clarification else None


async def interpret_instruction(
    *,
    instruction: str,
    document_sample: str,
    client: LlamaCppClient | None,
    settings: Settings,
    mode: Literal["model", "deterministic"] = "model",
    document_names: list[str] | None = None,
) -> Interpretation:
    """Produce a validated :class:`MatchPlan` for one instruction.

    ``mode='model'`` requires the local server; failures propagate as
    ``ModelError`` rather than degrading silently. ``mode='deterministic'`` skips
    the model entirely and says so in the response.
    """
    parsed = parse_instruction(instruction)
    plan = _base_plan(parsed)

    interpretation = Interpretation(
        source="deterministic",
        plan=plan,
        document_scope="all",
        document_names=list(document_names or []),
        ambiguous=parsed.ambiguous,
        model_used=False,
        parsed={
            "remove_clauses": parsed.remove_clauses,
            "keep_clauses": parsed.keep_clauses,
            "exact_targets": parsed.exact_targets,
            "pattern_targets": parsed.pattern_targets,
            "exclusion_terms": parsed.exclusion_terms,
        },
    )

    # Read the two lists back off the model rather than keeping the ones passed
    # to the constructor: pydantic copies list-valued fields during validation,
    # so appending to the caller's list leaves the interpretation showing an
    # empty report. That silently hid every rejected model proposal and every
    # warning, which is the opposite of this module's contract.
    warnings = interpretation.warnings
    rejected = interpretation.rejected_model_proposals
    warnings.extend(parsed.notes)

    if mode == "deterministic":
        if client is not None:
            warnings.append(
                "INTERPRETATION_MODE=deterministic: the local model was not consulted."
            )
        _finalise(interpretation)
        return interpretation

    if client is None:  # pragma: no cover - guarded by the caller
        raise InstructionUnusableError(
            "Model interpretation was requested but no local model client is configured."
        )

    bounded_sample = document_sample[: max(0, settings.llm_max_prompt_chars)]
    if len(document_sample) > len(bounded_sample):
        warnings.append(
            "Only the first "
            f"{len(bounded_sample)} characters of the documents were shown to the local "
            "model. Detection itself always runs over the full text."
        )

    user_prompt = (
        f'Instruction: "{instruction.strip()}"\n\n'
        f"Uploaded documents: {', '.join(document_names or []) or 'unspecified'}\n\n"
        f"Document sample:\n{bounded_sample}\n\nOutput the JSON plan:"
    )

    model_output = await client.chat_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=build_plan_schema(),
    )

    scope, names, model_ambiguous, clarification = _merge_model_output(
        plan, model_output, parsed, bounded_sample, warnings, rejected
    )

    interpretation.source = "model+deterministic"
    interpretation.model_used = True
    interpretation.model_raw = model_output
    interpretation.document_scope = scope
    if names:
        interpretation.document_names = names
    # The deterministic parser decides ambiguity: a small model flags almost
    # everything as ambiguous, which would block usable instructions.
    interpretation.ambiguous = parsed.ambiguous
    interpretation.clarification_needed = clarification if parsed.ambiguous else None

    if rejected:
        logger.info("model_proposals_rejected", extra={"count": len(rejected)})

    _finalise(interpretation)
    return interpretation


def _finalise(interpretation: Interpretation) -> None:
    """Raise an actionable error when nothing can be acted on."""
    if interpretation.plan.targets:
        interpretation.ambiguous = False
        return

    hint = interpretation.clarification_needed
    base = (
        "Nothing in this instruction could be turned into a concrete redaction "
        "target. Be specific about the data to remove."
    )
    if hint:
        base = f"{base} The local model suggested: {hint[:300]}"
    examples = (
        " Try, for example: \"Remove all student names, phone numbers and home "
        "addresses, but keep grades and course names unchanged.\" or \"Redact every "
        "email address and the phrase 'Project Falcon'.\""
    )
    raise InstructionAmbiguousError(
        base + examples,
        details={
            "exclusions_recognised": [e.term for e in interpretation.plan.exclusions][:20],
            "supported_detectors": supported_pattern_names(),
            "clarification_needed": hint,
        },
    )
