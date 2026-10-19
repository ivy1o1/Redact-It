"""API request and response models.

Everything the frontend renders comes from here, so the shapes are explicit
rather than ad-hoc dicts. Nothing in these models carries a filesystem path or an
internal identifier that could be used to reach outside a session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .redaction.verify import VerificationReport

MAX_DETECTION_IDS = 20_000
MAX_ID_LENGTH = 128


# ---------------------------------------------------------------------- system
class ModelHealthOut(BaseModel):
    reachable: bool
    status: str
    models: list[str] = Field(default_factory=list)
    base_url: str
    detail: str | None = None
    expected_model: str
    expected_model_loaded: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    storage_root_exists: bool


class ModelStatusResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: ModelHealthOut
    interpretation_mode: Literal["model", "deterministic"]
    hint: str | None = None


# ------------------------------------------------------------------- documents
class DocumentSummary(BaseModel):
    document_id: str
    document_name: str
    file_kind: Literal["pdf", "docx"]
    size_bytes: int
    sha256: str
    page_count: int | None = None
    char_count: int = 0
    has_text_layer: bool = False
    scanned: bool = False
    encrypted: bool = False
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DocumentFailure(BaseModel):
    document_name: str
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------- detections
class DetectionOut(BaseModel):
    detection_id: str
    document_id: str
    document_name: str
    file_kind: Literal["pdf", "docx"]
    category: str
    matched_text: str
    pattern_name: str | None = None
    strategy: Literal["exact", "pattern"]
    source: Literal["instruction", "model", "both"] = "instruction"
    confidence: float
    heuristic: bool = False
    fuzzy: bool = False
    occurrence: int = 1
    page: int | None = None
    paragraph_path: str | None = None
    segment_index: int
    char_start: int
    char_end: int
    context_before: str = ""
    context_after: str = ""


class TargetOut(BaseModel):
    kind: Literal["exact", "pattern"]
    term: str
    pattern: str | None = None
    category: str
    source: Literal["instruction", "model"]


class ExclusionOut(BaseModel):
    kind: Literal["exact", "pattern"]
    term: str
    pattern: str | None = None
    source: Literal["instruction", "model"]
    protected_texts: list[str] = Field(default_factory=list)


class InterpretationOut(BaseModel):
    source: Literal["deterministic", "model+deterministic"]
    model_used: bool = False
    interpretation_mode: Literal["model", "deterministic"]
    document_scope: Literal["all", "named"] = "all"
    document_names: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    clarification_needed: str | None = None
    targets: list[TargetOut] = Field(default_factory=list)
    exclusions: list[ExclusionOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejected_model_proposals: list[str] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    documents_uploaded: int
    documents_analysed: int
    documents_failed: int
    detections: int
    by_category: dict[str, int] = Field(default_factory=dict)
    by_pattern: dict[str, int] = Field(default_factory=dict)
    suppressed_by_exclusion: int = 0
    unmatched_exact_terms: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    expires_at: datetime
    instruction: str
    documents: list[DocumentSummary] = Field(default_factory=list)
    document_failures: list[DocumentFailure] = Field(default_factory=list)
    interpretation: InterpretationOut
    summary: AnalysisSummary
    detections: list[DetectionOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    review_endpoint: str
    next_step: str


# ---------------------------------------------------------------------- review
class ReviewRequest(BaseModel):
    approved_detection_ids: list[str] = Field(default_factory=list)
    rejected_detection_ids: list[str] = Field(default_factory=list)

    @field_validator("approved_detection_ids", "rejected_detection_ids")
    @classmethod
    def _check_ids(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_DETECTION_IDS:
            raise ValueError(f"At most {MAX_DETECTION_IDS} detection ids per request.")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Detection ids must be non-empty strings.")
            if len(item) > MAX_ID_LENGTH:
                raise ValueError(f"Detection id longer than {MAX_ID_LENGTH} characters.")
            cleaned.append(item.strip())
        return cleaned


class DocumentTally(BaseModel):
    document_id: str
    document_name: str
    total: int
    approved: int
    rejected: int


class ReviewResponse(BaseModel):
    session_id: str
    status: str
    approved: int
    rejected: int
    ignored: int
    per_document: list[DocumentTally] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    redact_endpoint: str
    next_step: str


# --------------------------------------------------------------------- redact
class OutputOut(BaseModel):
    download_token: str
    document_id: str
    source_name: str
    output_name: str
    size_bytes: int
    sha256: str
    created_at: datetime
    expires_at: datetime
    download_url: str


class RedactionOutcomeOut(BaseModel):
    document_id: str
    source_name: str
    output_name: str
    applied: int
    skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RedactResponse(BaseModel):
    session_id: str
    status: str
    outcomes: list[RedactionOutcomeOut] = Field(default_factory=list)
    outputs: list[OutputOut] = Field(default_factory=list)
    verification: list[VerificationReport] = Field(default_factory=list)
    residual_text_found: bool = False
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_step: str


# ------------------------------------------------------------------ session mgmt
class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    instruction_present: bool
    documents: list[DocumentSummary] = Field(default_factory=list)
    detections: int
    approved: int
    outputs: list[OutputOut] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime


class DeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    detail: str


# ---------------------------------------------------------------------- errors
class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


__all__ = [
    "AnalysisResponse",
    "AnalysisSummary",
    "DeleteResponse",
    "DetectionOut",
    "DocumentFailure",
    "DocumentSummary",
    "DocumentTally",
    "ErrorBody",
    "ErrorResponse",
    "ExclusionOut",
    "HealthResponse",
    "InterpretationOut",
    "ModelHealthOut",
    "ModelStatusResponse",
    "OutputOut",
    "RedactionOutcomeOut",
    "RedactResponse",
    "ReviewRequest",
    "ReviewResponse",
    "SessionStatusResponse",
    "TargetOut",
]
