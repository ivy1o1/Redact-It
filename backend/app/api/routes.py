"""HTTP surface.

Nine endpoints covering the whole workflow: health, analysis, review, redaction,
download and session teardown. Handlers are thin - the ordering rules live in
:class:`app.services.analysis.AnalysisService` and the storage layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from ..config import Settings
from ..dependencies import get_llm_client, get_service, get_settings, get_store
from ..extraction.base import ExtractedDocument
from ..llm.adapter import LlamaCppClient
from ..logging_setup import get_logger
from ..matching.engine import Detection
from ..redaction.verify import VERIFICATION_LIMITATIONS
from ..schemas import (
    AnalysisResponse,
    AnalysisSummary,
    DeleteResponse,
    DetectionOut,
    DocumentFailure,
    DocumentSummary,
    DocumentTally,
    ExclusionOut,
    HealthResponse,
    InterpretationOut,
    ModelHealthOut,
    ModelStatusResponse,
    OutputOut,
    RedactionOutcomeOut,
    RedactResponse,
    ReviewRequest,
    ReviewResponse,
    SessionStatusResponse,
    TargetOut,
)
from ..services.analysis import GENERAL_LIMITATIONS, AnalysisService
from ..storage import OutputFile, Session, SessionStore

logger = get_logger("api")

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_NO_SNIFF = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}

system_router = APIRouter(tags=["System"])
api_router = APIRouter(tags=["Redaction"])


# --------------------------------------------------------------------- helpers
def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _document_summary(document: ExtractedDocument) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        document_name=document.document_name,
        file_kind=document.file_kind,
        size_bytes=document.size_bytes,
        sha256=document.sha256,
        page_count=document.page_count,
        char_count=document.char_count,
        has_text_layer=document.has_text_layer,
        scanned=document.scanned,
        encrypted=document.encrypted,
        truncated=document.truncated,
        warnings=document.warnings,
        limitations=document.limitations,
    )


def _detection_out(detection: Detection) -> DetectionOut:
    location = detection.location
    return DetectionOut(
        detection_id=detection.detection_id,
        document_id=location.document_id,
        document_name=location.document_name,
        file_kind=location.file_kind,
        category=detection.category,
        matched_text=detection.matched_text,
        pattern_name=detection.pattern_name,
        strategy=detection.strategy,
        source=detection.source,
        confidence=detection.confidence,
        heuristic=detection.heuristic,
        fuzzy=detection.fuzzy,
        occurrence=detection.occurrence,
        page=location.page,
        paragraph_path=location.paragraph_path,
        segment_index=location.segment_index,
        char_start=location.char_start,
        char_end=location.char_end,
        context_before=detection.context_before,
        context_after=detection.context_after,
    )


def _interpretation_out(session: Session) -> InterpretationOut:
    interpretation = session.interpretation
    plan = interpretation.plan

    protected: dict[str, set[str]] = {}
    result = session.match_result
    if result is not None:
        for exclusions in result.resolved_exclusions.values():
            for exclusion in exclusions:
                protected.setdefault(exclusion.term.lower(), set()).update(
                    exclusion.effective_terms()
                )

    return InterpretationOut(
        source=interpretation.source,
        model_used=interpretation.model_used,
        interpretation_mode=session.interpretation_mode,
        document_scope=interpretation.document_scope,
        document_names=interpretation.document_names,
        ambiguous=interpretation.ambiguous,
        clarification_needed=interpretation.clarification_needed,
        targets=[
            TargetOut(
                kind=target.kind,
                term=target.term,
                pattern=target.pattern,
                category=target.category,
                source=target.source,
            )
            for target in plan.targets
        ],
        exclusions=[
            ExclusionOut(
                kind=exclusion.kind,
                term=exclusion.term,
                pattern=exclusion.pattern,
                source=exclusion.source,
                protected_texts=sorted(protected.get(exclusion.term.lower(), set()))[:12],
            )
            for exclusion in plan.exclusions
        ],
        warnings=interpretation.warnings,
        rejected_model_proposals=interpretation.rejected_model_proposals,
    )


def _output_out(output: OutputFile, settings: Settings) -> OutputOut:
    return OutputOut(
        download_token=output.token,
        document_id=output.document_id,
        source_name=output.source_name,
        output_name=output.output_name,
        size_bytes=output.size_bytes,
        sha256=output.sha256,
        created_at=output.created_at,
        expires_at=output.expires_at,
        download_url=f"{settings.api_prefix}/downloads/{output.token}",
    )


# ---------------------------------------------------------------------- system
@system_router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def root_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="redactit-api",
        version=settings.app_version,
        environment=settings.environment,
        storage_root_exists=settings.resolved_storage_root.exists(),
    )


@api_router.get("/health", response_model=HealthResponse)
def api_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return root_health(settings)


@api_router.get(
    "/health/model",
    response_model=ModelStatusResponse,
    summary="Probe the local model server (sends no document content)",
)
async def model_health(
    settings: Settings = Depends(get_settings),
    client: LlamaCppClient | None = Depends(get_llm_client),
) -> ModelStatusResponse:
    if client is None:
        return ModelStatusResponse(
            status="not_ready",
            model=ModelHealthOut(
                reachable=False,
                status="not_configured",
                models=[],
                base_url=settings.llm_base_url,
                detail="No local model client was created for this process.",
                expected_model=settings.llm_model,
                expected_model_loaded=False,
            ),
            interpretation_mode=settings.interpretation_mode,
            hint=(
                "Set INTERPRETATION_MODE=deterministic to analyse without the model, "
                "or configure LLM_BASE_URL."
            ),
        )

    health = await client.health()
    loaded = any(
        model.lower() == settings.llm_model.lower() for model in health.models
    )
    hint = None
    if health.status != "ok":
        hint = (
            "Start llama-server on 127.0.0.1 and wait for /health to return 'ok'. "
            "RedactIt never falls back to a cloud model."
        )
    elif not loaded:
        hint = (
            f"The server is up but does not report a model named "
            f"{settings.llm_model!r}. Reported: {health.models or 'none'}. Check the "
            "--alias llama-server was started with, or set LLM_MODEL to match."
        )

    return ModelStatusResponse(
        status="ready" if health.status == "ok" and loaded else "not_ready",
        model=ModelHealthOut(
            reachable=health.reachable,
            status=health.status,
            models=health.models,
            base_url=health.base_url,
            detail=health.detail,
            expected_model=settings.llm_model,
            expected_model_loaded=loaded,
        ),
        interpretation_mode=settings.interpretation_mode,
        hint=hint,
    )


# --------------------------------------------------------------------- analyze
@api_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=200,
    summary="Upload documents and get a reviewable redaction plan",
)
async def analyze(
    files: list[UploadFile] = File(description="One or more PDF or DOCX files"),
    instruction: str = Form(description="Plain-English description of what to redact"),
    interpretation_mode: Literal["model", "deterministic"] | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    service: AnalysisService = Depends(get_service),
) -> AnalysisResponse:
    session = await service.analyze(files, instruction, mode=interpretation_mode)
    return _analysis_response(session, settings)


def _analysis_response(session: Session, settings: Settings) -> AnalysisResponse:
    result = session.match_result
    by_category = Counter(detection.category for detection in session.detections)

    limitations = _unique(
        GENERAL_LIMITATIONS
        + [item for document in session.extracted for item in document.limitations]
    )
    warnings = _unique(session.warnings)

    if session.document_failures:
        warnings.insert(
            0,
            f"{len(session.document_failures)} uploaded file(s) could NOT be analysed "
            "and were not redacted. See 'document_failures' for the reason.",
        )

    return AnalysisResponse(
        session_id=session.session_id,
        status=session.status.value,
        expires_at=session.expires_at,
        instruction=session.instruction,
        documents=[_document_summary(document) for document in session.extracted],
        document_failures=[
            DocumentFailure(
                document_name=failure.document_name,
                code=failure.code,
                message=failure.message,
                details=failure.details,
            )
            for failure in session.document_failures
        ],
        interpretation=_interpretation_out(session),
        summary=AnalysisSummary(
            documents_uploaded=len(session.documents),
            documents_analysed=len(session.extracted),
            documents_failed=len(session.document_failures),
            detections=len(session.detections),
            by_category=dict(sorted(by_category.items())),
            by_pattern=dict(sorted((result.per_pattern_counts if result else {}).items())),
            suppressed_by_exclusion=result.suppressed_by_exclusion if result else 0,
            unmatched_exact_terms=list(result.unmatched_exact_terms) if result else [],
        ),
        detections=[_detection_out(detection) for detection in session.detections],
        warnings=warnings,
        limitations=limitations,
        review_endpoint=f"{settings.api_prefix}/sessions/{session.session_id}/review",
        next_step=(
            "Review the detections, then POST the approved detection_id values to "
            f"{settings.api_prefix}/sessions/{session.session_id}/review. Nothing is "
            "changed until you do."
        ),
    )


# -------------------------------------------------------------------- sessions
@api_router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
def session_status(
    session_id: str,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_store),
) -> SessionStatusResponse:
    session = store.get(session_id)
    approved = len(session.review.approved_detection_ids) if session.review else 0
    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        instruction_present=bool(session.instruction),
        documents=[_document_summary(document) for document in session.extracted],
        detections=len(session.detections),
        approved=approved,
        outputs=[_output_out(output, settings) for output in session.outputs],
        created_at=session.created_at,
        expires_at=session.expires_at,
    )


@api_router.delete("/sessions/{session_id}", response_model=DeleteResponse)
def delete_session(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> DeleteResponse:
    deleted = store.purge_session(session_id)
    return DeleteResponse(
        session_id=session_id,
        deleted=deleted,
        detail=(
            "Session and all of its files were deleted."
            if deleted
            else "No live session with that id; nothing was deleted."
        ),
    )


@api_router.post(
    "/sessions/{session_id}/review",
    response_model=ReviewResponse,
    summary="Approve or reject detections by id",
)
def review_session(
    session_id: str,
    payload: ReviewRequest,
    settings: Settings = Depends(get_settings),
    service: AnalysisService = Depends(get_service),
) -> ReviewResponse:
    session = service.review(
        session_id,
        approved_ids=payload.approved_detection_ids,
        rejected_ids=payload.rejected_detection_ids,
    )
    assert session.review is not None  # guaranteed by review()

    totals: dict[str, DocumentTally] = {}
    for document in session.extracted:
        totals[document.document_id] = DocumentTally(
            document_id=document.document_id,
            document_name=document.document_name,
            total=0,
            approved=0,
            rejected=0,
        )
    rejected = set(session.review.rejected_detection_ids)
    for detection in session.detections:
        tally = totals.get(detection.location.document_id)
        if tally is None:  # pragma: no cover - scope-filtered documents
            continue
        tally.total += 1
        if detection.detection_id in session.review.approved_detection_ids:
            tally.approved += 1
        elif detection.detection_id in rejected:
            tally.rejected += 1

    return ReviewResponse(
        session_id=session.session_id,
        status=session.status.value,
        approved=len(session.review.approved_detection_ids),
        rejected=len(session.review.rejected_detection_ids),
        ignored=len(session.detections)
        - len(session.review.approved_detection_ids)
        - len(session.review.rejected_detection_ids),
        per_document=[tally for tally in totals.values() if tally.total],
        warnings=session.review_warnings,
        redact_endpoint=f"{settings.api_prefix}/sessions/{session_id}/redact",
        next_step=(
            f"POST {settings.api_prefix}/sessions/{session_id}/redact to generate the "
            "sanitised files. Only the approved detections will be applied and the "
            "originals are never modified."
        ),
    )


@api_router.post(
    "/sessions/{session_id}/redact",
    response_model=RedactResponse,
    summary="Apply approved redactions to copies of the documents",
)
def redact_session(
    session_id: str,
    settings: Settings = Depends(get_settings),
    service: AnalysisService = Depends(get_service),
) -> RedactResponse:
    session = service.redact(session_id)

    outcomes = [
        RedactionOutcomeOut(
            document_id=outcome.document_id,
            source_name=outcome.source_name,
            output_name=outcome.output_name,
            applied=outcome.applied,
            skipped=outcome.skipped,
            warnings=outcome.warnings,
        )
        for outcome in session.outcomes
    ]
    reports = list(session.verification.values())
    residual = any(report.residual_text_found for report in reports)

    warnings = _unique(
        [warning for outcome in session.outcomes for warning in outcome.warnings]
        + [skip for outcome in session.outcomes for skip in outcome.skipped]
        + [
            report.not_verifiable_reason
            for report in reports
            if report.not_verifiable_reason
        ]
        + [
            f"{verified.text!r} is still present in {report.output_name}: {verified.detail}"
            for report in reports
            for verified in report.results
            if verified.status == "failed"
        ]
    )
    if residual:
        warnings.insert(
            0,
            "VERIFICATION FOUND RESIDUAL TEXT. At least one approved string is still "
            "in an output file. Do not distribute these files; inspect them manually.",
        )

    limitations = _unique(
        list(VERIFICATION_LIMITATIONS)
        + [item for document in session.extracted for item in document.limitations]
        + [item for report in reports for item in report.limitations]
    )

    return RedactResponse(
        session_id=session.session_id,
        status=session.status.value,
        outcomes=outcomes,
        outputs=[_output_out(output, settings) for output in session.outputs],
        verification=reports,
        residual_text_found=residual,
        warnings=warnings,
        limitations=limitations,
        next_step=(
            "Download each file from its download_url before it expires. The originals "
            "are unchanged and still on disk for this session."
            if not residual
            else "Fix the residual findings before distributing these files."
        ),
    )


# ------------------------------------------------------------------- downloads
@api_router.get(
    "/downloads/{token}",
    summary="Download one generated file with its opaque token",
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def download(
    token: str,
    store: SessionStore = Depends(get_store),
) -> FileResponse:
    _, output = store.resolve_download(token)
    suffix = Path(output.output_name).suffix.lstrip(".")
    return FileResponse(
        output.path,
        media_type=MEDIA_TYPES.get(suffix, "application/octet-stream"),
        filename=output.output_name,
        content_disposition_type="attachment",
        headers=_NO_SNIFF,
    )


__all__ = ["api_router", "system_router"]
