"""Workflow orchestration: analyze -> review -> redact -> download.

The service owns the ordering guarantees the product depends on:

* analysis never touches the source files;
* nothing is redacted before an explicit review step;
* only detection ids the user approved are applied;
* every generated file is checked against the strings it was supposed to remove.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from ..config import Settings
from ..errors import (
    AlreadyRedactedError,
    EmptyInstructionError,
    FileTooLargeError,
    InstructionUnusableError,
    InvalidSelectionError,
    NoApprovedRedactionsError,
    NoFilesUploadedError,
    RedactItError,
    RedactionError,
    ReviewRequiredError,
    StaleDetectionIdError,
    TooManyFilesError,
)
from ..extraction.base import ExtractedDocument
from ..extraction.docx import extract_docx
from ..extraction.pdf import extract_pdf
from ..llm.adapter import LlamaCppClient
from ..llm.interpreter import Interpretation, interpret_instruction
from ..logging_setup import get_logger
from ..matching.engine import Detection, MatchResult, detect
from ..redaction import output_name_for, redact_document, verify_redaction
from ..storage import (
    DocumentFailureRecord,
    Review,
    Session,
    SessionStatus,
    SessionStore,
    StoredDocument,
    new_document_id,
)
from ..uploads import kind_from_extension, sanitize_filename, save_upload

logger = get_logger("services.analysis")

_DETECTION_ID_RE = re.compile(r"^det_[0-9a-f]{24}$")

GENERAL_LIMITATIONS = [
    "Detection is pattern- and keyword-based: it proposes candidates, it does not "
    "understand the document. It can miss sensitive data written in an unrecognised "
    "form, and it can propose matches that are not sensitive at all. Nothing is "
    "removed until you approve it.",
    "RedactIt is not a legal-compliance tool and does not guarantee that all "
    "personal or sensitive data has been found or removed.",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_text(
    path: Path,
    *,
    document_id: str,
    document_name: str,
    file_kind: str,
    max_chars: int,
) -> ExtractedDocument:
    if file_kind == "pdf":
        return extract_pdf(
            path, document_id=document_id, document_name=document_name, max_chars=max_chars
        )
    return extract_docx(
        path, document_id=document_id, document_name=document_name, max_chars=max_chars
    )


def _scope_matches(document_name: str, wanted: set[str]) -> bool:
    candidates = {
        document_name.lower(),
        Path(document_name).name.lower(),
        Path(document_name).stem.lower(),
    }
    return bool(candidates & wanted)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        client: LlamaCppClient | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._client = client

    # ----------------------------------------------------------------- analyze
    async def analyze(
        self,
        files: list[UploadFile],
        instruction: str,
        *,
        mode: Literal["model", "deterministic"] | None = None,
    ) -> Session:
        instruction = (instruction or "").strip()
        if not instruction:
            raise EmptyInstructionError()

        limit = self._settings.max_instruction_chars
        if len(instruction) > limit:
            raise InstructionUnusableError(
                f"The instruction is {len(instruction)} characters long; the limit is "
                f"{limit}. Shorten it to the essentials: what to remove, and what to keep.",
                details={"length": len(instruction), "limit": limit},
            )

        if not files:
            raise NoFilesUploadedError()
        if len(files) > self._settings.max_upload_files:
            raise TooManyFilesError(
                f"{len(files)} files were sent but MAX_UPLOAD_FILES is "
                f"{self._settings.max_upload_files}. Upload fewer files per analysis.",
                details={"received": len(files), "limit": self._settings.max_upload_files},
            )

        session = self._store.create_session(instruction)
        try:
            await self._intake(session, files)
            self._extract_all(session)
            if not session.extracted:
                # Every file failed. Surface the first reason instead of returning an
                # empty analysis that looks like "nothing sensitive was found".
                raise _as_error(session.document_failures[0])
            await self._interpret(session, mode)
            self._detect(session)
        except BaseException:
            self._store.purge_session(session.session_id)
            raise

        logger.info(
            "analysis_complete",
            extra={
                "session_id": session.session_id,
                "documents": len(session.extracted),
                "failures": len(session.document_failures),
                "detections": len(session.detections),
            },
        )
        return session

    async def _intake(self, session: Session, files: list[UploadFile]) -> None:
        total = 0
        for upload in files:
            # Validate the declared name before any bytes reach the disk.
            display_name = sanitize_filename(upload.filename)
            file_kind = kind_from_extension(display_name, self._settings.allowed_extensions)

            document_id = new_document_id()
            path, display_name, file_kind, size_bytes = await save_upload(
                upload,
                destination_dir=session.uploads_dir,
                stored_name=f"{document_id}.{file_kind}",
                allowed_extensions=self._settings.allowed_extensions,
                max_file_size_bytes=self._settings.max_file_size_bytes,
            )

            total += size_bytes
            if total > self._settings.max_total_upload_bytes:
                path.unlink(missing_ok=True)
                raise FileTooLargeError(
                    f"The batch exceeds MAX_TOTAL_UPLOAD_BYTES "
                    f"({self._settings.max_total_upload_bytes // (1024 * 1024)} MB). "
                    "Upload fewer or smaller files.",
                    details={
                        "limit_bytes": self._settings.max_total_upload_bytes,
                        "received_bytes": total,
                    },
                )

            session.documents.append(
                StoredDocument(
                    document_id=document_id,
                    display_name=display_name,
                    file_kind=file_kind,
                    stored_path=path,
                    size_bytes=size_bytes,
                    sha256="",
                )
            )

    def _extract_all(self, session: Session) -> None:
        for stored in session.documents:
            try:
                extracted = _extract_text(
                    stored.stored_path,
                    document_id=stored.document_id,
                    document_name=stored.display_name,
                    file_kind=stored.file_kind,
                    max_chars=self._settings.max_extracted_chars,
                )
            except RedactItError as exc:
                # One unreadable file must not hide the files that did work, but it
                # must never look like it was processed either.
                logger.warning(
                    "document_extraction_failed",
                    extra={
                        "document_id": stored.document_id,
                        "code": exc.code,
                        "status_code": exc.status_code,
                    },
                )
                session.document_failures.append(
                    DocumentFailureRecord(
                        document_name=stored.display_name,
                        code=exc.code,
                        message=exc.message,
                        status_code=exc.status_code,
                        details=exc.details,
                    )
                )
                continue
            except Exception as exc:
                logger.error(
                    "document_extraction_crashed",
                    extra={
                        "document_id": stored.document_id,
                        "error_type": type(exc).__name__,
                    },
                )
                session.document_failures.append(
                    DocumentFailureRecord(
                        document_name=stored.display_name,
                        code="EXTRACTION_FAILED",
                        message=(
                            f"{stored.display_name!r} could not be parsed "
                            f"({type(exc).__name__}). The file may be corrupt."
                        ),
                    )
                )
                continue

            stored.sha256 = extracted.sha256
            session.extracted.append(extracted)
            session.warnings.extend(extracted.warnings)

    async def _interpret(
        self, session: Session, mode: Literal["model", "deterministic"] | None
    ) -> None:
        effective_mode = mode or self._settings.interpretation_mode
        session.interpretation_mode = effective_mode

        if effective_mode == "model" and self._client is None:
            raise RedactionError(
                "Model interpretation was requested but no local model client is "
                "configured. Point LLM_BASE_URL at a running llama-server, or set "
                "INTERPRETATION_MODE=deterministic to analyse without the model."
            )

        budget = max(0, self._settings.llm_max_prompt_chars)
        per_document = max(256, budget // max(1, len(session.extracted)))
        sample = "\n\n".join(
            f"--- {document.document_name} ---\n{document.sample(per_document)}"
            for document in session.extracted
        )[:budget]

        interpretation: Interpretation = await interpret_instruction(
            instruction=session.instruction,
            document_sample=sample,
            client=self._client,
            settings=self._settings,
            mode=effective_mode,
            document_names=[document.document_name for document in session.extracted],
        )

        session.interpretation = interpretation
        session.plan = interpretation.plan
        session.warnings.extend(interpretation.warnings)

    def _detect(self, session: Session) -> None:
        interpretation: Interpretation = session.interpretation
        scope_ids: list[str] | None = None

        if interpretation.document_scope == "named" and interpretation.document_names:
            wanted = {
                name.strip().lower()
                for name in interpretation.document_names
                if name.strip()
            }
            scope_ids = [
                document.document_id
                for document in session.extracted
                if _scope_matches(document.document_name, wanted)
            ]
            if not scope_ids:
                scope_ids = None
                session.warnings.append(
                    "The instruction named specific files "
                    f"({', '.join(interpretation.document_names[:5])}) but none matched an "
                    "uploaded document, so all documents were analysed."
                )

        result: MatchResult = detect(
            session.extracted, interpretation.plan, document_scope=scope_ids
        )
        session.match_result = result
        session.detections = list(result.detections)
        session.warnings.extend(result.warnings)
        session.status = SessionStatus.ANALYSED

    # ------------------------------------------------------------------ review
    def review(
        self,
        session_id: str,
        *,
        approved_ids: list[str],
        rejected_ids: list[str],
    ) -> Session:
        session = self._store.get(session_id)
        if session.status == SessionStatus.REDACTED:
            raise AlreadyRedactedError(
                "This session was already redacted. Re-run the analysis to start a "
                "new review.",
                details={"session_id": session_id},
            )

        approved = _dedupe(approved_ids)
        rejected = _dedupe(rejected_ids)

        malformed = sorted({i for i in approved + rejected if not _DETECTION_ID_RE.match(i)})
        if malformed:
            raise InvalidSelectionError(
                "Some detection ids are not in the form RedactIt issues ('det_' plus 24 "
                f"hex characters): {malformed[:5]}. Use the ids returned by /analyze.",
                details={"malformed_ids": malformed[:50]},
            )

        known = {detection.detection_id for detection in session.detections}
        unknown = sorted({i for i in approved + rejected if i not in known})
        if unknown:
            raise StaleDetectionIdError(
                f"{len(unknown)} detection id(s) do not belong to this analysis, for "
                f"example {unknown[0]}. Detection ids are only valid for the analysis "
                "that produced them. Re-run /analyze and select from the fresh list.",
                details={"unknown_ids": unknown[:50], "session_id": session_id},
            )

        conflicting = sorted(set(approved) & set(rejected))
        if conflicting:
            raise InvalidSelectionError(
                f"{len(conflicting)} detection id(s) appear in both the approved and the "
                "rejected list. Decide each one.",
                details={"conflicting_ids": conflicting[:50]},
            )

        warnings: list[str] = []
        if len(approved) != len(approved_ids):
            warnings.append("Duplicate ids in the approval list were ignored.")
        if len(rejected) != len(rejected_ids):
            warnings.append("Duplicate ids in the rejection list were ignored.")

        undecided = len(known - set(approved) - set(rejected))
        if undecided:
            warnings.append(
                f"{undecided} detection(s) were neither approved nor rejected and will "
                "NOT be redacted. Anything not explicitly approved is kept."
            )
        if not approved:
            warnings.append(
                "No detections were approved, so /redact will fail with "
                "NO_APPROVED_REDACTIONS."
            )

        session.review = Review(
            approved_detection_ids=approved,
            rejected_detection_ids=rejected,
            reviewed_at=_now(),
        )
        session.review_warnings = warnings
        session.status = SessionStatus.REVIEWED
        logger.info(
            "review_recorded",
            extra={
                "session_id": session.session_id,
                "approved": len(approved),
                "rejected": len(rejected),
                "undecided": undecided,
            },
        )
        return session

    # ------------------------------------------------------------------ redact
    def redact(self, session_id: str) -> Session:
        session = self._store.get(session_id)
        if session.status == SessionStatus.REDACTED:
            raise AlreadyRedactedError(
                "This session was already redacted and its outputs are registered. "
                "Download them, or re-run the analysis for a new session.",
                details={
                    "session_id": session_id,
                    "outputs": [output.output_name for output in session.outputs],
                },
            )
        if session.review is None:
            raise ReviewRequiredError(
                "No review has been recorded for this session. POST the approved "
                "detection ids to the review endpoint before requesting redaction.",
                details={"session_id": session_id},
            )

        approved_ids = set(session.review.approved_detection_ids)
        if not approved_ids:
            raise NoApprovedRedactionsError(
                "The review approved no detections, so there is nothing to redact. "
                "Approve at least one detection id.",
                details={"session_id": session_id},
            )

        approved_by_document: dict[str, list[Detection]] = defaultdict(list)
        retained_by_document: dict[str, list[Detection]] = defaultdict(list)
        for detection in session.detections:
            bucket = (
                approved_by_document
                if detection.detection_id in approved_ids
                else retained_by_document
            )
            bucket[detection.location.document_id].append(detection)

        # Build every output first and register nothing until all of them exist. A
        # half-finished redaction that reports success would be worse than a failure.
        staged: list[
            tuple[object, Path, str, ExtractedDocument, list[Detection], list[Detection]]
        ] = []
        try:
            for document in session.extracted:
                detections = approved_by_document.get(document.document_id)
                if not detections:
                    continue
                stored = next(
                    (s for s in session.documents if s.document_id == document.document_id),
                    None,
                )
                if stored is None:  # pragma: no cover - invariant
                    raise RedactionError(
                        f"The stored file for {document.document_name!r} is missing "
                        "from this session."
                    )

                name = output_name_for(document.document_name, document.file_kind)
                destination = session.outputs_dir / f"{document.document_id}_{name}"
                outcome = redact_document(
                    stored.stored_path,
                    destination,
                    detections,
                    document,
                    replacement_text=self._settings.redaction_replacement_text,
                    redact_images=self._settings.pdf_redact_images,
                )
                if not destination.is_file():
                    raise RedactionError(
                        f"Redaction of {document.document_name!r} produced no output file. "
                        + (outcome.warnings[0] if outcome.warnings else ""),
                        details={"document_id": document.document_id},
                    )
                if outcome.applied == 0:
                    raise RedactionError(
                        f"No redactions could be applied to {document.document_name!r} "
                        "even though detections were approved. "
                        + (outcome.warnings[0] if outcome.warnings else ""),
                        details={"document_id": document.document_id},
                    )
                staged.append(
                    (
                        outcome,
                        destination,
                        name,
                        document,
                        detections,
                        retained_by_document.get(document.document_id, []),
                    )
                )
        except BaseException:
            for _, destination, _, _, _, _ in staged:
                destination.unlink(missing_ok=True)
            raise

        if not staged:
            raise RedactionError(
                "None of the approved detections belong to a document in this session.",
                details={"session_id": session_id},
            )

        for outcome, destination, name, document, detections, retained in staged:
            session.outcomes.append(outcome)
            self._store.register_output(
                session,
                document_id=document.document_id,
                source_name=document.document_name,
                output_name=name,
                path=destination,
            )
            session.verification[document.document_id] = verify_redaction(
                destination,
                source=document,
                approved=detections,
                retained=retained,
                max_chars=self._settings.max_extracted_chars,
            )

        session.status = SessionStatus.REDACTED
        logger.info(
            "redaction_complete",
            extra={
                "session_id": session.session_id,
                "outputs": len(session.outputs),
                "applied": sum(getattr(o, "applied", 0) for o in session.outcomes),
                "residual": any(
                    report.residual_text_found for report in session.verification.values()
                ),
            },
        )
        return session


def _as_error(failure: DocumentFailureRecord) -> RedactItError:
    """Re-raise the first document failure with its original status and code."""
    error = RedactionError(failure.message, details=failure.details)
    error.code = failure.code
    error.status_code = failure.status_code
    return error
