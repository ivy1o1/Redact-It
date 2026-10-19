"""Error taxonomy and the uniform JSON error envelope.

Every failure the frontend can act on is a :class:`RedactItError` subclass with a
stable machine-readable ``code``, an HTTP status and an actionable message.
"""

from __future__ import annotations

from typing import Any


class RedactItError(Exception):
    """Base class for all expected, client-facing failures."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.headers = headers or {}


# --------------------------------------------------------------------- uploads
class UploadError(RedactItError):
    code = "UPLOAD_INVALID"
    status_code = 400


class NoFilesUploadedError(UploadError):
    code = "NO_FILES_UPLOADED"

    def __init__(self) -> None:
        super().__init__(
            "No files were uploaded. Send at least one file in the 'files' "
            "multipart field."
        )


class TooManyFilesError(UploadError):
    code = "TOO_MANY_FILES"


class FileTooLargeError(UploadError):
    code = "FILE_TOO_LARGE"
    status_code = 413


class UnsupportedFileTypeError(UploadError):
    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 415


class FileContentMismatchError(UploadError):
    code = "FILE_CONTENT_MISMATCH"
    status_code = 415


class UnsafeFilenameError(UploadError):
    code = "UNSAFE_FILENAME"


# ------------------------------------------------------------------ extraction
class ExtractionError(RedactItError):
    code = "EXTRACTION_FAILED"
    status_code = 422


class ScannedDocumentError(ExtractionError):
    code = "SCANNED_DOCUMENT_UNSUPPORTED"

    def __init__(self, document_name: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"{document_name!r} has no extractable text layer (it appears to be a "
            "scan or image-only PDF). RedactIt does not perform OCR, so nothing was "
            "processed and no redactions can be proposed for it. Run OCR on the "
            "document first, or remove it from the batch.",
            details=details,
        )


class EncryptedDocumentError(ExtractionError):
    code = "ENCRYPTED_DOCUMENT_UNSUPPORTED"


class EmptyDocumentError(ExtractionError):
    code = "EMPTY_DOCUMENT"


class ExtractionBudgetError(ExtractionError):
    """The character budget was smaller than the document's first page."""

    code = "EXTRACTION_BUDGET_TOO_SMALL"

    def __init__(self, document_name: str, *, max_chars: int, needed: int) -> None:
        super().__init__(
            f"{document_name!r} has a text layer, but MAX_EXTRACTED_CHARS={max_chars} "
            f"is smaller than its first extractable page or paragraph ({needed} "
            "characters), so nothing could be analysed. Raise MAX_EXTRACTED_CHARS or "
            "split the document.",
            details={
                "document_name": document_name,
                "max_extracted_chars": max_chars,
                "first_block_chars": needed,
            },
        )


# --------------------------------------------------------------------- session
class SessionNotFoundError(RedactItError):
    code = "SESSION_NOT_FOUND"
    status_code = 404

    def __init__(self, session_id: str) -> None:
        super().__init__(f"No active analysis session with id {session_id!r}.")


class SessionExpiredError(RedactItError):
    code = "SESSION_EXPIRED"
    status_code = 410

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id!r} expired and its files were deleted. "
            "Re-run the analysis to get a fresh session.",
        )


class ReviewRequiredError(RedactItError):
    code = "REVIEW_REQUIRED"
    status_code = 409


class InvalidSelectionError(RedactItError):
    code = "INVALID_DETECTION_SELECTION"
    status_code = 422


class StaleDetectionIdError(RedactItError):
    code = "STALE_DETECTION_ID"
    status_code = 409


class NoApprovedRedactionsError(RedactItError):
    code = "NO_APPROVED_REDACTIONS"
    status_code = 409


class AlreadyRedactedError(RedactItError):
    code = "ALREADY_REDACTED"
    status_code = 409


# ------------------------------------------------------------------- downloads
class DownloadNotFoundError(RedactItError):
    code = "DOWNLOAD_NOT_FOUND"
    status_code = 404


class DownloadExpiredError(RedactItError):
    code = "DOWNLOAD_EXPIRED"
    status_code = 410


# ------------------------------------------------------------ instruction / LLM
class InstructionError(RedactItError):
    code = "INSTRUCTION_INVALID"
    status_code = 422


class EmptyInstructionError(InstructionError):
    code = "EMPTY_INSTRUCTION"

    def __init__(self) -> None:
        super().__init__(
            "The instruction is empty. Describe what to redact in plain English, "
            "for example: 'Remove all student names and phone numbers, keep grades.'"
        )


class InstructionAmbiguousError(InstructionError):
    code = "INSTRUCTION_AMBIGUOUS"


class InstructionUnusableError(InstructionError):
    code = "INSTRUCTION_UNUSABLE"


class ModelError(RedactItError):
    code = "MODEL_ERROR"
    status_code = 503


class ModelUnavailableError(ModelError):
    code = "MODEL_UNAVAILABLE"

    def __init__(self, base_url: str, *, reason: str | None = None) -> None:
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"Cannot reach the local model server at {base_url}{detail}. "
            "Start llama-server and confirm it answers /health, then retry. "
            "RedactIt will not fall back to a cloud model.",
            details={"base_url": base_url, "reason": reason},
        )


class ModelLoadingError(ModelError):
    code = "MODEL_LOADING"

    def __init__(self, base_url: str) -> None:
        super().__init__(
            f"The model server at {base_url} is still loading its model. "
            "Wait for /health to return 'ok' and retry.",
            details={"base_url": base_url},
        )


class ModelTimeoutError(ModelError):
    code = "MODEL_TIMEOUT"
    status_code = 504


class ModelAuthError(ModelError):
    code = "MODEL_AUTH_ERROR"
    status_code = 502


class ModelResponseError(ModelError):
    code = "MALFORMED_MODEL_RESPONSE"
    status_code = 502


class ModelNotFoundError(ModelError):
    code = "MODEL_NOT_LOADED"

    def __init__(self, expected: str, base_url: str) -> None:
        super().__init__(
            f"The local server at {base_url} is responding but does not report a "
            f"model matching {expected!r}. Check that llama-server was started with "
            "the intended GGUF and that LLM_MODEL matches its id or alias.",
            details={"expected_model": expected, "base_url": base_url},
        )


# ------------------------------------------------------------------- redaction
class RedactionError(RedactItError):
    code = "REDACTION_FAILED"
    status_code = 500


class RedactionVerificationError(RedactionError):
    code = "REDACTION_VERIFICATION_INCOMPLETE"
