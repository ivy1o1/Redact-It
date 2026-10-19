"""In-memory session store backed by per-session temporary working directories.

No database: sessions live in a dict guarded by a lock, and every session owns an
isolated directory under ``STORAGE_ROOT``. Originals are written under
``uploads/`` and are never modified; generated files go to ``outputs/``.

Identifiers are opaque (``secrets.token_urlsafe``) so neither session ids nor
download tokens are guessable or enumerable.
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import (
    DownloadExpiredError,
    DownloadNotFoundError,
    SessionExpiredError,
    SessionNotFoundError,
)
from .extraction.base import ExtractedDocument
from .logging_setup import get_logger

logger = get_logger("storage")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_session_id() -> str:
    return f"sess_{secrets.token_urlsafe(16)}"


def new_document_id() -> str:
    return f"doc_{secrets.token_urlsafe(8)}"


def new_download_token() -> str:
    return f"dl_{secrets.token_urlsafe(24)}"


class SessionStatus(str, Enum):
    ANALYSED = "analysed"
    REVIEWED = "reviewed"
    REDACTED = "redacted"
    FAILED = "failed"


@dataclass(slots=True)
class StoredDocument:
    document_id: str
    display_name: str
    file_kind: str
    stored_path: Path
    size_bytes: int
    sha256: str


@dataclass(slots=True)
class DocumentFailureRecord:
    """Why one uploaded file could not be analysed."""

    document_name: str
    code: str
    message: str
    status_code: int = 422
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputFile:
    token: str
    document_id: str
    source_name: str
    output_name: str
    path: Path
    size_bytes: int
    sha256: str
    created_at: datetime
    expires_at: datetime


@dataclass(slots=True)
class Review:
    approved_detection_ids: list[str] = field(default_factory=list)
    rejected_detection_ids: list[str] = field(default_factory=list)
    reviewed_at: datetime | None = None


@dataclass(slots=True)
class Session:
    session_id: str
    instruction: str
    work_dir: Path
    created_at: datetime
    expires_at: datetime
    status: SessionStatus = SessionStatus.ANALYSED
    documents: list[StoredDocument] = field(default_factory=list)
    extracted: list[ExtractedDocument] = field(default_factory=list)
    document_failures: list[DocumentFailureRecord] = field(default_factory=list)
    interpretation: Any = None
    interpretation_mode: str = "model"
    plan: Any = None
    match_result: Any = None
    detections: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review: Review | None = None
    review_warnings: list[str] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)
    outputs: list[OutputFile] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)

    @property
    def uploads_dir(self) -> Path:
        return self.work_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.work_dir / "outputs"

    @property
    def is_expired(self) -> bool:
        return _now() >= self.expires_at

    def detection_by_id(self, detection_id: str) -> Any | None:
        for detection in self.detections:
            if detection.detection_id == detection_id:
                return detection
        return None

    def document_by_id(self, document_id: str) -> ExtractedDocument | None:
        for document in self.extracted:
            if document.document_id == document_id:
                return document
        return None


class SessionStore:
    """Thread-safe registry of live sessions and their working directories."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}
        self._tokens: dict[str, str] = {}  # download token -> session id
        self._lock = threading.RLock()
        self._root: Path | None = None

    # ------------------------------------------------------------------ setup
    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = self._settings.resolved_storage_root / "sessions"
            self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    # --------------------------------------------------------------- sessions
    def create_session(self, instruction: str) -> Session:
        session_id = new_session_id()
        now = _now()
        work_dir = self.root / session_id
        (work_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (work_dir / "outputs").mkdir(parents=True, exist_ok=True)

        session = Session(
            session_id=session_id,
            instruction=instruction,
            work_dir=work_dir,
            created_at=now,
            expires_at=now + timedelta(minutes=self._settings.session_ttl_minutes),
        )
        with self._lock:
            self._sessions[session_id] = session
        logger.info(
            "session_created",
            extra={"session_id": session_id, "ttl_minutes": self._settings.session_ttl_minutes},
        )
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            # A purged session is indistinguishable from an unknown one; report
            # it as expired when the id is well formed so clients know to re-run.
            if session_id.startswith("sess_"):
                raise SessionExpiredError(session_id)
            raise SessionNotFoundError(session_id)
        if session.is_expired:
            self.purge_session(session_id)
            raise SessionExpiredError(session_id)
        return session

    def purge_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            for token in [t for t, sid in self._tokens.items() if sid == session_id]:
                self._tokens.pop(token, None)
        if session is None:
            return False
        _remove_tree(session.work_dir)
        logger.info("session_purged", extra={"session_id": session_id})
        return True

    def purge_expired(self) -> int:
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for session_id in expired:
            self.purge_session(session_id)
        if expired:
            logger.info("expired_sessions_purged", extra={"count": len(expired)})
        return len(expired)

    def purge_orphans(self) -> int:
        """Remove working directories left behind by a previous process."""
        root = self.root
        if not root.exists():
            return 0
        removed = 0
        with self._lock:
            live = set(self._sessions)
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("sess_") and child.name not in live:
                if _remove_tree(child):
                    removed += 1
        if removed:
            logger.info("orphaned_workdirs_removed", extra={"count": removed})
        return removed

    def purge_all(self) -> int:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.purge_session(session_id)
        return len(session_ids)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "download_tokens": len(self._tokens),
            }

    # -------------------------------------------------------------- downloads
    def register_output(
        self,
        session: Session,
        *,
        document_id: str,
        source_name: str,
        output_name: str,
        path: Path,
    ) -> OutputFile:
        now = _now()
        expires_at = now + timedelta(minutes=self._settings.download_ttl_minutes)
        output = OutputFile(
            token=new_download_token(),
            document_id=document_id,
            source_name=source_name,
            output_name=output_name,
            path=path,
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            session.outputs.append(output)
            self._tokens[output.token] = session.session_id
        return output

    def resolve_download(self, token: str) -> tuple[Session, OutputFile]:
        with self._lock:
            session_id = self._tokens.get(token)
        if session_id is None:
            raise DownloadNotFoundError(token)

        session = self.get(session_id)  # raises if expired, and purges
        output = next((o for o in session.outputs if o.token == token), None)
        if output is None:
            raise DownloadNotFoundError(token)
        if _now() >= output.expires_at:
            with self._lock:
                self._tokens.pop(token, None)
            _remove_file(output.path)
            raise DownloadExpiredError(
                f"This download link expired at {output.expires_at.isoformat()}. "
                "Re-run the redaction to generate a fresh link.",
                details={"token_prefix": token[:8], "expired_at": output.expires_at.isoformat()},
            )

        # Defence in depth: the token came from the store, but never serve a path
        # that has escaped the session's own outputs directory.
        resolved = output.path.resolve()
        if not resolved.is_relative_to(session.outputs_dir.resolve()):
            logger.error(
                "download_path_escape_blocked", extra={"session_id": session.session_id}
            )
            raise DownloadNotFoundError(token)
        if not resolved.is_file():
            raise DownloadNotFoundError(token)
        return session, output

    def purge_expired_downloads(self) -> int:
        now = _now()
        removed = 0
        with self._lock:
            items = list(self._tokens.items())
        for token, session_id in items:
            with self._lock:
                session = self._sessions.get(session_id)
            if session is None:
                with self._lock:
                    self._tokens.pop(token, None)
                continue
            for output in list(session.outputs):
                if output.token == token and now >= output.expires_at:
                    with self._lock:
                        self._tokens.pop(token, None)
                    if _remove_file(output.path):
                        removed += 1
                    session.outputs.remove(output)
        return removed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def _remove_file(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover - locked file on Windows
        logger.warning("file_remove_failed", extra={"exists": path.exists()})
        return False
