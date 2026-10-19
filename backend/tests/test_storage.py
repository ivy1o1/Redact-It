"""Tests for the session / temporary-file layer.

This is the security-critical part of the system: sessions own isolated
directories under ``STORAGE_ROOT``, identifiers must be opaque and unguessable,
and download tokens must never leak a file that has escaped its session's own
``outputs/`` directory. Every assertion here was written against behaviour
measured by probing the real code, not against assumed semantics.

Only synthetic sessions and files are used, and everything is created under
pytest's ``tmp_path`` (via the ``settings`` / ``store`` fixtures), so nothing is
ever written into the repository.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.errors import (
    DownloadExpiredError,
    DownloadNotFoundError,
    SessionExpiredError,
    SessionNotFoundError,
)
from app.extraction.base import ExtractedDocument
from app.matching.engine import Detection, DetectionLocation
from app.storage import (
    Session,
    SessionStatus,
    SessionStore,
    new_document_id,
    new_download_token,
    new_session_id,
)


# --------------------------------------------------------------------- helpers
def make_expired(session: Session) -> Session:
    session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    return session


def write_output(store: SessionStore, session: Session, name: str, data: bytes):
    path = session.outputs_dir / name
    path.write_bytes(data)
    return store.register_output(
        session,
        document_id="doc_1",
        source_name="original.pdf",
        output_name=name,
        path=path,
    )


# ---------------------------------------------------------------- identifiers
class TestIdentifiers:
    def test_each_identifier_carries_its_prefix(self) -> None:
        assert new_session_id().startswith("sess_")
        assert new_document_id().startswith("doc_")
        assert new_download_token().startswith("dl_")

    def test_identifiers_are_long_enough_to_be_unguessable(self) -> None:
        # Measured lengths: sess_ + 22 chars, doc_ + 11 chars, dl_ + 32 chars.
        # The bodies come from secrets.token_urlsafe(16/8/24).
        assert len(new_session_id()) >= 20
        assert len(new_document_id()) >= 12
        assert len(new_download_token()) >= 30

    def test_identifiers_are_opaque_rather_than_sequential_or_numeric(self) -> None:
        for generator in (new_session_id, new_document_id, new_download_token):
            value = generator()
            body = value.split("_", 1)[1]
            assert not body.isdigit(), "opaque ids must not be a plain counter"
            assert any(c.isalpha() for c in body)

    def test_identifiers_are_unique_across_many_samples(self) -> None:
        for generator in (new_session_id, new_document_id, new_download_token):
            samples = [generator() for _ in range(1000)]
            assert len(set(samples)) == len(samples)


# ------------------------------------------------------------- create_session
class TestCreateSession:
    def test_session_gets_an_isolated_work_dir_with_uploads_and_outputs(
        self, store: SessionStore
    ) -> None:
        session = store.create_session("remove names")
        assert session.work_dir == store.root / session.session_id
        assert session.work_dir.is_dir()
        assert session.uploads_dir.is_dir()
        assert session.outputs_dir.is_dir()

    def test_work_dirs_live_entirely_under_the_store_root(self, store: SessionStore) -> None:
        session = store.create_session("remove names")
        root_resolved = store.root.resolve()
        assert session.uploads_dir.resolve().is_relative_to(root_resolved)
        assert session.outputs_dir.resolve().is_relative_to(root_resolved)

    def test_expires_at_is_set_from_session_ttl_minutes(self, store: SessionStore) -> None:
        session = store.create_session("remove names")
        # make_settings pins SESSION_TTL_MINUTES to 30, so the gap is 1800s.
        assert (session.expires_at - session.created_at) == timedelta(minutes=30)

    def test_original_written_into_uploads_never_escapes_the_session_dir(
        self, store: SessionStore
    ) -> None:
        session = store.create_session("remove names")
        original = session.uploads_dir / "report.pdf"
        original.write_bytes(b"fake pdf bytes")
        assert original.resolve().is_relative_to(session.work_dir.resolve())
        # Purging the session takes the original with it.
        store.purge_session(session.session_id)
        assert not session.work_dir.exists()


# -------------------------------------------------------------------------- get
class TestGet:
    def test_known_session_is_returned(self, store: SessionStore) -> None:
        session = store.create_session("remove names")
        assert store.get(session.session_id) is session

    def test_expired_session_is_purged_and_raises(self, store: SessionStore) -> None:
        session = make_expired(store.create_session("remove names"))
        work_dir = session.work_dir
        with pytest.raises(SessionExpiredError):
            store.get(session.session_id)
        assert not work_dir.exists()

    def test_unknown_but_wellformed_id_is_reported_as_expired(
        self, store: SessionStore
    ) -> None:
        # A purged session is indistinguishable from one that was never created,
        # so the store maps any well-formed sess_ id it does not know to
        # SessionExpiredError. This keeps the "session is gone" signal uniform
        # for clients (they must re-run) instead of leaking whether an id existed.
        with pytest.raises(SessionExpiredError):
            store.get("sess_" + "a" * 22)

    def test_malformed_id_is_reported_as_not_found(self, store: SessionStore) -> None:
        # A garbage id cannot be a session id at all, so it is a distinct error
        # from "expired": there was never anything to expire.
        with pytest.raises(SessionNotFoundError):
            store.get("definitely-not-a-session")


# ---------------------------------------------------------------- purge_session
class TestPurgeSession:
    def test_purge_removes_the_directory_tree_from_disk(self, store: SessionStore) -> None:
        session = store.create_session("remove names")
        work_dir = session.work_dir
        assert work_dir.is_dir()
        assert store.purge_session(session.session_id) is True
        assert not work_dir.exists()

    def test_purge_is_idempotent_for_unknown_ids(self, store: SessionStore) -> None:
        assert store.purge_session("sess_doesnotexist") is False
        assert store.purge_session("nonsense") is False

    def test_purge_drops_that_sessions_download_tokens(self, store: SessionStore) -> None:
        session = store.create_session("remove names")
        output = write_output(store, session, "redacted.pdf", b"data")
        assert store.snapshot()["download_tokens"] == 1
        store.purge_session(session.session_id)
        assert store.snapshot()["download_tokens"] == 0
        with pytest.raises(DownloadNotFoundError):
            store.resolve_download(output.token)


# ------------------------------------------------------------- purge family
class TestPurgeFamily:
    def test_purge_expired_removes_only_expired_sessions(self, store: SessionStore) -> None:
        live = store.create_session("keep")
        gone = make_expired(store.create_session("drop"))
        removed = store.purge_expired()
        assert removed == 1
        assert live.work_dir.exists()
        assert not gone.work_dir.exists()

    def test_purge_all_removes_every_session(self, store: SessionStore) -> None:
        dirs = [store.create_session("x").work_dir for _ in range(3)]
        assert store.purge_all() == 3
        assert store.snapshot() == {"sessions": 0, "download_tokens": 0}
        for path in dirs:
            assert not path.exists()

    def test_snapshot_counts_sessions_and_tokens(self, store: SessionStore) -> None:
        session = store.create_session("x")
        write_output(store, session, "a.pdf", b"1")
        write_output(store, session, "b.pdf", b"2")
        assert store.snapshot() == {"sessions": 1, "download_tokens": 2}

    def test_purge_orphans_removes_stray_sess_dirs_but_keeps_live_and_foreign(
        self, store: SessionStore
    ) -> None:
        live = store.create_session("live").work_dir
        stray = store.root / "sess_stray_from_a_previous_process"
        stray.mkdir()
        (stray / "leftover.txt").write_text("junk")
        foreign = store.root / "not-a-session-dir"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("must survive")

        assert store.purge_orphans() == 1
        assert not stray.exists()
        assert live.exists()
        assert foreign.exists()


# ---------------------------------------------------------------- register_output
class TestRegisterOutput:
    def test_records_size_independent_sha256_ttl_and_token(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        session = store.create_session("x")
        payload = b"redacted document bytes 12345"
        output = write_output(store, session, "redacted.pdf", payload)

        assert output.token.startswith("dl_")
        assert output.size_bytes == len(payload)
        # sha256 must equal an independent hashlib computation over the same file.
        assert output.sha256 == hashlib.sha256(payload).hexdigest()
        # DOWNLOAD_TTL_MINUTES is pinned to 15 in make_settings, so 900s.
        assert (output.expires_at - output.created_at) == timedelta(minutes=15)
        assert output in session.outputs


# --------------------------------------------------------------- resolve_download
class TestResolveDownload:
    def test_happy_path_returns_session_and_output(self, store: SessionStore) -> None:
        session = store.create_session("x")
        output = write_output(store, session, "redacted.pdf", b"data")
        resolved_session, resolved_output = store.resolve_download(output.token)
        assert resolved_session is session
        assert resolved_output is output

    def test_unknown_token_raises_not_found(self, store: SessionStore) -> None:
        with pytest.raises(DownloadNotFoundError):
            store.resolve_download("dl_this-token-was-never-issued")

    def test_expired_token_raises_and_deletes_the_file(self, store: SessionStore) -> None:
        session = store.create_session("x")
        output = write_output(store, session, "redacted.pdf", b"data")
        output.expires_at = output.created_at - timedelta(seconds=1)
        path = output.path
        with pytest.raises(DownloadExpiredError) as info:
            store.resolve_download(output.token)
        assert not path.exists()
        # Measured: the expired branch reports token_prefix and expired_at.
        assert set(info.value.details) >= {"token_prefix", "expired_at"}

    def test_path_escape_guard_blocks_output_outside_outputs_dir(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        # Seed the output so its recorded path points at a real file that lives
        # OUTSIDE session.outputs_dir. This simulates a corrupted store (a
        # tampered token map / mis-registered path); the resolve_download
        # is_relative_to check is defence in depth so such a file is never served.
        session = store.create_session("x")
        output = write_output(store, session, "redacted.pdf", b"data")
        outside = tmp_path / "leaked.pdf"
        outside.write_bytes(b"should never be served")
        output.path = outside
        with pytest.raises(DownloadNotFoundError):
            store.resolve_download(output.token)

    def test_vanished_file_raises_not_found(self, store: SessionStore) -> None:
        session = store.create_session("x")
        output = write_output(store, session, "redacted.pdf", b"data")
        output.path.unlink()
        with pytest.raises(DownloadNotFoundError):
            store.resolve_download(output.token)

    def test_expired_session_blocks_its_live_token(self, store: SessionStore) -> None:
        # resolve_download calls get() first, so an expired session surfaces as
        # SessionExpiredError before its (still unexpired) token is checked.
        session = store.create_session("x")
        output = write_output(store, session, "redacted.pdf", b"data")
        make_expired(session)
        with pytest.raises(SessionExpiredError):
            store.resolve_download(output.token)


# ------------------------------------------------------- purge_expired_downloads
class TestPurgeExpiredDownloads:
    def test_removes_expired_outputs_and_keeps_live_ones(self, store: SessionStore) -> None:
        session = store.create_session("x")
        live = write_output(store, session, "live.pdf", b"keep me")
        dead = write_output(store, session, "dead.pdf", b"drop me")
        dead.expires_at = dead.created_at - timedelta(seconds=1)
        dead_path = dead.path

        assert store.purge_expired_downloads() == 1
        assert not dead_path.exists()
        assert live.path.exists()
        assert [o.token for o in session.outputs] == [live.token]
        with pytest.raises(DownloadNotFoundError):
            store.resolve_download(dead.token)


# -------------------------------------------------------------------- threading
class TestThreading:
    def test_concurrent_create_and_purge_keep_registry_and_disk_consistent(
        self, store: SessionStore
    ) -> None:
        created: list[str] = []
        dirs: list[Path] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            try:
                barrier.wait()
                session = store.create_session(f"thread-{index}")
                with store._lock:
                    created.append(session.session_id)
                    dirs.append(session.work_dir)
                assert store.purge_session(session.session_id) is True
            except BaseException as exc:  # noqa: BLE001 - surfaced via the list
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(created) == 8
        # Registry drained, and every working directory really disappeared.
        assert store.snapshot() == {"sessions": 0, "download_tokens": 0}
        assert all(not path.exists() for path in dirs)


# ------------------------------------------------------------- session helpers
def _detection(detection_id: str, matched_text: str) -> Detection:
    return Detection(
        detection_id=detection_id,
        matched_text=matched_text,
        category="named text",
        strategy="exact",
        confidence=1.0,
        location=DetectionLocation(
            document_id="doc_a",
            document_name="report.pdf",
            file_kind="pdf",
            segment_index=0,
            page=1,
            char_start=0,
            char_end=len(matched_text),
        ),
    )


class TestSessionHelpers:
    def test_detection_by_id(self, store: SessionStore) -> None:
        session = store.create_session("x")
        session.detections = [_detection("det_1", "Alice Johnson"), _detection("det_2", "555-000-1111")]
        assert session.detection_by_id("det_2").matched_text == "555-000-1111"
        assert session.detection_by_id("det_missing") is None

    def test_document_by_id(self, store: SessionStore) -> None:
        session = store.create_session("x")
        doc = ExtractedDocument(
            document_id="doc_a",
            document_name="report.pdf",
            file_kind="pdf",
            size_bytes=10,
            sha256="",
        )
        session.extracted = [doc]
        assert session.document_by_id("doc_a") is doc
        assert session.document_by_id("doc_zzz") is None

    def test_is_expired_boundary(self, store: SessionStore) -> None:
        # Measured: is_expired is `now >= expires_at`, so expiry exactly at the
        # deadline counts as expired; one second in the future does not.
        future = store.create_session("x")
        future.expires_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        assert future.is_expired is False

        past = store.create_session("y")
        past.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert past.is_expired is True

    def test_default_status_is_analysed(self, store: SessionStore) -> None:
        assert store.create_session("x").status is SessionStatus.ANALYSED
