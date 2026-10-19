"""Tests for upload intake: filename hygiene, type validation and size caps.

Nothing here trusts a client-declared filename, extension or content-type. The
payload on disk must agree with the declared extension, and an oversized file
must be refused without leaving a partial copy behind.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import pytest
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile

from app.errors import (
    FileContentMismatchError,
    FileTooLargeError,
    UnsafeFilenameError,
    UnsupportedFileTypeError,
)
from app.uploads import (
    kind_from_extension,
    sanitize_filename,
    save_upload,
    validate_content,
)

from .conftest import build_docx, build_pdf

ALLOWED = {".pdf", ".docx"}
OLE_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def make_upload(filename: str | None, data: bytes) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": "application/octet-stream"}),
    )


def save(
    upload: UploadFile,
    tmp_path: Path,
    *,
    stored_name: str = "stored.bin",
    max_bytes: int = 2 * 1024 * 1024,
):
    return asyncio.run(
        save_upload(
            upload,
            destination_dir=tmp_path / "session",
            stored_name=stored_name,
            allowed_extensions=ALLOWED,
            max_file_size_bytes=max_bytes,
        )
    )


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report.pdf", "report.pdf"),
            ("my report (final).pdf", "my report _final_.pdf"),
            ("file.name.with.dots.docx", "file.name.with.dots.docx"),
            ("-.pdf", "-.pdf"),
        ],
    )
    def test_usable_names_pass_through(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "../../etc/passwd.pdf",
            "../../../secret.docx",
            "/etc/passwd.pdf",
            r"C:\Users\me\..\secret.docx",
            r"\\server\share\report.pdf",
            "sub/dir/report.pdf",
        ],
    )
    def test_directories_are_stripped_down_to_the_basename(self, raw: str) -> None:
        cleaned = sanitize_filename(raw)
        assert "/" not in cleaned and "\\" not in cleaned
        assert ".." not in cleaned

    @pytest.mark.parametrize("raw", ["..", ".", "/", "\\", "   ", "\t"])
    def test_names_that_resolve_to_nothing_are_refused(self, raw: str) -> None:
        with pytest.raises(UnsafeFilenameError) as info:
            sanitize_filename(raw)
        assert info.value.code == "UNSAFE_FILENAME"
        assert info.value.details.get("reason") == "empty_or_relative"

    @pytest.mark.parametrize("raw", ["", None])
    def test_a_missing_filename_is_refused(self, raw: str | None) -> None:
        with pytest.raises(UnsafeFilenameError) as info:
            sanitize_filename(raw)
        assert info.value.code == "UNSAFE_FILENAME"
        assert ".pdf or .docx" in info.value.message

    @pytest.mark.parametrize("raw", ["re\x00port.pdf", "tab\tname.pdf", "nl\nname.pdf"])
    def test_control_characters_are_refused(self, raw: str) -> None:
        with pytest.raises(UnsafeFilenameError) as info:
            sanitize_filename(raw)
        assert info.value.details.get("reason") == "control_characters"

    def test_a_name_with_no_usable_characters_is_refused(self) -> None:
        with pytest.raises(UnsafeFilenameError) as info:
            sanitize_filename("....")
        assert info.value.details.get("reason") == "no_usable_characters"

    def test_characters_outside_the_allow_list_are_replaced(self) -> None:
        cleaned = sanitize_filename("na\u00efve r\u00e9sum\u00e9.pdf")
        assert cleaned.endswith(".pdf")
        assert all(ch.isascii() for ch in cleaned)

    def test_a_long_name_is_shortened_but_keeps_its_extension(self) -> None:
        cleaned = sanitize_filename("a" * 200 + ".pdf")
        assert len(cleaned) <= 120
        # Losing the suffix here would make a valid PDF fail the type check with
        # a confusing "unsupported extension" error.
        assert cleaned.endswith(".pdf")
        assert kind_from_extension(cleaned, ALLOWED) == "pdf"

    def test_sanitizing_is_idempotent(self) -> None:
        once = sanitize_filename("../../my report (final).pdf")
        assert sanitize_filename(once) == once

    def test_the_result_never_contains_a_path_separator(self) -> None:
        for raw in ("..\\..\\x.pdf", "a/b/c.pdf", "a\x5cb.pdf"):
            cleaned = sanitize_filename(raw)
            assert "/" not in cleaned
            assert "\\" not in cleaned
            assert Path(cleaned).name == cleaned


class TestKindFromExtension:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [("a.pdf", "pdf"), ("A.PDF", "pdf"), ("a.Pdf", "pdf"), ("a.docx", "docx"), ("A.DOCX", "docx")],
    )
    def test_accepted_extensions(self, filename: str, expected: str) -> None:
        assert kind_from_extension(filename, ALLOWED) == expected

    @pytest.mark.parametrize(
        "filename",
        ["a.txt", "a.pdf.exe", "a", "tar.gz", "a.doc", "a.pdf ", ".hidden.pdf.bak"],
    )
    def test_rejected_extensions(self, filename: str) -> None:
        with pytest.raises(UnsupportedFileTypeError) as info:
            kind_from_extension(filename, ALLOWED)
        assert info.value.status_code == 415
        assert info.value.details["allowed_extensions"] == [".docx", ".pdf"]
        assert ".docx, .pdf" in info.value.message

    def test_a_double_extension_does_not_sneak_through(self) -> None:
        # The last suffix decides, so "report.pdf.exe" is an executable, not a PDF.
        assert Path("report.pdf.exe").suffix == ".exe"
        with pytest.raises(UnsupportedFileTypeError):
            kind_from_extension("report.pdf.exe", ALLOWED)


class TestContentValidation:
    def write(self, tmp_path: Path, name: str, data: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    def test_a_real_pdf_passes(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "ok.pdf", build_pdf())
        assert validate_content(path, "pdf", "ok.pdf") is None

    def test_a_real_docx_passes(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "ok.docx", build_docx())
        assert validate_content(path, "docx", "ok.docx") is None

    def test_a_pdf_header_after_leading_junk_is_still_accepted(self, tmp_path: Path) -> None:
        # Some tools prepend a byte-order mark or a comment before "%PDF-".
        path = self.write(tmp_path, "bom.pdf", b"\xef\xbb\xbf" + build_pdf())
        assert validate_content(path, "pdf", "bom.pdf") is None

    def test_a_docx_renamed_to_pdf_is_caught(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "fake.pdf", build_docx())
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "pdf", "fake.pdf")
        assert info.value.status_code == 415
        assert info.value.details["detected"] == "zip"

    def test_a_legacy_ole_document_named_pdf_is_caught(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "old.pdf", OLE_HEADER + b"\x00" * 512)
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "pdf", "old.pdf")
        assert info.value.details["detected"] == "ole_compound_file"

    def test_arbitrary_bytes_named_pdf_are_caught(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "junk.pdf", b"this is not a pdf at all")
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "pdf", "junk.pdf")
        assert info.value.details["detected"] == "unknown"
        assert "PDF header" in info.value.message

    def test_arbitrary_bytes_named_docx_are_caught(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "junk.docx", b"this is not a zip at all")
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "docx", "junk.docx")
        assert info.value.details["detected"] == "unknown"

    def test_a_legacy_doc_named_docx_explains_itself(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "old.docx", OLE_HEADER + b"\x00" * 512)
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "docx", "old.docx")
        assert info.value.details["detected"] == "ole_compound_file"
        assert "legacy .doc" in info.value.message

    def test_a_zip_that_is_not_a_word_document_is_caught(self, tmp_path: Path) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "just a zip")
        path = self.write(tmp_path, "archive.docx", buffer.getvalue())
        with pytest.raises(FileContentMismatchError) as info:
            validate_content(path, "docx", "archive.docx")
        assert info.value.details["detected"] == "zip_not_docx"
        assert "word/document.xml" in info.value.message

    def test_a_corrupt_zip_is_caught(self, tmp_path: Path) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")
            archive.writestr("[Content_Types].xml", "<Types/>")
        truncated = buffer.getvalue()[: len(buffer.getvalue()) // 2]
        path = self.write(tmp_path, "corrupt.docx", truncated)
        with pytest.raises(FileContentMismatchError):
            validate_content(path, "docx", "corrupt.docx")

    def test_an_unknown_kind_is_refused(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "x.bin", b"data")
        with pytest.raises(UnsupportedFileTypeError):
            validate_content(path, "exe", "x.bin")

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "empty.pdf", b"")
        with pytest.raises(FileContentMismatchError):
            validate_content(path, "pdf", "empty.pdf")


class TestSaveUpload:
    def test_a_valid_pdf_is_stored_under_the_generated_name(self, tmp_path: Path) -> None:
        data = build_pdf()
        path, display_name, kind, size = save(
            make_upload("../../report.pdf", data), tmp_path, stored_name="doc_abc.pdf"
        )
        assert path == tmp_path / "session" / "doc_abc.pdf"
        assert path.name == "doc_abc.pdf"
        assert display_name == "report.pdf"
        assert kind == "pdf"
        assert size == len(data)
        assert path.read_bytes() == data

    def test_a_valid_docx_is_stored(self, tmp_path: Path) -> None:
        data = build_docx()
        path, display_name, kind, size = save(
            make_upload("report.docx", data), tmp_path, stored_name="doc_xyz.docx"
        )
        assert (display_name, kind, size) == ("report.docx", "docx", len(data))
        assert path.read_bytes() == data

    def test_the_stored_name_never_comes_from_the_client(self, tmp_path: Path) -> None:
        path, _, _, _ = save(
            make_upload("../../windows/system32/evil.pdf", build_pdf()),
            tmp_path,
            stored_name="doc_token.pdf",
        )
        assert path.name == "doc_token.pdf"
        assert path.parent == tmp_path / "session"

    def test_an_unsupported_extension_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            save(make_upload("notes.txt", b"hello"), tmp_path)
        assert not (tmp_path / "session").exists() or list((tmp_path / "session").iterdir()) == []

    def test_a_missing_filename_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafeFilenameError):
            save(make_upload(None, build_pdf()), tmp_path)

    def test_an_empty_upload_is_refused_and_nothing_is_left_behind(self, tmp_path: Path) -> None:
        with pytest.raises(FileContentMismatchError) as info:
            save(make_upload("empty.pdf", b""), tmp_path)
        assert "empty (0 bytes)" in info.value.message
        assert list((tmp_path / "session").iterdir()) == []

    def test_a_mismatched_payload_is_refused_and_nothing_is_left_behind(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileContentMismatchError):
            save(make_upload("fake.pdf", build_docx()), tmp_path)
        assert list((tmp_path / "session").iterdir()) == []

    def test_an_oversized_file_is_refused_at_the_cap(self, tmp_path: Path) -> None:
        limit = 512 * 1024
        oversized = build_pdf() + b"\x00" * (limit + 4096)
        with pytest.raises(FileTooLargeError) as info:
            save(make_upload("big.pdf", oversized), tmp_path, max_bytes=limit)
        assert info.value.status_code == 413
        assert info.value.details["limit_bytes"] == limit
        assert info.value.details["received_bytes"] > limit
        assert "MAX_FILE_SIZE_BYTES" in info.value.message

    def test_an_oversized_upload_leaves_no_partial_file(self, tmp_path: Path) -> None:
        # The cap is enforced while streaming, so the refusal must also clean up.
        with pytest.raises(FileTooLargeError):
            save(make_upload("big.pdf", b"\x00" * 1024 * 1024), tmp_path, max_bytes=64 * 1024)
        assert list((tmp_path / "session").iterdir()) == []

    def test_a_file_exactly_at_the_cap_is_accepted(self, tmp_path: Path) -> None:
        data = build_pdf()
        _, _, _, size = save(make_upload("exact.pdf", data), tmp_path, max_bytes=len(data))
        assert size == len(data)

    def test_a_file_one_byte_over_the_cap_is_refused(self, tmp_path: Path) -> None:
        data = build_pdf()
        with pytest.raises(FileTooLargeError):
            save(make_upload("over.pdf", data), tmp_path, max_bytes=len(data) - 1)

    def test_the_destination_directory_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "session"
        path, _, _, _ = asyncio.run(
            save_upload(
                make_upload("report.pdf", build_pdf()),
                destination_dir=target,
                stored_name="doc_1.pdf",
                allowed_extensions=ALLOWED,
                max_file_size_bytes=1024 * 1024,
            )
        )
        assert path.parent == target
        assert path.exists()
