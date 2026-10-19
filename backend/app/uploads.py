"""Upload intake: filename hygiene, extension and content validation, size caps.

Uploaded bytes are streamed to disk in bounded chunks so a hostile or merely
large file cannot exhaust memory. Declared extensions are never trusted on their
own - the payload is sniffed and must agree with the extension.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from fastapi import UploadFile

from .errors import (
    FileContentMismatchError,
    FileTooLargeError,
    UnsafeFilenameError,
    UnsupportedFileTypeError,
)
from .logging_setup import get_logger

logger = get_logger("uploads")

_CHUNK = 256 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")
_MAX_NAME_LENGTH = 120

_KIND_BY_EXTENSION = {".pdf": "pdf", ".docx": "docx"}

_OLE_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_HEADER = b"PK\x03\x04"


def sanitize_filename(raw: str | None) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips directories, drive letters and anything outside a conservative
    character allow-list. The result is used for display and for the stored
    filename, never to build a path from user input directly.
    """
    if not raw:
        raise UnsafeFilenameError(
            "Uploaded file has no filename. Every file must carry a name with a "
            ".pdf or .docx extension."
        )

    # Clients sometimes send absolute paths or backslash separators.
    name = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise UnsafeFilenameError(
            f"Filename {raw!r} is not a usable file name.",
            details={"reason": "empty_or_relative"},
        )
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        raise UnsafeFilenameError(
            "Filename contains control characters.",
            details={"reason": "control_characters"},
        )

    cleaned = _UNSAFE.sub("_", name).strip(" .")
    if not cleaned:
        raise UnsafeFilenameError(
            f"Filename {raw!r} contains no usable characters.",
            details={"reason": "no_usable_characters"},
        )
    if len(cleaned) > _MAX_NAME_LENGTH:
        # The suffix survives the cut, otherwise a merely long filename would be
        # rejected later as an unsupported file type.
        suffix = Path(cleaned).suffix
        cleaned = cleaned[: _MAX_NAME_LENGTH - len(suffix)].rstrip(" .") + suffix
    return cleaned


def kind_from_extension(filename: str, allowed_extensions: set[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise UnsupportedFileTypeError(
            f"{filename!r} has an unsupported extension. Accepted types: {allowed}.",
            details={"filename": filename, "allowed_extensions": sorted(allowed_extensions)},
        )
    return _KIND_BY_EXTENSION[suffix]


def _validate_pdf_header(path: Path, filename: str) -> None:
    with path.open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF-" not in head:
        if head.startswith(_OLE_HEADER):
            raise FileContentMismatchError(
                f"{filename!r} is an older OLE/Microsoft binary document, not a PDF. "
                "Re-save it as PDF and upload again.",
                details={"filename": filename, "detected": "ole_compound_file"},
            )
        if head.startswith(_ZIP_HEADER):
            raise FileContentMismatchError(
                f"{filename!r} has a .pdf extension but its contents are a ZIP "
                "package (possibly a .docx). Upload the correct file type.",
                details={"filename": filename, "detected": "zip"},
            )
        raise FileContentMismatchError(
            f"{filename!r} does not start with a PDF header, so it is not a "
            "readable PDF.",
            details={"filename": filename, "detected": "unknown"},
        )


def _validate_docx_package(path: Path, filename: str) -> None:
    if not zipfile.is_zipfile(path):
        with path.open("rb") as handle:
            head = handle.read(8)
        detected = "ole_compound_file" if head.startswith(_OLE_HEADER) else "unknown"
        hint = (
            "It looks like a legacy .doc binary. Re-save it as .docx and upload again."
            if detected == "ole_compound_file"
            else "The file is not a ZIP package."
        )
        raise FileContentMismatchError(
            f"{filename!r} has a .docx extension but is not a DOCX package. {hint}",
            details={"filename": filename, "detected": detected},
        )

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise FileContentMismatchError(
            f"{filename!r} is a corrupt ZIP package and cannot be read.",
            details={"filename": filename, "detected": "corrupt_zip"},
        ) from exc

    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise FileContentMismatchError(
            f"{filename!r} is a ZIP archive but not a Word document (missing "
            "word/document.xml). Upload a real .docx file.",
            details={"filename": filename, "detected": "zip_not_docx"},
        )


def validate_content(path: Path, file_kind: str, filename: str) -> None:
    """Confirm the bytes on disk really are the declared document type."""
    if file_kind == "pdf":
        _validate_pdf_header(path, filename)
    elif file_kind == "docx":
        _validate_docx_package(path, filename)
    else:  # pragma: no cover - guarded by kind_from_extension
        raise UnsupportedFileTypeError(
            f"Unsupported file kind {file_kind!r}.", details={"file_kind": file_kind}
        )


async def save_upload(
    upload: UploadFile,
    *,
    destination_dir: Path,
    stored_name: str,
    allowed_extensions: set[str],
    max_file_size_bytes: int,
) -> tuple[Path, str, str, int]:
    """Stream one upload to disk.

    Returns ``(stored_path, safe_display_name, file_kind, size_bytes)``.
    """
    display_name = sanitize_filename(upload.filename)
    file_kind = kind_from_extension(display_name, allowed_extensions)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / stored_name

    written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_file_size_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise FileTooLargeError(
                        f"{display_name!r} exceeds the {max_file_size_bytes // (1024 * 1024)} MB "
                        f"per-file limit. Split the document or raise "
                        "MAX_FILE_SIZE_BYTES.",
                        details={
                            "filename": display_name,
                            "limit_bytes": max_file_size_bytes,
                            "received_bytes": written,
                        },
                    )
                handle.write(chunk)
    except FileTooLargeError:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise FileContentMismatchError(
            f"{display_name!r} could not be read from the upload stream.",
            details={"filename": display_name, "error_type": type(exc).__name__},
        ) from exc

    if written == 0:
        destination.unlink(missing_ok=True)
        raise FileContentMismatchError(
            f"{display_name!r} is empty (0 bytes).", details={"filename": display_name}
        )

    try:
        validate_content(destination, file_kind, display_name)
    except Exception:
        # A rejected upload leaves nothing behind: the bytes on disk are not the
        # document type the client claimed, so there is no reason to keep them.
        destination.unlink(missing_ok=True)
        raise

    logger.info(
        "upload_saved",
        extra={"file_kind": file_kind, "size_bytes": written},
    )
    return destination, display_name, file_kind, written
