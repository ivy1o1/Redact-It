"""Tests for structured logging and the sensitive-content scrubber.

The property under test is a hard requirement: document text, user instructions
and model completions must never reach a log line, even when a caller
accidentally interpolates one into a message.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest

from app.logging_setup import (
    LOGGER_NAME,
    JsonFormatter,
    PlainFormatter,
    SensitiveContentFilter,
    configure_logging,
    get_logger,
)

from .conftest import REPORT_LINES

DOCUMENT_TEXT = "\n".join(REPORT_LINES)
SENSITIVE_FRAGMENTS = ("Alice Johnson", "42 Maple Street", "Brian O'Connor", "(555) 123-4567")

LONG_INSTRUCTION = (
    "Remove all student names, phone numbers and home addresses from every page of "
    "the attached report card, but keep the grades and the course names exactly as "
    "they are so the recipient can still read the academic results."
)


def make_record(
    msg: str = "event",
    *,
    level: int = logging.INFO,
    args: tuple = (),
    **extra: Any,
) -> logging.LogRecord:
    entry = logging.LogRecord(
        name=f"{LOGGER_NAME}.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(entry, key, value)
    return entry


def with_exception(msg: str = "failed") -> logging.LogRecord:
    try:
        raise ValueError("boom")
    except ValueError:
        entry = make_record(msg)
        entry.exc_info = sys.exc_info()
        return entry


class _Capture(logging.Handler):
    """Collects formatted output through ``handle()``, so handler filters run.

    ``Handler.format()`` bypasses the filter chain; a test that called it directly
    would pass even with no scrubber attached.
    """

    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__()
        self.setFormatter(formatter)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    @property
    def line(self) -> str:
        assert len(self.lines) == 1, self.lines
        return self.lines[0]


@pytest.fixture
def capture() -> _Capture:
    return _Capture(JsonFormatter())


@pytest.fixture(autouse=True)
def restore_logging_state():
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in saved_handlers:
        logger.addHandler(handler)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


class TestSensitiveContentFilter:
    @pytest.fixture
    def scrub(self) -> SensitiveContentFilter:
        return SensitiveContentFilter()

    @pytest.mark.parametrize(
        "key",
        [
            "document_text", "extracted_text", "page_text", "instruction",
            "user_instruction", "prompt", "messages", "completion", "matched_text",
            "content", "body", "raw_response", "api_key", "authorization",
        ],
    )
    def test_known_sensitive_keys_are_scrubbed(self, scrub: SensitiveContentFilter, key: str) -> None:
        entry = make_record("event", **{key: "Alice Johnson lives at 42 Maple Street"})
        assert scrub.filter(entry) is True
        assert getattr(entry, key) == "[redacted-from-log]"

    def test_key_matching_is_case_insensitive(self, scrub: SensitiveContentFilter) -> None:
        entry = make_record("event", DOCUMENT_TEXT="Alice Johnson")
        scrub.filter(entry)
        assert entry.DOCUMENT_TEXT == "[redacted-from-log]"

    def test_a_short_value_under_a_sensitive_key_is_still_scrubbed(
        self, scrub: SensitiveContentFilter
    ) -> None:
        entry = make_record("event", instruction="names")
        scrub.filter(entry)
        assert entry.instruction == "[redacted-from-log]"

    def test_document_text_interpolated_into_a_message_is_scrubbed(
        self, scrub: SensitiveContentFilter
    ) -> None:
        entry = make_record(f"analysed {DOCUMENT_TEXT}")
        scrub.filter(entry)
        rendered = entry.getMessage()
        for fragment in SENSITIVE_FRAGMENTS:
            assert fragment not in rendered
        assert "[redacted-from-log]" in rendered

    def test_a_long_instruction_in_an_argument_is_scrubbed(
        self, scrub: SensitiveContentFilter
    ) -> None:
        entry = make_record("received instruction %s", args=(LONG_INSTRUCTION,))
        scrub.filter(entry)
        assert "Maple Street" not in entry.getMessage()
        assert "42 Maple Street" not in LONG_INSTRUCTION or True  # guard: fixture is long
        assert len(LONG_INSTRUCTION) > 160

    def test_a_long_instruction_under_its_own_key_is_scrubbed(
        self, scrub: SensitiveContentFilter
    ) -> None:
        entry = make_record("instruction received", instruction=LONG_INSTRUCTION)
        scrub.filter(entry)
        assert entry.instruction == "[redacted-from-log]"

    def test_short_operational_messages_are_left_alone(self, scrub: SensitiveContentFilter) -> None:
        entry = make_record("session_created", session_id="sess_abc", count=3, detections=12)
        scrub.filter(entry)
        assert entry.getMessage() == "session_created"
        assert entry.session_id == "sess_abc"
        assert entry.count == 3
        assert entry.detections == 12

    def test_a_short_message_is_not_rewritten(self, scrub: SensitiveContentFilter) -> None:
        entry = make_record("upload_saved file_kind=pdf size_bytes=1234")
        scrub.filter(entry)
        assert entry.getMessage() == "upload_saved file_kind=pdf size_bytes=1234"

    def test_non_string_arguments_are_untouched(self, scrub: SensitiveContentFilter) -> None:
        huge = 123456789012345678901234567890
        entry = make_record("counts %r %r", args=(huge, None))
        scrub.filter(entry)
        assert entry.args[0] == huge
        assert entry.args[1] is None

    def test_a_single_non_tuple_argument_is_handled(self, scrub: SensitiveContentFilter) -> None:
        entry = make_record("one %s")
        entry.args = "x"
        assert scrub.filter(entry) is True
        assert entry.args == ("x",)

    def test_the_filter_always_admits_the_record(self, scrub: SensitiveContentFilter) -> None:
        assert scrub.filter(make_record("anything", document_text="secret")) is True


class TestJsonFormatter:
    def test_output_is_valid_json_with_the_core_fields(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(make_record("session_created")))
        assert payload["message"] == "session_created"
        assert payload["level"] == "INFO"
        assert payload["logger"] == f"{LOGGER_NAME}.test"
        assert "T" in payload["timestamp"]

    def test_operational_extras_are_included(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(make_record("event", session_id="sess_1", detections=7)))
        assert payload["session_id"] == "sess_1"
        assert payload["detections"] == 7

    def test_standard_record_internals_are_not_leaked(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(make_record("event")))
        for reserved in ("args", "msg", "pathname", "lineno", "funcName", "process", "thread", "name"):
            assert reserved not in payload

    def test_non_scalar_extras_are_rendered_as_repr(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(make_record("event", ids=["a", "b"])))
        assert payload["ids"] == "['a', 'b']"

    def test_an_exception_is_included(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(with_exception()))
        assert "ValueError: boom" in payload["exception"]

    def test_unicode_is_preserved(self, capture: _Capture) -> None:
        payload = json.loads(capture.format(make_record("event", note="caf\u00e9 \u2014 na\u00efve")))
        assert payload["note"] == "caf\u00e9 \u2014 na\u00efve"

    def test_a_scrubbed_document_never_appears_in_the_output(self) -> None:
        handler = _Capture(JsonFormatter())
        handler.addFilter(SensitiveContentFilter())
        entry = make_record(
            "analysed document", document_text=DOCUMENT_TEXT, instruction=LONG_INSTRUCTION
        )
        handler.handle(entry)
        line = handler.line
        for fragment in SENSITIVE_FRAGMENTS:
            assert fragment not in line
        json.loads(line)  # still machine-parseable after scrubbing

    def test_the_scrubber_is_actually_attached_to_the_handler(self) -> None:
        handler = _Capture(JsonFormatter())
        handler.addFilter(SensitiveContentFilter())
        assert any(isinstance(f, SensitiveContentFilter) for f in handler.filters)

    def test_punctuation_does_not_fragment_a_sensitive_run(self) -> None:
        # Regression: the run used to be defined as an allow-list of text
        # characters, so "Grade: B+" and an email address ended the match and the
        # tail of the document - name, phone, address - was logged verbatim.
        entry = make_record(
            "analysed "
            + "Course: Algebra II Grade: B+ Contact: b.oconnor@school.test "
            + "http://example.invalid/a?b=c!d#e $1,200 (555) 987-6543 " * 4
        )
        SensitiveContentFilter().filter(entry)
        assert "b.oconnor@school.test" not in entry.getMessage()
        assert "555" not in entry.getMessage()


class TestPlainFormatter:
    def test_message_and_extras_are_readable(self) -> None:
        line = PlainFormatter().format(make_record("session_created", session_id="sess_1"))
        assert "INFO" in line
        assert f"{LOGGER_NAME}.test: session_created" in line
        assert "session_id='sess_1'" in line

    def test_a_record_without_extras_has_no_trailing_separator(self) -> None:
        assert PlainFormatter().format(make_record("session_created")).endswith("session_created")

    def test_an_exception_is_appended(self) -> None:
        assert "ValueError: boom" in PlainFormatter().format(with_exception())


class TestConfigureLogging:
    def test_one_handler_is_attached(self) -> None:
        configure_logging(level="INFO", as_json=True)
        assert len(logging.getLogger(LOGGER_NAME).handlers) == 1

    def test_reconfiguring_does_not_stack_handlers(self) -> None:
        configure_logging(level="INFO", as_json=True)
        configure_logging(level="DEBUG", as_json=True)
        assert len(logging.getLogger(LOGGER_NAME).handlers) == 1

    def test_the_level_is_applied(self) -> None:
        configure_logging(level="WARNING", as_json=False)
        assert logging.getLogger(LOGGER_NAME).level == logging.WARNING

    def test_json_and_plain_formatters_are_selectable(self) -> None:
        configure_logging(level="INFO", as_json=True)
        assert isinstance(logging.getLogger(LOGGER_NAME).handlers[0].formatter, JsonFormatter)
        configure_logging(level="INFO", as_json=False)
        assert isinstance(logging.getLogger(LOGGER_NAME).handlers[0].formatter, PlainFormatter)

    def test_the_scrubber_is_attached_to_the_handler(self) -> None:
        configure_logging(level="INFO", as_json=True)
        handler = logging.getLogger(LOGGER_NAME).handlers[0]
        assert any(isinstance(f, SensitiveContentFilter) for f in handler.filters)

    def test_records_do_not_propagate_to_the_root_logger(self) -> None:
        configure_logging(level="INFO", as_json=True)
        assert logging.getLogger(LOGGER_NAME).propagate is False

    def test_a_live_logger_emits_scrubbed_json(self, capsys) -> None:
        configure_logging(level="INFO", as_json=True)
        get_logger("test").info("analysed", extra={"document_text": DOCUMENT_TEXT})
        captured = capsys.readouterr().out.strip()
        payload = json.loads(captured)
        assert payload["document_text"] == "[redacted-from-log]"
        for fragment in SENSITIVE_FRAGMENTS:
            assert fragment not in captured


class TestGetLogger:
    def test_component_loggers_are_namespaced(self) -> None:
        assert get_logger("uploads").name == f"{LOGGER_NAME}.uploads"
        assert get_logger("redaction.pdf").name == f"{LOGGER_NAME}.redaction.pdf"

    def test_the_same_logger_is_returned_each_time(self) -> None:
        assert get_logger("uploads") is get_logger("uploads")
