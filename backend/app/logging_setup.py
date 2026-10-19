"""Structured logging.

Two guarantees matter here:
  * output is machine-parseable (JSON) in non-debug environments;
  * document text, user instructions and model completions are never logged.

``SensitiveContentFilter`` is a defence-in-depth backstop: even if a caller
accidentally interpolates one of those values into a log message, the record is
scrubbed before it reaches a handler.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "redactit"

_SENSITIVE_KEYS = frozenset(
    {
        "document_text",
        "extracted_text",
        "page_text",
        "instruction",
        "user_instruction",
        "prompt",
        "messages",
        "completion",
        "matched_text",
        "content",
        "body",
        "raw_response",
        "api_key",
        "authorization",
    }
)

_SCRUB = "[redacted-from-log]"

# Anything that looks like a long run of natural-language text is suspicious in a
# log line. Document/instruction content is the realistic source.
#
# Deliberately defined by exclusion (any character that is not a control code,
# plus tab/newline/CR) rather than by an allow-list of "text-like" punctuation.
# An allow-list fragments the run: a report body containing "Grade: B+" or an
# email address stops matching at '+' and '@', so the tail of the document -
# emergency-contact name, phone number, address - survives and reaches the log.
_LONG_TEXT = re.compile(r"[^\x00-\x08\x0e-\x1f\x7f]{160,}")


class SensitiveContentFilter(logging.Filter):
    """Scrub known-sensitive keys and over-long free text from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(getattr(record, "__dict__", {})):
            if key.lower() in _SENSITIVE_KEYS:
                setattr(record, key, _SCRUB)

        if isinstance(record.msg, str) and record.msg:
            record.msg = _LONG_TEXT.sub(_SCRUB, record.msg)
        if record.args:
            record.args = tuple(
                _LONG_TEXT.sub(_SCRUB, arg) if isinstance(arg, str) else arg
                for arg in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        return True


class JsonFormatter(logging.Formatter):
    _RESERVED = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "taskName", "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            else:
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<8} {record.name}: {record.getMessage()}"
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in JsonFormatter._RESERVED and not k.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
            base = f"{base} | {rendered}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(*, level: str = "INFO", as_json: bool = True) -> None:
    """Attach a single stream handler to the ``redactit`` logger."""
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level.upper())
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else PlainFormatter())
    handler.addFilter(SensitiveContentFilter())
    root.addHandler(handler)


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
