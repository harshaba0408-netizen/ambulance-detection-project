"""Structured logging with secret redaction.

The existing prototype writes free-text lines to a Tk widget. That is fine for
a demo but impossible to parse. These helpers add a JSON-lines formatter and a
single place to configure handlers, while leaving the legacy UI untouched.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

LOGGER_NAME = "ambulance_system"

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
)


def _redact(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, key) for item in value]
    if key is not None and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "***"
    return value


class TextFormatter(logging.Formatter):
    """Human-readable one-line format for console and text files."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S%z")


class JsonlFormatter(logging.Formatter):
    """One JSON object per line, with redacted structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in ("event", "event_id", "approach", "source", "confidence"):
            if hasattr(record, field_name):
                payload[field_name] = _redact(getattr(record, field_name), field_name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str | int = "INFO",
    *,
    text_file: Optional[str | Path] = None,
    json_file: Optional[str | Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the package logger and return it.

    Calling this repeatedly is safe: previous handlers are removed and closed
    first, so tests do not accumulate duplicate output.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(TextFormatter())
        logger.addHandler(stream)

    if text_file is not None:
        text_path = Path(text_file)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_handler = logging.FileHandler(text_path, encoding="utf-8")
        text_handler.setFormatter(TextFormatter())
        logger.addHandler(text_handler)

    if json_file is not None:
        json_path = Path(json_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(json_path, encoding="utf-8")
        json_handler.setFormatter(JsonlFormatter())
        logger.addHandler(json_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the package logger without configuring handlers."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
