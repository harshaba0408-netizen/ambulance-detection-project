from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ambulance_system.logging_utils import (
    LOGGER_NAME,
    configure_logging,
    get_logger,
)


@pytest.fixture()
def quiet_logger():
    yield
    configure_logging("WARNING", console=False)


def test_configure_logging_is_idempotent(tmp_path):
    logger = configure_logging("INFO", json_file=tmp_path / "first.jsonl", console=False)
    configure_logging("INFO", json_file=tmp_path / "second.jsonl", console=False)

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.FileHandler)


def test_invalid_level_is_rejected():
    with pytest.raises(ValueError):
        configure_logging("NOT-A-LEVEL", console=False)


def test_json_log_contains_structured_fields(tmp_path, quiet_logger):
    log_path = tmp_path / "events.jsonl"
    configure_logging("INFO", json_file=log_path, console=False)
    logger = get_logger("test")

    logger.info(
        "ambulance confirmed",
        extra={
            "event": {"event_id": "abc123", "confidence": 0.87},
            "approach": "Road 1",
        },
    )

    # Reconfigure to flush and close the file handler before reading.
    configure_logging("INFO", json_file=None, console=False)

    payload = json.loads(Path(log_path).read_text(encoding="utf-8").strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == f"{LOGGER_NAME}.test"
    assert payload["message"] == "ambulance confirmed"
    assert payload["approach"] == "Road 1"
    assert payload["event"]["event_id"] == "abc123"
    assert payload["event"]["confidence"] == pytest.approx(0.87)
    assert payload["timestamp"].endswith("+00:00")


def test_json_log_redacts_sensitive_keys(tmp_path, quiet_logger):
    log_path = tmp_path / "redacted.jsonl"
    configure_logging("INFO", json_file=log_path, console=False)
    logger = get_logger("security")

    logger.warning(
        "camera connection",
        extra={
            "event": {
                "rtsp_url": "rtsp://camera.local/stream",
                "password": "hunter2",
                "nested": {"api_key": "secret-value", "safe": "visible"},
            }
        },
    )
    configure_logging("INFO", json_file=None, console=False)

    payload = json.loads(Path(log_path).read_text(encoding="utf-8").strip())
    assert payload["event"]["password"] == "***"
    assert payload["event"]["nested"]["api_key"] == "***"
    assert payload["event"]["nested"]["safe"] == "visible"
    assert "hunter2" not in Path(log_path).read_text(encoding="utf-8")


def test_text_log_is_written(tmp_path, quiet_logger):
    log_path = tmp_path / "app.log"
    configure_logging("INFO", text_file=log_path, console=False)
    get_logger("text").info("hello")
    configure_logging("INFO", console=False)

    contents = Path(log_path).read_text(encoding="utf-8")
    assert "hello" in contents
    assert "INFO" in contents


def test_get_logger_is_namespaced():
    assert get_logger().name == LOGGER_NAME
    assert get_logger("vision").name == f"{LOGGER_NAME}.vision"
    assert get_logger(f"{LOGGER_NAME}.vision").name == f"{LOGGER_NAME}.vision"
