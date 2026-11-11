"""Shared helpers for the Stage 10 benchmark scripts.

These functions only format measurements; they never fabricate a value. A
missing or undefined ratio is reported as `None`, not as zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0.0, 1.0]")
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def latency_summary_ms(values_s: Iterable[float]) -> dict[str, Optional[float]]:
    values = [float(value) * 1000.0 for value in values_s]
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 4),
        "median_ms": round(median, 4),
        "p95_ms": round(percentile(ordered, 0.95), 4),
        "max_ms": round(ordered[-1], 4),
    }


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["ram_total_gb"] = round(memory.total / 1024**3, 2)
        info["ram_available_gb"] = round(memory.available / 1024**3, 2)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = None
    return info


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
