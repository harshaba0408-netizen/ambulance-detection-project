"""Assign acoustic content groups to the local audio dataset, read-only.

The script does the following and nothing else:

1. scans `data/audio/{siren,non-siren}` for WAV metadata and SHA-256 hashes;
2. drops the non-canonical copies of exact byte duplicates, matching the splitter;
3. groups the retained files by acoustic content:
   * non-siren: mono zero-lag Pearson `r >= 0.99`;
   * siren: exact SHA-256, mono zero-lag `r >= 0.995`, or best-lag
     `r >= 0.95` with at least 1.5 s of overlap;
   * the six files that match the external Freesound 87130 reference are
     explicitly unioned (13-15 are probable, 85-87 are verified);
4. writes `data/audio/content_groups.csv` with one row per retained file.

A content group is acoustic evidence, not a verified recording session, source
or licence. The CSV is a split manifest for `make_audio_split.py`, not a
provenance manifest. The audio files are never modified.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\audio_content_groups.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\audio_content_groups.py
    .\\venv\\Scripts\\python.exe scripts\\audio_content_groups.py --report benchmarks\\results\\audio_content_groups.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audio_integrity  # noqa: E402
import benchmark_common  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data" / "audio" / "content_groups.csv"
ANALYSIS_RATE = 22050
CLIP_SECONDS = 3.0
ANALYSIS_SAMPLES = int(round(ANALYSIS_RATE * CLIP_SECONDS))

NON_SIREN_ZERO_R = 0.99
SIREN_ZERO_R = 0.995
SIREN_BEST_R = 0.95
SIREN_MIN_OVERLAP_S = 1.5
SIREN_MIN_OVERLAP = int(round(SIREN_MIN_OVERLAP_S * ANALYSIS_RATE))
# Approximate FFT correlation is only a prefilter for the exact Pearson check.
# The box-filtered coarse search can lose a little correlation when the true
# shift is not a multiple of the decimation factor, so keep this threshold low.
SIREN_APPROX_R = 0.50

FIELDS = (
    "file",
    "group_id",
    "group_method",
    "max_r",
    "source_id",
    "source_confidence",
    "license",
    "notes",
)

# The Freesound 87130 reference is 60.316 s, recorded in Seville on 2009-12-29 by
# dobroide and released under CC BY 4.0. These files match that reference; only
# the first three are verified. See docs/AUDIO_DATASET_PROVENANCE.md.
REFERENCE_SOURCE = "freesound:87130"
REFERENCE_LICENSE = "CC-BY-4.0"
REFERENCE_VERIFIED = (85, 86, 87)
REFERENCE_PROBABLE = (13, 14, 15)
REFERENCE_NUMBERS = REFERENCE_VERIFIED + REFERENCE_PROBABLE
REFERENCE_NOTE = (
    "matches dobroide Freesound 87130 reference; acoustic evidence only, "
    "not a verified recording session"
)


class UnionFind:
    """Deterministic union-find; the smallest root wins."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for item in range(len(self.parent)):
            result[self.find(item)].append(item)
        return dict(result)


def _resample(mono: Any, sample_rate: int) -> Any:
    import numpy as np
    from scipy.signal import resample_poly

    if sample_rate != ANALYSIS_RATE:
        divisor = math.gcd(sample_rate, ANALYSIS_RATE)
        mono = resample_poly(mono, ANALYSIS_RATE // divisor, sample_rate // divisor)
    if len(mono) < ANALYSIS_SAMPLES:
        mono = np.pad(mono, (0, ANALYSIS_SAMPLES - len(mono)))
    return np.asarray(mono[:ANALYSIS_SAMPLES], dtype="float64")


def load_signal(path: Path) -> Any:
    """Load a WAV as a mono, common-rate 3 s analysis signal."""
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    return _resample(data.mean(axis=1), int(sample_rate))


def exact_r(left: Any, right: Any, lag: int) -> float:
    """Pearson r with `right` shifted by `lag` samples; only overlap is used."""
    import numpy as np

    size = len(left)
    if lag >= 0:
        a = left[lag:]
        b = right[: size - lag]
    else:
        a = left[: size + lag]
        b = right[-lag:]
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


_LAG_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def _lag_axis(size: int, min_overlap: int) -> tuple[Any, Any]:
    """Cached lag values and the overlap-valid mask for one signal length."""
    import numpy as np

    key = (size, min_overlap)
    if key not in _LAG_CACHE:
        lags = np.arange(-(size - 1), size)
        valid = np.abs(lags) <= size - min_overlap
        _LAG_CACHE[key] = (lags, valid)
    return _LAG_CACHE[key]


def _pool(signal: Any, factor: int) -> Any:
    """Box-filter and decimate a signal for the approximate lag search."""
    usable = len(signal) // factor * factor
    return signal[:usable].reshape(-1, factor).mean(axis=1)


def _best_lag_coarse(
    left: Any, right: Any, *, min_overlap: int, factor: int
) -> tuple[float, int, float]:
    """Decimated search; the exact native-rate check happens in the caller."""
    import numpy as np
    from scipy.signal import fftconvolve

    a = _pool(left, factor)
    b = _pool(right, factor)
    size = len(a)
    a = a - a.mean()
    b = b - b.mean()
    correlation = fftconvolve(a, b[::-1], mode="full")
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator == 0.0:
        return 0.0, 0, 0.0
    correlation /= denominator
    lags, valid = _lag_axis(size, max(1, min_overlap // factor))
    masked = np.where(valid, correlation, -np.inf)
    approximate = float(masked.max())
    if approximate < SIREN_APPROX_R:
        return approximate, 0, approximate
    return approximate, int(lags[int(np.argmax(masked))]), approximate


def best_lag_r(
    left: Any, right: Any, *, min_overlap: int = SIREN_MIN_OVERLAP
) -> tuple[float, int, float]:
    """Return (exact r, lag, approximate r) for the best valid-lag overlap."""
    import numpy as np
    from scipy.signal import fftconvolve

    size = len(left)
    factor = 4
    if size > min_overlap + factor:
        approximate, coarse_lag, _ = _best_lag_coarse(
            left, right, min_overlap=min_overlap, factor=factor
        )
        if approximate < SIREN_APPROX_R:
            return approximate, 0, approximate
        candidates = [coarse_lag * factor + offset for offset in range(-(factor + 2), factor + 3)]
    else:
        candidates = []

    if not candidates:
        a = left - left.mean()
        b = right - right.mean()
        correlation = fftconvolve(a, b[::-1], mode="full")
        denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        if denominator == 0.0:
            return 0.0, 0, 0.0
        correlation /= denominator
        lags, valid = _lag_axis(size, min_overlap)
        masked = np.where(valid, correlation, -np.inf)
        approximate = float(masked.max())
        if approximate < SIREN_APPROX_R:
            return approximate, 0, approximate
        peak = int(np.argmax(masked))
        candidates = [int(lags[index]) for index in range(max(0, peak - 2), min(len(lags), peak + 3))]

    best_r, best_lag = -1.0, 0
    for lag in candidates:
        if abs(lag) > size - min_overlap:
            continue
        value = exact_r(left, right, lag)
        if value > best_r:
            best_r, best_lag = value, lag
    return best_r, best_lag, approximate


def zero_lag_matrix(signals: list[Any]) -> Any:
    import numpy as np

    matrix = np.stack(signals)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (matrix @ matrix.T) / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def group_non_siren(members: list[dict[str, Any]], signals: list[Any]):
    """Return (union-find, accepted-pair r matrix, method) for non-siren files."""
    import numpy as np

    matrix = zero_lag_matrix(signals)
    evidence = np.zeros_like(matrix)
    union = UnionFind(len(members))
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            value = float(matrix[left, right])
            if value >= NON_SIREN_ZERO_R:
                union.union(left, right)
                evidence[left, right] = evidence[right, left] = value
    return union, evidence, f"non-siren zero-lag r>={NON_SIREN_ZERO_R}"


def group_siren(members: list[dict[str, Any]], signals: list[Any]):
    """Return (union-find, accepted-pair r matrix, method counts) for siren files."""
    import numpy as np

    matrix = zero_lag_matrix(signals)
    evidence = np.zeros_like(matrix)
    union = UnionFind(len(members))
    methods: dict[tuple[int, int], str] = {}
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            zero = float(matrix[left, right])
            if zero >= SIREN_ZERO_R:
                union.union(left, right)
                evidence[left, right] = evidence[right, left] = zero
                methods[(left, right)] = f"zero-lag r>={SIREN_ZERO_R}"
                continue
            best, _lag, _approximate = best_lag_r(signals[left], signals[right])
            if best >= SIREN_BEST_R:
                union.union(left, right)
                evidence[left, right] = evidence[right, left] = best
                methods[(left, right)] = (
                    f"best-lag r>={SIREN_BEST_R} with overlap>={SIREN_MIN_OVERLAP_S}s"
                )

    by_number = {member["number"]: index for index, member in enumerate(members)}
    if all(number in by_number for number in REFERENCE_NUMBERS):
        anchor = by_number[REFERENCE_NUMBERS[0]]
        for number in REFERENCE_NUMBERS[1:]:
            union.union(anchor, by_number[number])
    return union, evidence, methods


def _group_id(class_name: str, index: int) -> str:
    prefix = "non_siren" if class_name == "non-siren" else "siren"
    return f"{prefix}_g{index:03d}"


def _reference_row(number: Optional[int]) -> dict[str, str]:
    if number in REFERENCE_VERIFIED:
        confidence = "verified"
    elif number in REFERENCE_PROBABLE:
        confidence = "probable"
    else:
        return {
            "source_id": "",
            "source_confidence": "unverified",
            "license": "",
            "notes": "acoustic content group; no verified source or session",
        }
    return {
        "source_id": REFERENCE_SOURCE,
        "source_confidence": confidence,
        "license": REFERENCE_LICENSE,
        "notes": REFERENCE_NOTE,
    }


def build_rows(
    records: list[dict[str, Any]],
    signals: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group retained records and return (CSV rows, summary).

    `signals` is injectable for tests. When omitted, each retained WAV is loaded
    at the common analysis rate.
    """
    duplicate_groups = audio_integrity.duplicate_groups(records)
    dropped = {path for group in duplicate_groups for path in group["duplicates"]}
    retained = [record for record in records if record["relative_path"] not in dropped]

    if signals is None:
        signals = {
            record["relative_path"]: load_signal(ROOT / record["relative_path"])
            for record in retained
        }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "retained_files": len(retained),
        "excluded_duplicates": len(dropped),
        "classes": {},
    }

    for class_name in audio_integrity.CLASSES:
        members = sorted(
            (record for record in retained if record["class"] == class_name),
            key=lambda record: (record["number"] or 0, record["relative_path"]),
        )
        if not members:
            continue
        class_signals = [signals[member["relative_path"]] for member in members]
        if class_name == "siren":
            union, evidence, _methods = group_siren(members, class_signals)
            method_label = f"siren acoustic-content grouping (zero-lag r>={SIREN_ZERO_R} or best-lag r>={SIREN_BEST_R} with overlap>={SIREN_MIN_OVERLAP_S}s)"
        else:
            union, evidence, _methods = group_non_siren(members, class_signals)
            method_label = f"non-siren zero-lag r>={NON_SIREN_ZERO_R}"

        groups = union.groups()
        ordered = sorted(groups.values(), key=lambda indexes: min(indexes))
        index_to_number = {index: members[index]["number"] for index in range(len(members))}
        class_summary = {
            "files": len(members),
            "groups": len(ordered),
            "group_sizes": sorted((len(group) for group in ordered), reverse=True),
            "method": method_label,
        }
        summary["classes"][class_name] = class_summary

        for group_number, indexes in enumerate(ordered, start=1):
            group_id = _group_id(class_name, group_number)
            if len(indexes) == 1:
                group_method = "singleton"
            elif class_name == "siren" and any(
                index_to_number[index] in REFERENCE_NUMBERS for index in indexes
            ):
                group_method = "siren reference union (Freesound 87130)"
            elif class_name == "siren":
                group_method = f"siren acoustic content (zero-lag r>={SIREN_ZERO_R} or best-lag r>={SIREN_BEST_R})"
            else:
                group_method = f"non-siren zero-lag r>={NON_SIREN_ZERO_R}"

            for index in indexes:
                member = members[index]
                siblings = [other for other in indexes if other != index]
                best = max((float(evidence[index, other]) for other in siblings), default=0.0)
                max_r = f"{best:.4f}" if best > 0.0 else ""
                row = {
                    "file": member["relative_path"].replace("\\", "/"),
                    "group_id": group_id,
                    "group_method": group_method,
                    "max_r": max_r,
                }
                metadata = _reference_row(member["number"])
                if not max_r and metadata["source_id"]:
                    metadata["notes"] += (
                        "; grouped via the shared external reference because no "
                        f">={SIREN_MIN_OVERLAP_S}s pairwise overlap met the threshold"
                    )
                row.update(metadata)
                rows.append(row)

    rows.sort(key=lambda row: (row["file"].split("/")[-2], int(Path(row["file"]).stem.split("_")[1])))
    summary["groups_per_class"] = {
        class_name: class_summary["groups"] for class_name, class_summary in summary["classes"].items()
    }
    return rows, summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(audio_integrity.AUDIO_ROOT), help="audio dataset root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="content-group CSV path")
    parser.add_argument("--report", default=None, help="optional JSON summary path")
    parser.add_argument("--dry-run", action="store_true", help="print the summary without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = audio_integrity.scan(Path(args.root))
    rows, summary = build_rows(records)
    payload = {
        "benchmark": "audio_content_groups",
        "kind": "acoustic_content_grouping",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset_root": str(Path(args.root).resolve()),
        "analysis_rate": ANALYSIS_RATE,
        "clip_seconds": CLIP_SECONDS,
        "thresholds": {
            "non_siren_zero_r": NON_SIREN_ZERO_R,
            "siren_zero_r": SIREN_ZERO_R,
            "siren_best_r": SIREN_BEST_R,
            "siren_min_overlap_s": SIREN_MIN_OVERLAP_S,
        },
        "summary": summary,
        "caveat": (
            "These are acoustic content groups, not verified recording sessions, "
            "sources or licences. The split they produce is content-group held-out, "
            "not source-independent."
        ),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    output = Path(args.output)
    write_csv(rows, output)
    payload["output"] = str(output)
    if args.report:
        benchmark_common.write_json(payload, Path(args.report))
    print(json.dumps(summary, indent=2))
    print(f"written: {output}")
    if args.report:
        print(f"report: {Path(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
