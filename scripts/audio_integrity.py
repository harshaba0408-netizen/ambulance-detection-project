"""Audit the local audio dataset without modifying it.

The audit reports only observable facts: file hashes, durations, formats,
embedded encoder tags, duplicate content, and numbering gaps. It does not claim
that a format or encoder cohort is a recording session.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\audio_integrity.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\audio_integrity.py
    .\\venv\\Scripts\\python.exe scripts\\audio_integrity.py --output benchmarks\\results\\audio_integrity.json

Quarantine is opt-in and never happens by default. The default output is a
plan listing which duplicate copies would be moved. To execute it:

    .\\venv\\Scripts\\python.exe scripts\\audio_integrity.py --quarantine --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_common  # noqa: E402

AUDIO_ROOT = ROOT / "data" / "audio"
CLASSES = ("siren", "non-siren")
QUARANTINE_ROOT = ROOT / "artifacts" / "quarantine" / "audio"


def wav_info(path: Path) -> dict[str, Any]:
    """Read format data and the ffmpeg ISFT tag when present."""
    import soundfile as sf

    info = sf.info(str(path))
    return {
        "duration_s": round(float(info.duration), 6),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "subtype": str(info.subtype),
        "encoder": _riff_encoder(path),
    }


def _riff_encoder(path: Path) -> Optional[str]:
    """Return the LIST/INFO ISFT value, or None. Malformed chunks stop safely."""
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if header[:4] not in (b"RIFF", b"RF64"):
                return None
            while True:
                chunk = handle.read(8)
                if len(chunk) < 8:
                    return None
                chunk_id, size = struct.unpack("<4sI", chunk)
                if size > path.stat().st_size:
                    return None
                data = handle.read(size)
                if size % 2:
                    handle.read(1)
                if chunk_id == b"LIST" and data.startswith(b"INFOISFT"):
                    return data[12:].rstrip(b"\x00").decode("latin1", "replace")
    except OSError:
        return None


def _number(path: Path) -> Optional[int]:
    stem = path.stem
    if "_" not in stem:
        return None
    try:
        return int(stem.rsplit("_", 1)[1])
    except ValueError:
        return None


def scan(root: Path = AUDIO_ROOT) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_name in CLASSES:
        directory = root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"audio class directory not found: {directory}")
        for path in sorted(directory.glob("*.wav")):
            metadata = wav_info(path)
            records.append(
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "class": class_name,
                    "number": _number(path),
                    "sha256": benchmark_common.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    **metadata,
                }
            )
    return records


def signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["sample_rate"],
        record["channels"],
        record["subtype"],
        record["encoder"],
    )


def duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)

    groups: list[dict[str, Any]] = []
    for digest, members in by_hash.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: (item["class"], item["number"] or 0))
        groups.append(
            {
                "sha256": digest,
                "class": members[0]["class"],
                "numbers": [item["number"] for item in members],
                "files": [item["relative_path"] for item in members],
                "keep": members[0]["relative_path"],
                "duplicates": [item["relative_path"] for item in members[1:]],
            }
        )
    return sorted(groups, key=lambda item: (item["class"], item["numbers"][0] or 0))


def quarantine_plan(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One action per duplicate group: keep the first, move the rest."""
    return [
        {
            "sha256": group["sha256"],
            "keep": group["keep"],
            "quarantine": group["duplicates"],
            "reason": "exact content duplicate",
        }
        for group in duplicate_groups(records)
    ]


def cohorts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contiguous runs sharing a format/encoder signature.

    This is an observable encoding batch, not proof of one recording session.
    """
    result: list[dict[str, Any]] = []
    for class_name in CLASSES:
        ordered = sorted(
            (item for item in records if item["class"] == class_name),
            key=lambda item: item["number"] or 0,
        )
        run: list[dict[str, Any]] = []
        for record in ordered:
            if run and signature(record) != signature(run[-1]):
                result.append(_cohort_summary(class_name, run))
                run = []
            run.append(record)
        if run:
            result.append(_cohort_summary(class_name, run))
    return result


def _cohort_summary(class_name: str, run: list[dict[str, Any]]) -> dict[str, Any]:
    numbers = [item["number"] for item in run if item["number"] is not None]
    first = run[0]
    return {
        "class": class_name,
        "numbers": [min(numbers), max(numbers)] if numbers else None,
        "count": len(run),
        "contiguous": bool(numbers) and numbers == list(range(min(numbers), max(numbers) + 1)),
        "sample_rate": first["sample_rate"],
        "channels": first["channels"],
        "subtype": first["subtype"],
        "encoder": first["encoder"],
    }


def missing_numbers(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for class_name in CLASSES:
        numbers = sorted(
            item["number"] for item in records if item["class"] == class_name and item["number"] is not None
        )
        if not numbers:
            result[class_name] = []
            continue
        present = set(numbers)
        result[class_name] = [n for n in range(numbers[0], numbers[-1] + 1) if n not in present]
    return result


def build_report(records: list[dict[str, Any]], root: Path = AUDIO_ROOT) -> dict[str, Any]:
    groups = duplicate_groups(records)
    plan = quarantine_plan(records)
    class_counts = Counter(item["class"] for item in records)
    formats = Counter(
        f"{item['sample_rate']} Hz / {item['channels']} ch / {item['subtype']} / {item['encoder'] or 'no encoder tag'}"
        for item in records
    )
    numbers = {
        cls: sorted(item["number"] for item in records if item["class"] == cls and item["number"] is not None)
        for cls in CLASSES
    }
    return {
        "benchmark": "audio_integrity",
        "kind": "dataset_audit",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset_root": str(root.resolve()),
        "provenance_status": "unverified",
        "source_metadata_present": False,
        "caveat": (
            "No source, session, recording, licence or author metadata exists in the WAV "
            "files beyond an ffmpeg encoder tag. Format/encoder cohorts are observed "
            "encoding batches, not verified recording sessions."
        ),
        "summary": {
            "files": len(records),
            "classes": dict(class_counts),
            "number_ranges": {
                cls: ([numbers[cls][0], numbers[cls][-1]] if numbers[cls] else None)
                for cls in CLASSES
            },
            "missing_numbers": missing_numbers(records),
            "duplicate_groups": len(groups),
            "duplicate_files": sum(len(group["duplicates"]) for group in groups),
            "encoding_cohorts": len(cohorts(records)),
        },
        "formats": dict(formats),
        "duplicate_groups": groups,
        "encoding_cohorts": cohorts(records),
        "quarantine_plan": plan,
    }


def execute_quarantine(plan: list[dict[str, Any]], root: Path = AUDIO_ROOT) -> list[dict[str, Any]]:
    """Move duplicate copies into artifacts/quarantine/audio, preserving class."""
    moved: list[dict[str, Any]] = []
    for action in plan:
        for relative in action["quarantine"]:
            source = ROOT / relative
            class_name = source.parent.name
            target = QUARANTINE_ROOT / class_name / source.name
            if not source.is_file():
                raise FileNotFoundError(f"duplicate no longer exists: {source}")
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(
                {
                    "from": str(source.relative_to(ROOT)),
                    "to": str(target.relative_to(ROOT)),
                    "kept": action["keep"],
                }
            )
    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
        "moved": moved,
        "note": (
            "Files were moved, not deleted. Restore by moving them back to their "
            "original class directory."
        ),
    }
    benchmark_common.write_json(manifest, QUARANTINE_ROOT / "manifest.json")
    return moved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(AUDIO_ROOT), help="audio dataset root")
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="move duplicate copies into artifacts/quarantine/audio (requires --yes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm the destructive move requested by --quarantine",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the audit report without writing a result file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quarantine and not args.yes:
        raise SystemExit("--quarantine requires --yes; no files were moved")

    records = scan(Path(args.root))
    report = build_report(records, Path(args.root))

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    output = (
        Path(args.output)
        if args.output
        else ROOT / "benchmarks" / "results" / f"audio_integrity_{benchmark_common.utc_timestamp()}.json"
    )
    report["output"] = str(output)
    benchmark_common.write_json(report, output)
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {output}")

    if args.quarantine:
        moved = execute_quarantine(report["quarantine_plan"], Path(args.root))
        print(json.dumps({"moved": moved}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
