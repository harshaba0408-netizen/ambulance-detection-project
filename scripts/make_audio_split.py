"""Build a deterministic, leakage-aware audio train/validation/test split.

Three modes:

1. **Source-independent mode** requires `--manifest`, a JSON or CSV file that
   maps every retained WAV to a `session_id` or `source_id`. Groups are assigned
   whole to one split, so no known session appears in two splits.
2. **Content-group mode** requires a manifest whose rows carry a `group_id`.
   Groups are acoustic content groups (for example, files that share a recording
   or a near-duplicate waveform). No group spans two splits, but a content group
   is **not** a verified session or source, so the result is not
   source-independent.
3. **Provisional mode** (default) has no metadata. It removes exact content
   duplicates and assigns each remaining file by a salted hash. This is
   held-out at the file level, but it is **not** source-independent.

`group_id` takes precedence over `session_id`, which takes precedence over
`source_id`. `source_independent` is true only when every retained row has a
verified `session_id`/`source_id` and no row uses `group_id`.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\make_audio_split.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\make_audio_split.py
    .\\venv\\Scripts\\python.exe scripts\\make_audio_split.py --manifest data\\audio\\provenance.csv
    .\\venv\\Scripts\\python.exe scripts\\make_audio_split.py --manifest data\\audio\\content_groups.csv --seed stage12-20261111

Manifest columns: `file`, `group_id`, `session_id`, `source_id`, `license`, `notes`.
`file` may be absolute or relative to the repository root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audio_integrity  # noqa: E402
import benchmark_common  # noqa: E402

CLASSES = ("siren", "non-siren")
DEFAULT_OUTPUT = ROOT / "data" / "audio" / "splits" / "audio_split.json"
DEFAULT_SEED = "stage11-20261110"
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Return `relative_path -> metadata` for JSON or CSV manifests."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("files", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON manifest must be a list or contain a 'files' list")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("manifest must be .json or .csv")

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("file"):
            raise ValueError("every manifest row needs a 'file' value")
        key = _relative_key(str(row["file"]))
        result[key] = {str(k): "" if v is None else str(v) for k, v in row.items()}
    return result


def _relative_key(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"manifest path is outside the repository: {value}") from exc
    return str(path).replace("\\", "/")


def group_key(record: dict[str, Any], manifest: Optional[dict[str, dict[str, str]]]) -> str:
    """Use a content group, then a verified session/source id, then the hash."""
    if manifest is None:
        return f"hash:{record['sha256']}"
    row = manifest.get(record["relative_path"].replace("\\", "/"))
    if row is None:
        raise ValueError(f"manifest has no entry for {record['relative_path']}")
    for field in ("group_id", "session_id", "source_id"):
        value = (row.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    raise ValueError(
        f"manifest entry for {record['relative_path']} has neither session_id nor source_id "
        "(and no group_id)"
    )


def manifest_mode(
    manifest: Optional[dict[str, dict[str, str]]],
    kept: list[dict[str, Any]],
) -> Optional[str]:
    """Return `content_group`, `session_or_source`, or None for no manifest."""
    if manifest is None:
        return None
    rows = [manifest[record["relative_path"].replace("\\", "/")] for record in kept]
    if any((row.get("group_id") or "").strip() for row in rows):
        return "content_group"
    return "session_or_source"


def retained_records(
    records: list[dict[str, Any]], manifest: Optional[dict[str, dict[str, str]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop non-canonical exact duplicates; verify every retained file is mapped."""
    groups = audio_integrity.duplicate_groups(records)
    for group in groups:
        classes = sorted({Path(path).parent.name for path in group["files"]})
        if len(classes) > 1:
            raise ValueError(
                f"exact duplicate {group['sha256'][:12]} spans classes {classes}; "
                "resolve it before splitting"
            )
    drop = {path for group in groups for path in group["duplicates"]}
    kept = [record for record in records if record["relative_path"] not in drop]
    excluded = [
        {
            "path": path,
            "duplicate_of": next(
                group["keep"] for group in groups if path in group["duplicates"]
            ),
        }
        for path in sorted(drop)
    ]
    if manifest is not None:
        missing = [
            record["relative_path"]
            for record in kept
            if record["relative_path"].replace("\\", "/") not in manifest
        ]
        if missing:
            raise ValueError(f"manifest is missing {len(missing)} retained file(s): {missing[:5]}")
    return kept, excluded


def _allocate(total: int) -> tuple[int, int, int]:
    """Split a small count without losing a split entirely."""
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    if total == 3:
        return 1, 1, 1
    train = max(1, int(round(total * TRAIN_FRACTION)))
    val = max(1, int(round(total * VAL_FRACTION)))
    test = total - train - val
    if test < 1:
        test = 1
        train = total - 1 - val
    return train, val, test


def _unit_order(unit: str, class_name: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{class_name}|{unit}".encode("utf-8")).hexdigest()


def build_split(
    records: list[dict[str, Any]],
    *,
    manifest: Optional[dict[str, dict[str, str]]],
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    kept, excluded = retained_records(records, manifest)
    mode = manifest_mode(manifest, kept)

    # Group retained files. A group may not span classes, otherwise a stratified
    # split would silently break the group.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in kept:
        grouped[group_key(record, manifest)].append(record)
    for key, members in grouped.items():
        classes = {member["class"] for member in members}
        if len(classes) > 1:
            raise ValueError(f"group {key} spans classes {sorted(classes)}")

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    groups_per_class: dict[str, int] = {}
    for class_name in CLASSES:
        units = sorted(
            (key for key, members in grouped.items() if members[0]["class"] == class_name),
            key=lambda key: _unit_order(key, class_name, seed),
        )
        groups_per_class[class_name] = len(units)
        train_n, val_n, _ = _allocate(len(units))
        assignment = (
            ["train"] * train_n + ["val"] * val_n + ["test"] * (len(units) - train_n - val_n)
        )
        for unit, split_name in zip(units, assignment):
            splits[split_name].extend(
                sorted(record["relative_path"] for record in grouped[unit])
            )

    for name in splits:
        splits[name].sort()

    _verify_no_group_leakage(splits, kept, manifest)
    counts = {
        name: dict(Counter(_class_of(path) for path in paths))
        for name, paths in splits.items()
    }
    source_independent = mode == "session_or_source"
    if mode == "content_group":
        grouping = "content_group_id"
        evaluation_kind = "held_out_content_group"
        provenance_status = "unverified"
        caveat = (
            "Splits are grouped by acoustic content-group id; no content group spans "
            "two splits, so repeated or near-duplicate recordings are held out "
            "together. Content groups are acoustic evidence only, not verified "
            "recording sessions or sources. Do not call these metrics "
            "source-independent."
        )
    elif source_independent:
        grouping = "session_or_source_id"
        evaluation_kind = "held_out_source_independent"
        provenance_status = "user_supplied"
        caveat = (
            "Splits are grouped by verified session/source id; no known session spans "
            "two splits."
        )
    else:
        grouping = "exact_content_hash_only"
        evaluation_kind = "held_out_file_level"
        provenance_status = "unverified"
        caveat = (
            "No provenance manifest was supplied. Exact duplicates are excluded and "
            "the split is file-level held-out, but different windows from the same "
            "unidentified recording may still span splits. Do not call these metrics "
            "source-independent."
        )

    return {
        "benchmark": "audio_split",
        "kind": "deterministic_leakage_aware_split",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset_root": str((ROOT / "data" / "audio").resolve()),
        "provenance_status": provenance_status,
        "source_independent": source_independent,
        "grouping": grouping,
        "evaluation_kind": evaluation_kind,
        "seed": seed,
        "fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION},
        "manifest": None if manifest is None else {
            "kind": "content_group" if mode == "content_group" else "session_or_source",
            "entries": len(manifest),
            "sha256": None,  # filled by the caller when a path is available
        },
        "counts": {
            "files": len(kept),
            "excluded_duplicates": len(excluded),
            "groups_per_class": groups_per_class,
            "per_split": counts,
        },
        "splits": splits,
        "excluded_duplicates": excluded,
        "caveat": caveat,
    }


def _class_of(relative_path: str) -> str:
    parts = Path(relative_path).parts
    return parts[-2] if len(parts) >= 2 else "unknown"


def _verify_no_group_leakage(
    splits: dict[str, list[str]],
    kept: list[dict[str, Any]],
    manifest: Optional[dict[str, dict[str, str]]],
) -> None:
    by_path = {record["relative_path"]: record for record in kept}
    seen: dict[str, str] = {}
    for split_name, paths in splits.items():
        for path in paths:
            key = group_key(by_path[path], manifest)
            previous = seen.get(key)
            if previous is not None and previous != split_name:
                raise ValueError(f"group {key} leaked from {previous} to {split_name}")
            seen[key] = split_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(audio_integrity.AUDIO_ROOT), help="audio root")
    parser.add_argument("--manifest", default=None, help="JSON/CSV provenance manifest")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="split JSON path")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="deterministic salt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the split without writing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = audio_integrity.scan(Path(args.root))
    manifest = None
    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
    payload = build_split(records, manifest=manifest, seed=args.seed)
    if manifest_path is not None:
        payload["manifest"] = {
            "kind": payload["manifest"]["kind"] if payload["manifest"] else None,
            "path": str(manifest_path.resolve()),
            "entries": len(manifest or {}),
            "sha256": benchmark_common.sha256_file(manifest_path),
        }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    output = Path(args.output)
    payload["output"] = str(output)
    benchmark_common.write_json(payload, output)
    print(json.dumps(payload["counts"], indent=2))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
