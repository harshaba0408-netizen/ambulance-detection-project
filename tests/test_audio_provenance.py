"""Tests for the Stage 11 audio provenance tooling.

The tests use synthetic records, tiny temporary manifests and one hand-built
RIFF header. They never scan, load or move the real audio dataset.
"""

from __future__ import annotations

import importlib.util
import csv
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(filename: str):
    name = f"stage11_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integrity = _load("audio_integrity.py")
splitter = _load("make_audio_split.py")
content_groups = _load("audio_content_groups.py")


def record(
    name: str,
    class_name: str = "siren",
    number: int = 1,
    *,
    digest: str | None = None,
    sample_rate: int = 44100,
    channels: int = 2,
    encoder: str | None = "Lavf57.83.100",
) -> dict:
    return {
        "relative_path": f"data/audio/{class_name}/{name}",
        "class": class_name,
        "number": number,
        "sha256": digest if digest is not None else f"{class_name}-{number}",
        "size_bytes": 1,
        "duration_s": 3.0,
        "sample_rate": sample_rate,
        "channels": channels,
        "subtype": "PCM_16",
        "encoder": encoder,
    }


# ---------------------------------------------------------------------------
# audio_integrity
# ---------------------------------------------------------------------------
def test_number_parsing_handles_missing_and_invalid_suffixes():
    assert integrity._number(Path("sound_42.wav")) == 42
    assert integrity._number(Path("sound_42_extra.wav")) is None
    assert integrity._number(Path("sound_abc.wav")) is None


def test_duplicate_groups_keep_the_lowest_number_and_list_the_rest():
    records = [
        record("sound_76.wav", number=76, digest="same"),
        record("sound_73.wav", number=73, digest="same"),
        record("sound_74.wav", number=74, digest="same"),
        record("sound_99.wav", number=99, digest="unique"),
    ]

    groups = integrity.duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0]["numbers"] == [73, 74, 76]
    assert groups[0]["keep"].endswith("sound_73.wav")
    assert [Path(p).name for p in groups[0]["duplicates"]] == ["sound_74.wav", "sound_76.wav"]


def test_duplicate_groups_are_empty_when_every_hash_is_unique():
    records = [record(f"sound_{n}.wav", number=n) for n in range(1, 5)]

    assert integrity.duplicate_groups(records) == []


def test_quarantine_plan_keeps_one_copy_per_group():
    records = [
        record("sound_73.wav", number=73, digest="same"),
        record("sound_76.wav", number=76, digest="same"),
    ]

    plan = integrity.quarantine_plan(records)

    assert len(plan) == 1
    assert plan[0]["keep"].endswith("sound_73.wav")
    assert [Path(p).name for p in plan[0]["quarantine"]] == ["sound_76.wav"]
    assert plan[0]["reason"] == "exact content duplicate"


def test_cohorts_split_on_a_format_or_encoder_change():
    records = [
        record("sound_1.wav", number=1),
        record("sound_2.wav", number=2),
        record("sound_3.wav", number=3, sample_rate=48000),
        record("sound_4.wav", number=4, sample_rate=48000, encoder=None),
    ]

    cohorts = integrity.cohorts(records)

    assert [(c["numbers"], c["count"], c["sample_rate"], c["encoder"]) for c in cohorts] == [
        ([1, 2], 2, 44100, "Lavf57.83.100"),
        ([3, 3], 1, 48000, "Lavf57.83.100"),
        ([4, 4], 1, 48000, None),
    ]
    assert all(c["class"] == "siren" for c in cohorts)


def test_missing_numbers_reports_gaps_per_class():
    records = [
        record("sound_1.wav", number=1),
        record("sound_3.wav", number=3),
        record("sound_401.wav", class_name="non-siren", number=401),
        record("sound_404.wav", class_name="non-siren", number=404),
    ]

    assert integrity.missing_numbers(records) == {"siren": [2], "non-siren": [402, 403]}


def test_build_report_states_unverified_provenance_and_counts_duplicates():
    records = [
        record("sound_1.wav", number=1, digest="same"),
        record("sound_2.wav", number=2, digest="same"),
        record("sound_401.wav", class_name="non-siren", number=401),
    ]

    report = integrity.build_report(records)

    assert report["provenance_status"] == "unverified"
    assert report["source_metadata_present"] is False
    assert report["summary"]["files"] == 3
    assert report["summary"]["duplicate_groups"] == 1
    assert report["summary"]["duplicate_files"] == 1
    assert "not verified recording sessions" in report["caveat"]


def _write_riff_with_encoder(path: Path, encoder: str) -> None:
    tag = encoder.encode("latin1")
    payload = b"INFO" + b"ISFT" + struct.pack("<I", len(tag) + 1) + tag + b"\x00"
    if len(payload) % 2:
        payload += b"\x00"
    body = (
        b"WAVE"
        + b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)
        + b"LIST" + struct.pack("<I", len(payload)) + payload
        + b"data" + struct.pack("<I", 0)
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_riff_encoder_reads_isft_and_ignores_other_files(tmp_path):
    wav = tmp_path / "tagged.wav"
    _write_riff_with_encoder(wav, "Lavf57.83.100")
    not_riff = tmp_path / "notes.txt"
    not_riff.write_bytes(b"plain text")

    assert integrity._riff_encoder(wav) == "Lavf57.83.100"
    assert integrity._riff_encoder(not_riff) is None


# ---------------------------------------------------------------------------
# make_audio_split
# ---------------------------------------------------------------------------
def test_allocate_never_leaves_a_large_split_empty_but_small_counts_may():
    assert splitter._allocate(0) == (0, 0, 0)
    assert splitter._allocate(1) == (1, 0, 0)
    assert splitter._allocate(2) == (1, 1, 0)
    assert splitter._allocate(3) == (1, 1, 1)
    train, val, test = splitter._allocate(20)
    assert (train, val, test) == (14, 3, 3)


def test_build_split_excludes_exact_duplicates_from_every_split():
    records = [
        record("sound_1.wav", number=1, digest="dup"),
        record("sound_2.wav", number=2, digest="dup"),
        record("sound_3.wav", number=3),
        record("sound_401.wav", class_name="non-siren", number=401),
    ]

    payload = splitter.build_split(records, manifest=None)

    assigned = payload["splits"]["train"] + payload["splits"]["val"] + payload["splits"]["test"]
    assert len(assigned) == len(set(assigned)) == 3
    assert not any(path.endswith("sound_2.wav") for path in assigned)
    assert [Path(item["path"]).name for item in payload["excluded_duplicates"]] == ["sound_2.wav"]
    assert payload["source_independent"] is False
    assert payload["provenance_status"] == "unverified"
    assert payload["grouping"] == "exact_content_hash_only"
    assert "Do not call these metrics" in payload["caveat"]


def test_build_split_rejects_an_exact_duplicate_that_spans_classes():
    records = [
        record("sound_1.wav", number=1, digest="cross"),
        record("sound_401.wav", class_name="non-siren", number=401, digest="cross"),
    ]

    with pytest.raises(ValueError, match="spans classes"):
        splitter.build_split(records, manifest=None)


def test_build_split_is_deterministic_for_the_same_seed():
    records = [
        record(f"sound_{n}.wav", number=n)
        for n in range(1, 21)
    ] + [
        record(f"sound_{n}.wav", class_name="non-siren", number=n)
        for n in range(401, 421)
    ]

    first = splitter.build_split(records, manifest=None, seed="alpha")
    second = splitter.build_split(records, manifest=None, seed="alpha")
    other = splitter.build_split(records, manifest=None, seed="beta")

    assert first["splits"] == second["splits"]
    assert first["splits"] != other["splits"]


def test_build_split_counts_every_class_in_every_allocated_split():
    records = [
        record(f"sound_{n}.wav", number=n) for n in range(1, 11)
    ] + [
        record(f"sound_{n}.wav", class_name="non-siren", number=n) for n in range(401, 411)
    ]

    payload = splitter.build_split(records, manifest=None)

    assert payload["counts"]["per_split"]["train"] == {"non-siren": 7, "siren": 7}
    assert payload["counts"]["per_split"]["val"] == {"non-siren": 2, "siren": 2}
    assert payload["counts"]["per_split"]["test"] == {"non-siren": 1, "siren": 1}


def test_load_manifest_reads_csv_and_normalises_slashes(tmp_path):
    manifest = tmp_path / "provenance.csv"
    manifest.write_text(
        "file,session_id,source_id,license\n"
        "data/audio/siren/sound_1.wav,S1,alpha,CC-BY-4.0\n",
        encoding="utf-8",
    )

    rows = splitter.load_manifest(manifest)

    assert rows["data/audio/siren/sound_1.wav"]["session_id"] == "S1"
    assert rows["data/audio/siren/sound_1.wav"]["license"] == "CC-BY-4.0"


def test_load_manifest_reads_json_and_rejects_unknown_suffix(tmp_path):
    manifest = tmp_path / "provenance.json"
    manifest.write_text(
        json.dumps({"files": [{"file": "data/audio/siren/sound_1.wav", "source_id": "mic-1"}]}),
        encoding="utf-8",
    )

    rows = splitter.load_manifest(manifest)

    assert rows["data/audio/siren/sound_1.wav"]["source_id"] == "mic-1"
    unknown = tmp_path / "provenance.txt"
    unknown.write_text("file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="json or .csv"):
        splitter.load_manifest(unknown)


def test_manifest_mode_keeps_every_session_inside_one_split():
    records = []
    manifest = {}
    for index in range(1, 9):
        name = f"sound_{index}.wav"
        records.append(record(name, number=index))
        manifest[f"data/audio/siren/{name}"] = {"session_id": "S1" if index <= 4 else "S2"}
    for index in range(401, 409):
        name = f"sound_{index}.wav"
        records.append(record(name, class_name="non-siren", number=index))
        manifest[f"data/audio/non-siren/{name}"] = {"session_id": "N1" if index <= 404 else "N2"}

    payload = splitter.build_split(records, manifest=manifest)

    assert payload["source_independent"] is True
    assert payload["provenance_status"] == "user_supplied"
    assert payload["grouping"] == "session_or_source_id"
    placements = {}
    for name, paths in payload["splits"].items():
        for path in paths:
            session = manifest[path]["session_id"]
            placements.setdefault(session, set()).add(name)
    assert all(len(names) == 1 for names in placements.values())


def test_manifest_mode_rejects_unmapped_retained_files():
    records = [record("sound_1.wav", number=1)]

    with pytest.raises(ValueError, match="manifest is missing"):
        splitter.build_split(records, manifest={})


def test_manifest_mode_rejects_rows_without_session_or_source():
    records = [record("sound_1.wav", number=1)]
    manifest = {"data/audio/siren/sound_1.wav": {"license": "unknown"}}

    with pytest.raises(ValueError, match="neither session_id nor source_id"):
        splitter.build_split(records, manifest=manifest)


def test_manifest_mode_rejects_a_group_that_spans_classes():
    records = [
        record("sound_1.wav", number=1),
        record("sound_401.wav", class_name="non-siren", number=401),
    ]
    manifest = {
        "data/audio/siren/sound_1.wav": {"session_id": "shared"},
        "data/audio/non-siren/sound_401.wav": {"session_id": "shared"},
    }

    with pytest.raises(ValueError, match="spans classes"):
        splitter.build_split(records, manifest=manifest)


def test_relative_key_rejects_paths_outside_the_repository(tmp_path):
    outside = Path(tmp_path.anchor) / "outside_repo_audio.wav"

    with pytest.raises(ValueError, match="outside the repository"):
        splitter._relative_key(str(outside))


def test_group_key_prefers_session_then_source_then_fails():
    source = {"relative_path": "data/audio/siren/sound_1.wav", "sha256": "abc"}

    assert splitter.group_key(source, None) == "hash:abc"
    assert splitter.group_key(
        source, {"data/audio/siren/sound_1.wav": {"session_id": "S1", "source_id": "mic"}}
    ) == "session_id:S1"
    assert splitter.group_key(
        source, {"data/audio/siren/sound_1.wav": {"source_id": "mic"}}
    ) == "source_id:mic"
    with pytest.raises(ValueError, match="neither session_id nor source_id"):
        splitter.group_key(source, {"data/audio/siren/sound_1.wav": {}})


# ---------------------------------------------------------------------------
# Stage 12 content-group mode
# ---------------------------------------------------------------------------
def test_group_key_prefers_group_id_over_session_and_source():
    source = {"relative_path": "data/audio/siren/sound_1.wav", "sha256": "abc"}
    manifest = {
        "data/audio/siren/sound_1.wav": {
            "group_id": "G1",
            "session_id": "S1",
            "source_id": "mic",
        }
    }

    assert splitter.group_key(source, manifest) == "group_id:G1"


def test_manifest_mode_detects_group_ids_and_reports_the_mode():
    kept = [record("sound_1.wav", number=1)]
    path = "data/audio/siren/sound_1.wav"

    assert splitter.manifest_mode(None, kept) is None
    assert splitter.manifest_mode({path: {"session_id": "S1"}}, kept) == "session_or_source"
    assert splitter.manifest_mode({path: {"source_id": "mic"}}, kept) == "session_or_source"
    assert splitter.manifest_mode({path: {"group_id": "G1"}}, kept) == "content_group"
    assert (
        splitter.manifest_mode({path: {"group_id": "G1", "session_id": "S1"}}, kept)
        == "content_group"
    )


def _content_group_manifest():
    records = []
    manifest = {}
    for index, group in enumerate(("A1", "A1", "A2", "A3", "A4", "A5"), start=1):
        name = f"sound_{index}.wav"
        records.append(record(name, number=index))
        manifest[f"data/audio/siren/{name}"] = {"group_id": group}
    for index, group in enumerate(("B1", "B1", "B2", "B3", "B4", "B5"), start=401):
        name = f"sound_{index}.wav"
        records.append(record(name, class_name="non-siren", number=index))
        manifest[f"data/audio/non-siren/{name}"] = {"group_id": group}
    return records, manifest


def test_content_group_mode_keeps_each_group_inside_one_split_and_is_not_source_independent():
    records, manifest = _content_group_manifest()

    payload = splitter.build_split(records, manifest=manifest, seed="stage12-test")

    assert payload["source_independent"] is False
    assert payload["provenance_status"] == "unverified"
    assert payload["grouping"] == "content_group_id"
    assert payload["evaluation_kind"] == "held_out_content_group"
    assert payload["manifest"]["kind"] == "content_group"
    assert payload["counts"]["files"] == 12
    assert payload["counts"]["groups_per_class"] == {"siren": 5, "non-siren": 5}

    assigned = payload["splits"]["train"] + payload["splits"]["val"] + payload["splits"]["test"]
    assert len(assigned) == len(set(assigned)) == 12
    placements = defaultdict(set)
    for split_name, paths in payload["splits"].items():
        for path in paths:
            placements[manifest[path]["group_id"]].add(split_name)
    assert all(len(names) == 1 for names in placements.values())
    assert "not verified" in payload["caveat"]
    assert "source-independent" in payload["caveat"]


def test_session_manifest_still_reports_source_independent_evaluation_kind():
    records = []
    manifest = {}
    for index in range(1, 7):
        name = f"sound_{index}.wav"
        records.append(record(name, number=index))
        manifest[f"data/audio/siren/{name}"] = {"session_id": "S1" if index <= 3 else "S2"}
    for index in range(401, 407):
        name = f"sound_{index}.wav"
        records.append(record(name, class_name="non-siren", number=index))
        manifest[f"data/audio/non-siren/{name}"] = {"session_id": "N1"}

    payload = splitter.build_split(records, manifest=manifest)

    assert payload["source_independent"] is True
    assert payload["grouping"] == "session_or_source_id"
    assert payload["evaluation_kind"] == "held_out_source_independent"
    assert payload["manifest"]["kind"] == "session_or_source"


def test_union_find_keeps_the_smallest_root():
    union = content_groups.UnionFind(5)
    union.union(3, 1)
    union.union(4, 3)
    assert union.find(3) == union.find(4) == 1
    union.union(1, 0)
    groups = union.groups()

    assert sorted(groups[0]) == [0, 1, 3, 4]
    assert groups[2] == [2]


def test_best_lag_r_finds_a_shifted_copy():
    generator = np.random.default_rng(0)
    left = generator.standard_normal(4000)
    right = np.concatenate([np.zeros(37), left])[: len(left)]

    r, lag, approximate = content_groups.best_lag_r(left, right, min_overlap=100)

    assert r > 0.999
    assert abs(lag) == 37
    assert approximate >= content_groups.SIREN_APPROX_R


def test_exact_r_is_one_for_an_identical_signal():
    signal = np.random.default_rng(1).standard_normal(1000)

    assert content_groups.exact_r(signal, signal, 0) == pytest.approx(1.0)


def test_build_rows_groups_near_identical_non_siren_signals():
    generator = np.random.default_rng(2)
    first = generator.standard_normal(1000)
    records = [
        record("sound_401.wav", class_name="non-siren", number=401, digest="a"),
        record("sound_402.wav", class_name="non-siren", number=402, digest="b"),
        record("sound_403.wav", class_name="non-siren", number=403, digest="c"),
        record("sound_404.wav", class_name="non-siren", number=404, digest="d"),
    ]
    signals = {
        "data/audio/non-siren/sound_401.wav": first,
        "data/audio/non-siren/sound_402.wav": first.copy(),
        "data/audio/non-siren/sound_403.wav": generator.standard_normal(1000),
        "data/audio/non-siren/sound_404.wav": generator.standard_normal(1000),
    }

    rows, summary = content_groups.build_rows(records, signals=signals)

    assert summary["groups_per_class"] == {"non-siren": 3}
    by_file = {row["file"]: row for row in rows}
    assert by_file["data/audio/non-siren/sound_401.wav"]["group_id"] == (
        by_file["data/audio/non-siren/sound_402.wav"]["group_id"]
    )
    assert by_file["data/audio/non-siren/sound_401.wav"]["max_r"] == "1.0000"
    assert by_file["data/audio/non-siren/sound_403.wav"]["group_method"] == "singleton"
    assert by_file["data/audio/non-siren/sound_403.wav"]["max_r"] == ""


def test_build_rows_unions_and_labels_the_freesound_reference_files():
    signal = np.random.default_rng(3).standard_normal(1000)
    numbers = (13, 14, 15, 85, 86, 87)
    records = [
        record(f"sound_{number}.wav", number=number, digest=f"s{number}") for number in numbers
    ]
    signals = {f"data/audio/siren/sound_{number}.wav": signal.copy() for number in numbers}

    rows, summary = content_groups.build_rows(records, signals=signals)

    assert summary["groups_per_class"] == {"siren": 1}
    assert len({row["group_id"] for row in rows}) == 1
    by_file = {row["file"]: row for row in rows}
    for number in (85, 86, 87):
        row = by_file[f"data/audio/siren/sound_{number}.wav"]
        assert row["source_confidence"] == "verified"
        assert row["source_id"] == "freesound:87130"
        assert row["license"] == "CC-BY-4.0"
    for number in (13, 14, 15):
        assert by_file[f"data/audio/siren/sound_{number}.wav"]["source_confidence"] == "probable"
    assert all(row["group_method"] == "siren reference union (Freesound 87130)" for row in rows)


def test_content_group_csv_is_complete_and_consistent():
    path = ROOT / "data" / "audio" / "content_groups.csv"
    assert path.is_file(), f"missing generated content-group manifest: {path}"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(content_groups.FIELDS)
    assert len(rows) == 396
    files = [row["file"] for row in rows]
    assert len(set(files)) == len(files)

    members = defaultdict(list)
    for row in rows:
        members[row["group_id"]].append(row["file"])
    classes = {
        group_id: {path.split("/")[-2] for path in paths}
        for group_id, paths in members.items()
    }
    assert all(len(values) == 1 for values in classes.values())

    non_siren_groups = [
        group_id for group_id, paths in members.items() if paths[0].split("/")[-2] == "non-siren"
    ]
    assert len(non_siren_groups) == 60
    assert Counter(len(members[group_id]) for group_id in non_siren_groups) == {4: 20, 3: 40}

    siren_groups = [
        group_id for group_id, paths in members.items() if paths[0].split("/")[-2] == "siren"
    ]
    assert len(siren_groups) == 174
    reference_groups = [
        group_id
        for group_id in siren_groups
        if any(Path(path).stem in {f"sound_{number}" for number in (13, 14, 15, 85, 86, 87)} for path in members[group_id])
    ]
    assert len(reference_groups) == 1
    reference_files = {Path(path).stem for path in members[reference_groups[0]]}
    assert reference_files == {f"sound_{number}" for number in (13, 14, 15, 85, 86, 87)}

