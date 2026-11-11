# Audio Dataset Provenance: Stage 11–12 Investigation

**Status:** Stage 11 investigation complete; Stage 12 content-group split and retrain complete; provenance **partially recovered**; source-independent claims remain blocked
**Last updated:** 2026-11-11
**Scope:** `data/audio/` — 400 local WAV files used by the siren classifier

This document records what can actually be verified about the audio dataset, what
cannot, and what that means for the Stage 11 file-level split, the Stage 12
content-group split and their metrics. It deliberately separates three different
things that are easy to confuse:

1. **verified source metadata** (a named recording, URL and licence);
2. **content groups** (clips that are copies or overlapping windows of the same
   audio, proven by waveform correlation);
3. **recording/session identity** (which clips came from one field session).

Only the first is provenance. A content group is evidence of reuse, not proof of
a session or source. No `session_id` is assigned anywhere in this document
unless the source was independently verified.

---

## 0. Findings at a glance

| Finding | Result |
|---|---|
| Files traced to a named, licensed source | **3 confirmed** (`sound_85`, `86`, `87`) plus **3 probable** (`sound_13`, `14`, `15`) |
| Verified source | Freesound **87130**, `20091229.ambulance.siren.wav` by **dobroide** |
| Verified licence | **CC BY 4.0** (attribution required) |
| Files with no verified source/licence | **397** (all 200 non-siren, 194 other siren, including the 3 probable `13`–`15`) |
| Non-siren independence | **Not independent.** 200 files reduce to **60 unique 3 s clips**, each repeated 3–4 times |
| Siren independence | **Not independent.** 4 byte-identical pairs, 3 further near-identical pairs, and 32 related pairs at `r >= 0.80` after alignment |
| Verified `session_id`/`source_id` manifest | **Cannot be built** from the local evidence |
| Provisional split validity | **Contaminated.** 44 of 60 non-siren content groups cross splits; every non-siren test file has a same-content sibling in train |
| Stage 12 content-group split | **Generated.** 234 acoustic groups (60 non-siren, 174 siren); **0 groups span splits** |
| Stage 12 result label | **`held_out_content_group`**; `source_independent` remains **false** |
| Stored `1.0000` audio scores | **Not valid held-out results**; the Stage 11 file-level result must stay labelled duplicate-contaminated |
| Stage 12 retrained score | **1.0000** val/test, but only content-group-held-out; **not** a source-independent result |
| Shipped `siren_classifier.pkl` | **Unchanged** |

The headline conclusion is therefore narrower than the old README implies: the
dataset is a **400-file collection containing far fewer independent recordings**,
and only one small part of it can currently be attributed and licensed.

---

## 1. Method and evidence base

The investigation was read-only. It covered:

- the 400 WAV files and their RIFF chunks (`fmt `, `LIST/INFO`, `data`);
- filenames, durations, sample rates, channel counts and encoder tags;
- the repository, its Git history and unreachable Git objects;
- the archived ZIP snapshot of the project;
- the two other local copies of the project;
- NTFS alternate data streams (download-origin metadata);
- the shipped `siren_classifier.pkl` (training metadata);
- content comparison against public web sources and the local video audio.

Content comparisons used:

- **non-siren:** zero-lag waveform Pearson correlation between candidate copies,
  after mono downmix and resampling to a common rate;
- **siren:** normalized cross-correlation at every lag, with a minimum overlap
  of 1.5 s, then explicit Pearson confirmation at the best lag;
- **known reference recordings:** normalized cross-correlation of each local
  clip against the full public preview, at 8 kHz mono, reporting the best offset.

Correlation is computed on waveforms decoded/resampled with `scipy.signal`/
`librosa`. MP3 previews cannot reach `r = 1.0` against PCM WAVs because of
lossy coding, so the reported values are lower bounds on the match.

No provenance was inferred from filename numbering, class labels or encoding
cohorts. Those are recorded only as weak administrative evidence.

---

## 2. Verified inventory

The audit was produced by:

```powershell
.\venv\Scripts\python.exe scripts\audio_integrity.py
```

Stored result: `benchmarks/results/audio_integrity_20261111T071208Z.json`.
The audit is read-only; it hashes and inspects files but never edits them.

| Property | Siren | Non-siren |
|---|---|---|
| Files | 200 | 200 |
| Filenames | `sound_1.wav` … `sound_200.wav` | `sound_401.wav` … `sound_600.wav` |
| Missing numbers | none | none |
| Duration | 3.000 s (one file 3.019 s) | 3.000 s |
| Sample rates | 22050, 44100, 48000 Hz | 44100 Hz |
| Channels | mono and stereo | stereo |
| Sample format | PCM_16 | PCM_16 |

All 400 files are exactly 3.0 s except `sound_200.wav` (3.019 s). There are no
numbering gaps. File timestamps (2025-11-01) are OneDrive sync times, not
recording dates, and are not used as provenance.

### 2.1 Embedded RIFF metadata

Siren files have a RIFF `LIST/INFO` chunk; non-siren files do not. The usual
field is `ISFT`, the ffmpeg/Lavf build that wrote the file:

| `ISFT` value | Siren files | Consecutive ranges |
|---|---:|---|
| `Lavf57.83.100` | 153 | 1–84, 91–159 |
| absent | 6 | 85–90 |
| `Lavf58.27.103` | 1 | 160 |
| `Lavf58.20.100` | 40 | 161–200 |

`ISFT` identifies **encoding batches**, not recording sessions. A batch may
contain unrelated recordings, and one recording may be re-encoded more than once.
The Stage 11 audit therefore calls these "encoding cohorts" and never treats them
as independent sources.

Only six siren files carry any non-encoder metadata:

| Files | Tags | Meaning |
|---|---|---|
| `sound_85`, `86`, `87` | `IART=dobroide`, `ICRD=2009`, `ISFT=Lavf57.83.100` | Artist and year preserved |
| `sound_88`, `89`, `90` | `ICRD=2011-05-18`, `ITCH=ZOOM Handy Recorder H2n` | Recording date and recorder |

No other `IART`, `ICRD`, `ICMT`, `ISRC`, `IGNR` or location tags exist. The
non-siren files have no `LIST/INFO` chunk at all. This is the entire embedded
provenance surface.

### 2.2 Exact duplicates

Four siren files are byte-identical:

| Kept | Duplicate | SHA-256 |
|---|---|---|
| `sound_73.wav` | `sound_76.wav` | identical |
| `sound_74.wav` | `sound_77.wav` | identical |
| `sound_75.wav` | `sound_78.wav` | identical |
| `sound_129.wav` | `sound_130.wav` | identical |

They are excluded from the provisional split. The earlier statement that the
non-siren set has no duplicates was **wrong**: it has no *byte-identical* pairs,
but it contains extensive content-level duplication, documented next.

---

## 3. Content groups: the dataset is much smaller than 400 recordings

### 3.1 Non-siren: 200 files = 60 unique clips

The 200 non-siren files are built from **60 unique 3-second clips**, repeated in
fixed 60-file blocks:

- **20 groups of 4:** `sound_401/461/521/581`, …, `sound_420/480/540/600`;
- **40 groups of 3:** `sound_421/481/541`, …, `sound_460/520/580`.

That accounts for exactly `20 × 4 + 40 × 3 = 200` files.

Pairwise zero-lag waveform correlation within these 240 pairs:

| Statistic | Min | Median | Max |
|---|---:|---:|---:|
| Pearson `r` | 0.9905 | 0.9964 | 0.9989 |
| Max absolute sample difference | 691.5 | 1079.0 | 5744.5 |
| RMS difference | 148.5 | 227.5 | 410.4 |
| Exactly equal samples | 0.052 % | 0.089 % | 0.163 % |

The copies are **not byte-identical**; they are re-encoded or re-rendered copies
of the same underlying 3 s audio. Example for the first group:

| Pair | `r` | Max abs diff |
|---|---:|---:|
| 401 vs 461 | 0.99052 | 5673 |
| 401 vs 521 | 0.99084 | 5744 |
| 401 vs 581 | 0.99060 | 5737 |
| 461 vs 521 | 0.99709 | 1066 |
| 461 vs 581 | 0.99737 | 958 |
| 521 vs 581 | 0.99714 | 1321 |

This is decisive content-level duplication. Whatever the original non-siren
source was, the 200 files do not represent 200 independent traffic scenes. They
represent 60 unique clips, with the block pattern suggesting a deliberate
augmentation/balancing step (possibly different codecs or channels) rather than
200 field recordings.

These are **content groups**, not verified recording sessions. The underlying 60
clips may themselves come from fewer than 60 sessions; that cannot be determined
from the local files.

### 3.2 Siren: exact, near-exact and shifted reuse

Seven siren pairs match at zero lag with `r >= 0.99998`:

| Pair | `r` | Note |
|---|---:|---|
| `73` ↔ `76` | 1.000000 | byte-identical |
| `74` ↔ `77` | 1.000000 | byte-identical |
| `75` ↔ `78` | 1.000000 | byte-identical |
| `129` ↔ `130` | 1.000000 | byte-identical |
| `1` ↔ `34` | 0.999993 | different files, same waveform |
| `2` ↔ `35` | 0.999992 | different files, same waveform |
| `3` ↔ `36` | 0.999981 | different files, same waveform |

A best-lag search then found **32 pairs with `r >= 0.80`**, including shifted
segments. The strongest non-zero-lag examples:

| Pair | `r` | Absolute offset |
|---|---:|---:|
| `2` ↔ `31` | 0.9925 | 0.392 s |
| `31` ↔ `35` | 0.9924 | 0.392 s |
| `32` ↔ `36` | 0.9895 | 0.392 s |
| `3` ↔ `32` | 0.9895 | 0.392 s |
| `110` ↔ `41` | 0.9848 | 1.019 s |
| `111` ↔ `42` | 0.9785 | 1.019 s |
| `11` ↔ `23` | 0.9800 | 0.168 s |
| `12` ↔ `24` | 0.9733 | — |
| `10` ↔ `22` | 0.9731 | — |
| `11` ↔ `31` | 0.9571 | — |
| `12` ↔ `32` | 0.9364 | — |
| `11` ↔ `2`/`35` | 0.9128 | — |
| `14` ↔ `87` | 0.9123 | — |
| `10` ↔ `34` | 0.8891 | — |
| `1` ↔ `10` | 0.8890 | — |
| `12` ↔ `36`/`3` | 0.8875 | — |
| `23` ↔ `31` | 0.8817 | — |
| `20` ↔ `71` | 0.8728 | — |
| `13` ↔ `86` | 0.8690 | — |
| `16` ↔ `18` | 0.8581 | — |
| `24` ↔ `32` | 0.8315 | — |
| `2` ↔ `23`/`35` | 0.8210 | — |

These are consistent with overlapping windows of the same siren pass or with
repeated use of the same siren material. They are **not** proof of a shared
session: two different recordings of the same siren type can correlate strongly.
They are reported as **acoustic/content groups**, not session IDs.

The cross-reference to the verified Freesound recording (`14 ↔ 87`, `13 ↔ 86`)
is significant and is treated in section 4.

### 3.3 What the content groups mean for independence

The important consequence is not the exact group count. It is that **file-level
independence is false**:

- every non-siren file has at least two other copies of its content;
- every siren file in the shifted/duplicate groups has at least one related file;
- a random split can put copies on both sides of the train/test boundary.

Content groups can be used as a **leakage-control key**. They cannot be used as a
verified `session_id`, because a content group may be a copy of a copy rather
than the original recording session.

---

## 4. Verified source: Freesound 87130 (`sound_85`–`87`)

### 4.1 Source record

| Field | Value |
|---|---|
| Title | `20091229.ambulance.siren.wav` |
| Recordist | **dobroide** |
| Freesound ID | **87130** |
| URL | <https://freesound.org/people/dobroide/sounds/87130/> |
| Recorded | 2009-12-29 |
| Uploaded | 2010-01-03 |
| Location | Seville, Spain (Feria and Resolana St crossing) |
| Equipment | Soundman OKM into Olympus LS10 |
| Duration | 60.316 s |
| Format | 44.1 kHz, 16-bit, stereo |
| Licence | **Attribution 4.0 (CC BY 4.0)** — <https://creativecommons.org/licenses/by/4.0/> |
| Original download name | `87130__dobroide__20091229ambulancesiren.wav` |

The public HQ preview is:

```text
https://cdn.freesound.org/previews/87/87130_8043-hq.mp3
```

Freesound page description: starts with heavy rain, then an ambulance siren
passing very close, then normal traffic and a pedestrian-crossing beeper.

### 4.2 Content verification

Each local clip was correlated against the full 60.316 s preview at 8 kHz. The
match is per stereo channel, because the source is binaural and its channels are
close to out of phase (`r = -0.17` between the two channels of `sound_85`).

| Local file | Best offset in preview | `r`, preview L ↔ file L | `r`, preview R ↔ file R |
|---|---:|---:|---:|
| `sound_85.wav` | 0.000 s | 0.99846 | 0.99858 |
| `sound_86.wav` | 3.000 s | 0.99877 | 0.99883 |
| `sound_87.wav` | 6.000 s | 0.99895 | 0.99893 |

This is a near-exact three-channel-pair match. The three files are the first
9 seconds of the recording, cut into consecutive 3-second windows. The
`IART=dobroide` / `ICRD=2009` tags independently agree with the source page.

**Confidence: verified.** The original WAV was not downloaded; the match is
against the public preview, but the per-channel `r > 0.998` and the metadata
agreement are decisive.

### 4.3 Additional probable excerpts (`sound_13`–`15`)

A full-reference scan of all 200 siren files found no other match above `0.22`
except `sound_13`, `14`, `15`:

| Local file | Best offset | Mono `r` |
|---|---:|---:|
| `sound_13.wav` | 2.714 s | 0.9609 |
| `sound_14.wav` | 5.714 s | 0.9694 |
| `sound_15.wav` | 8.714 s | 0.9609 |

The offsets are exactly 3 seconds apart, matching the same window grid as
`sound_85`–`87` but shifted by 2.714 s. The files are dual-mono (L == R exactly),
unlike the binaural `sound_85`–`87`. Their per-channel correlation with the
preview is only ~0.55–0.63; the mono mixdown reaches ~0.96.

This is a strong content relationship but not the exact channel match seen for
`85`–`87`. Possible explanations are a different mono mixdown, a lossy/processed
copy, or a different encode of the same session. **Confidence: probable, not
verified.** Do not count `13`–`15` as independent of `85`–`87`, and do not use
them to claim 6 verified files.

### 4.4 Licence consequence

For `sound_85`–`87`, the dataset must carry the attribution:

> "20091229.ambulance.siren.wav" by dobroide, Freesound sound 87130,
> licensed under CC BY 4.0. <https://freesound.org/s/87130/>

If `sound_13`–`15` are confirmed as the same recording, the same attribution
covers them. Nothing else in the audio dataset has a verified licence.

---

## 5. Unresolved source: `sound_88`–`90`

These three files carry:

```text
ICRD = 2011-05-18
ITCH = ZOOM Handy Recorder H2n
```

They have no `IART`, no `ISFT`, and no other source field. Their content does
**not** match Freesound 87130 (`r = 0.056`–`0.095` at the best offset). They
share a recorder/date tag and consecutive numbering, which suggests one
recording, but that is not proof.

Searches performed:

| Search | Result |
|---|---|
| Freesound `20110518` (global) | 0 sounds |
| Freesound `20110518` by `dobroide` | 0 sounds |
| `dobroide` sounds matching "ambulance" | 16, none from 2011 |
| Freesound `ambulance H2n` (182 parsed) | no 2011-05-18 title |
| Freesound `siren H2n` (195 parsed) | two 2011-titled candidates, neither matching the metadata |
| Web search for `"ZOOM Handy Recorder H2n"` + 2011 ambulance | no relevant result |

**Source and licence: unresolved.** The files remain in the dataset but must be
treated as unlicensed/unverified.

---

## 6. Repository, history and filesystem search

The following were checked for download scripts, notebooks, source URLs or
licence files:

| Location | Result |
|---|---|
| `data/audio/siren/sample.py` | Only a `wave`/`pylab` spectrogram generator; no download or extraction logic |
| `data/audio/non-siren/sample.py` | Same spectrogram generator |
| Current repository and Git history | No download/preprocessing notebook or script |
| Old commit `a290f5063` | Tracked all 400 WAVs and the same `sample.py`; no CSV, metadata or download script |
| Old commit `2e9a80ffc` | Contains YouTube/stock-footage **video** files; vision sources only |
| Old `test_siren.py` | References `D:/project/ambulance_detection_project/data/audio/...`; that path does not exist |
| Project ZIP archive | Older repo snapshot, `venv`, WAVs, same `sample.py`; no provenance file; `New folder/` empty |
| Alternate local copies | Same project/older upstream; no audio provenance metadata |
| Unreachable Git objects | 1 commit and 12 blobs: docs/README/LICENSE/test files only |
| NTFS alternate streams on sampled WAVs | Only `:$DATA`; no `Zone.Identifier` download URL |
| `siren_classifier.pkl` | RandomForest (100 estimators, `random_state=42`, 40 features); no training paths or source metadata |

The ZIP and alternate copies were inspected but not modified. No other
`sound_401.wav` or audio dataset directory was found outside the current project
and the two copies noted above.

Old READMEs mention "400 balanced audio files" but give no source or licence.
Only the vision dataset has a documented CC BY 4.0 licence.

---

## 7. Video audio cross-check (negative)

The local video audio tracks `data/Video/1.wav`, `2.wav`, `3.wav` were compared
against all 400 clips at 8 kHz with normalized best-lag correlation. Highest
matches:

| Video track | Best local match | `r` |
|---|---|---:|
| `1.wav` | `sound_163.wav` | 0.1294 |
| `2.wav` | `sound_103.wav` | 0.1024 |
| `3.wav` | `sound_132.wav` | 0.0852 |

No content match. The video files are not the source of any audio clip. This is
negative evidence only; it does not identify the actual source.

---

## 8. Why this blocks a source-independent claim

The classifier uses 40 MFCCs from 3-second windows. If two clips are copies or
overlapping windows of the same recording, a random file-level split can put one
copy in training and another in test. The model then looks better than it is.

That is not hypothetical here. In the Stage 11 provisional split (now
superseded by the content-group split in section 9.1):

- **44 of 60 non-siren content groups span more than one split**;
- the train split contains at least one copy of **all 60** non-siren groups;
- the test split has 30 non-siren files covering 27 groups, and **all 27 have a
  same-content sibling in train**;
- the validation split has 30 non-siren files covering 28 groups, all with a
  train sibling;
- at least one retained near-exact siren pair (`sound_2` ↔ `sound_35`) spans
  train/test; the 32 related siren pairs were not all mapped to splits, so the
  siren leakage count is a lower bound.

Consequently, for the **Stage 11 split only**:

- that split is **file-level held-out by filename**, but not content-held-out;
- its stored validation/test scores of **1.0000 are not valid held-out
  estimates** — they are duplicate-contaminated;
- a perfect score on it says only that this collection separates these clips
  easily, and that some test content was already seen in training.

The correct label for the **Stage 11 stored result** is:

```text
held_out_file_level_duplicate_contaminated
```

This label applies only to `benchmarks/results/audio_training_20261111T070907Z.json`.
The Stage 12 content-group split (section 9.1) supersedes the Stage 11 split for
leakage control: no known acoustic content group spans two splits, so the Stage
12 metrics are labelled `held_out_content_group`. That supersession does **not**
make any result source-independent, session-independent, recording-independent
or environment-independent — no result in this repository may be called that,
because 394 files still have no verified source, session or licence.

---

## 9. Stage 11 split and stored result

Command:

```powershell
.\venv\Scripts\python.exe scripts\make_audio_split.py
```

Stored Stage 11 split (superseded by section 9.1): `data/audio/splits/audio_split.json`
Seed: `stage11-20261110`, fractions 70/15/15, deterministic by salted SHA-256
ordering within each class.

| Split | Siren | Non-siren | Total |
|---|---:|---:|---:|
| train | 137 | 140 | 277 |
| val | 29 | 30 | 59 |
| test | 30 | 30 | 60 |
| **retained** | **196** | **200** | **396** |

The four byte-identical siren duplicates are excluded and recorded under
`excluded_duplicates`. The generator verifies that each retained file appears in
exactly one split.

The Stage 11 split file stated its original limitation (the current file was
regenerated in section 9.1):

```json
{
  "provenance_status": "unverified",
  "source_independent": false,
  "grouping": "exact_content_hash_only"
}
```

That limitation was too weak: `grouping` covers only exact hashes, not the
near-duplicate content groups. The split is therefore **not** leakage-aware in
the sense that matters. The stored metrics are kept for reproducibility, but
they must not be quoted as a model-quality result.

The training result
`benchmarks/results/audio_training_20261111T070907Z.json` is labelled
`held_out_file_level`. That label is accurate only in the filename sense. The
result file now carries explicit validity fields (`evaluation_validity`,
`duplicate_contaminated`, `validity_note`) marking it duplicate-contaminated;
the metrics themselves were not altered.

**The Stage 11 metrics above remain unchanged.** The shipped
`siren_classifier.pkl` was not modified, and the Stage 11 experiment model remains
under `artifacts/experiments/`. Stage 12, approved separately, is recorded below.

---

## 9.1 Stage 12: content-group split and retraining

Stage 12 regenerates the split with acoustic content-group ids and retrains an
experiment model. It does not modify `siren_classifier.pkl` and does not claim
source independence.

### Grouping

`scripts/audio_content_groups.py` scans the 400 WAV files, drops the same four
byte-identical duplicates as the splitter, and writes:

- `data/audio/content_groups.csv` — 396 rows, one per retained file;
- `benchmarks/results/audio_content_groups_20261111.json` — thresholds and
  group summary.

The grouping rules are:

- **non-siren:** mono zero-lag Pearson `r >= 0.99`;
- **siren:** fixed SHA-256, zero-lag `r >= 0.995`, or best-lag `r >= 0.95`
  with at least 1.5 s of overlap;
- **reference union:** `sound_13`–`15` and `sound_85`–`87` are explicitly
  unioned because all six match the Freesound 87130 reference.

The analysis signal is 22.05 kHz mono. The best-lag search is decimated by four
for speed; every accepted lag is rechecked with an exact native-rate Pearson
`r`. The CSV carries `source_id`, `source_confidence` and `license` only for the
six Freesound files (`verified` for `85`–`87`, `probable` for `13`–`15`). For
every other row those fields are blank.

| Class | Retained files | Acoustic groups | Group sizes |
|---|---:|---:|---|
| siren | 196 | 174 | 1×6, 2×5, 1×4, 6×2, 164×1 |
| non-siren | 200 | 60 | 20×4, 40×3 |
| **total** | **396** | **234** | |

These counts must be read carefully: **174 siren groups are not 174 sources**
and **60 non-siren groups are not 60 sessions**. They are the groups that the
waveform evidence can prove. The siren grouping found only 32 files in
multi-file groups; all other siren files are singletons because no other
retained clip overlapped them enough to meet the threshold.

### Split

Command:

```powershell
.\venv\Scripts\python.exe scripts\make_audio_split.py --manifest data\audio\content_groups.csv --seed stage12-20261111
```

Stored split: `data/audio/splits/audio_split.json`. The file now reports:

```json
{
  "provenance_status": "unverified",
  "source_independent": false,
  "grouping": "content_group_id",
  "evaluation_kind": "held_out_content_group"
}
```

| Split | Siren files | Non-siren files | Total files | Siren groups | Non-siren groups |
|---|---:|---:|---:|---:|---:|
| train | 138 | 142 | 280 | 122 | 42 |
| val | 32 | 30 | 62 | 26 | 9 |
| test | 26 | 28 | 54 | 26 | 9 |
| **retained** | **196** | **200** | **396** | **174** | **60** |

Verification: **0 of 234 groups span two splits**, so every non-siren test file
now belongs to a group that is absent from train and val. The four exact
duplicates remain excluded and listed under `excluded_duplicates`.

### Retraining

Command:

```powershell
.\venv\Scripts\python.exe scripts\train_audio.py --split data\audio\splits\audio_split.json
```

| Item | Path / value |
|---|---|
| Experiment model | `artifacts/experiments/audio_rf_20261111T181213Z/` |
| Exported metrics | `benchmarks/results/audio_training_20261111T181213Z.json` |
| Evaluation label | `held_out_content_group` |
| `source_independent` | `false` |
| `duplicate_contaminated` | `false` — known content groups are held out |
| Split SHA-256 | `c878fa6a5100b0e2baf042387d1edf44c0e950053f4fcb8b8049c6a8716395da` |

| Split | Samples | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 62 | 32 | 0 | 30 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 54 | 26 | 0 | 28 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The shipped `siren_classifier.pkl` was evaluated on the same split and also
scored 1.0000 on both val and test (same TP/FP/TN/FN counts).

### Interpretation

The Stage 12 result is an honest **content-group-held-out** result: known
repeated and near-duplicate clips cannot cross the split boundary. It is still
**not source-independent**:

- content groups are acoustic evidence, not verified sessions or sources;
- different excerpts of one source that do not overlap by at least 1.5 s can
  remain separate groups;
- 394 files still have no verified source, session or licence;
- a perfect score on this collection therefore does **not** establish
  generalisation to new sirens, new recorders or new cities.

The Stage 11 file-level result
(`benchmarks/results/audio_training_20261111T070907Z.json`) remains
duplicate-contaminated and must not be quoted as held-out model quality. The
Stage 12 metrics supersede it for leakage control, but neither result is a
source-independent estimate.

---

## 10. What a valid manifest would require

A real manifest must map every retained file to an independently verified
`session_id` or `source_id`, and must carry a licence and evidence link. For this
dataset that means, at minimum:

```csv
file,session_id,source_id,license,notes
data/audio/siren/sound_85.wav,FREESOUND-87130,freesound:87130,CC-BY-4.0,https://freesound.org/s/87130/ ; offset 0.000 s
data/audio/siren/sound_86.wav,FREESOUND-87130,freesound:87130,CC-BY-4.0,same recording; offset 3.000 s
data/audio/siren/sound_87.wav,FREESOUND-87130,freesound:87130,CC-BY-4.0,same recording; offset 6.000 s
```

The current `data/audio/provenance_template.csv` is a 396-row **blank scaffold**,
not a manifest. Rows with no `session_id`/`source_id` are rejected by the split
generator, so blank provenance cannot be mistaken for verified provenance.

Two levels should be kept separate:

| Field | Meaning | Allowed use |
|---|---|---|
| `group_id` | Acoustic/content group proven by correlation | Leakage control only; not a source claim |
| `session_id` | One recording session, independently verified | Source-independent splitting |
| `source_id` | Named dataset/URL/recordist, independently verified | Licence and attribution |

If only `group_id` is available, the honest result label is
**content-group-held-out**, not source-independent.

---

## 11. Recommended next action

1. **Stop quoting the stored 1.0000 as a held-out result.** Keep the metrics for
   reproducibility, but label them duplicate-contaminated. This is a
   documentation/labelling fix, not a model change.
2. **Record the verified attribution now.** Add the CC BY 4.0 notice for
   `sound_85`–`87` (and, if accepted, the probable `13`–`15`) to the dataset
   documentation and any redistribution package. Do not claim a licence for the
   remaining files (394 if `13`–`15` are accepted, otherwise 397).
3. **Content-group path: completed in Stage 12.** The splitter now accepts a
   `group_id` column, and the regenerated split/retrain are recorded in section
   9.1. This removes the known content-group leakage; it is **not**
   source-independent.
4. **True provenance recovery remains the only path to source independence.**
   Obtain the original source URLs, sessions and licences from the dataset
   owner. Build the verified manifest, then run
   `make_audio_split.py --manifest ...` and retrain.
5. **Do not invent session IDs** to obtain the `source_independent` label. Do not
   treat `sound_88`–`90` as one session without evidence. Do not treat the 60
   non-siren groups or the 174 siren groups as sessions without evidence.
6. **Before any public redistribution,** either obtain licences for the
   unresolved files or remove them. The dataset currently cannot be redistributed
   as a whole on the evidence available.
7. **Quarantine is optional.** The four byte-identical siren copies are already
   excluded from the split. Moving them is a tidy-up, not a fix for the
   content-level leakage.

The Stage 12 split regeneration and retraining changed stored artefacts, which
is why they were approved separately. A future source-independent run still
requires the same explicit decision and a verified manifest.

---

## 12. Reproduction summary

| Step | Command | Output |
|---|---|---|
| Audit | `python scripts\audio_integrity.py` | `benchmarks/results/audio_integrity_<timestamp>.json` |
| Content groups | `python scripts\audio_content_groups.py --report benchmarks\results\audio_content_groups_20261111.json` | `data/audio/content_groups.csv` |
| Provisional file-level split (historical) | `python scripts\make_audio_split.py --seed stage11-20261110` | file-level split, now superseded |
| Content-group split | `python scripts\make_audio_split.py --manifest data\audio\content_groups.csv --seed stage12-20261111` | `data/audio/splits/audio_split.json` |
| Train + content-group evaluation | `python scripts\train_audio.py --split data\audio\splits\audio_split.json` | `artifacts/experiments/audio_rf_<timestamp>/` |
| Export metrics | copy `metrics.json` | `benchmarks/results/audio_training_<timestamp>.json` |
| Provenance tests | `python -m pytest -q tests\test_audio_provenance.py` | 31 tests |

Use `.\venv\Scripts\python.exe` in place of `python` on this machine.

The content-group analysis is now a repository script
(`scripts/audio_content_groups.py`), not an ad-hoc notebook command. It is
read-only against the WAV files and writes only the CSV and the optional JSON
summary. The verified Freesound comparison used the public HQ preview at the URL
in section 4.1.

---

## 13. Input still needed from the dataset owner

1. The original source URL or dataset name for the remaining siren files
   (`sound_88`–`200`, excluding the Freesound group).
2. The original source for all 200 non-siren files, and why each 3 s clip
   appears 3–4 times.
3. Which files share a recording session or source device.
4. The licence and redistribution terms for every source.
5. Whether `sound_1..200` are distinct siren passes, overlapping windows, or
   augmented copies.
6. Whether the four byte-identical duplicates are intentional and which copy is
   canonical.

Until items 1–4 are answered, every audio metric in this repository must stay
labelled **file-level / content-group-contaminated / unverified provenance**.
