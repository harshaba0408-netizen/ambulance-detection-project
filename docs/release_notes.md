# Release Notes — `0.3.0`

**Release date:** 2026-11-12
**Previous version:** `0.2.0`
**Status:** research and engineering prototype; not for road-side deployment

---

## 1. Summary

`0.3.0` completes the documentation, limitations and release stage of the
advanced prototype. It adds two deliberately isolated research/engineering
layers — an offline GCC-PHAT direction experiment and a simulation/mock hardware
boundary — and publishes a final research report with explicit status labels and
non-claims.

No shipped model was retrained or replaced:

- `yolov8l_ambulance.pt` is unchanged.
- `siren_classifier.pkl` is unchanged (SHA-256
  `2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c`).

---

## 2. What is new

### Stage 13 — GCC-PHAT direction research (Experimental)

- `src/ambulance_system/localisation/gcc_phat.py` — GCC-PHAT TDOA estimation
  with configurable search bounds, oversampling and multi-peak diagnostics.
- `src/ambulance_system/localisation/doa.py` — pair and 2-D far-field bearing
  solvers with explicit sign conventions and clipping behaviour.
- `src/ambulance_system/localisation/simulation.py` — synthetic plane-wave and
  two-source generators with known delays.
- `tests/test_gcc_phat.py` — 27 tests.
- `scripts/benchmark_localisation.py` — synthetic method check.
- `docs/localisation.md` — derivation, results and limitations.
- Stored result:
  `benchmarks/results/localisation_synthetic_20261111T191746Z.json`.

The package imports no fusion, signal, dashboard or device module and is not
wired into the priority decision.

### Stage 14 — Hardware abstraction (simulation/mock only)

- `src/ambulance_system/hardware/` — `HardwareController` protocol,
  safe-default base controller, default `SimulationController` and
  `MockController` test double.
- `tests/test_hardware.py` — 18 tests, including a forbidden-import source scan.
- `docs/hardware_integration.md` — interface, safety invariants and the
  requirements for any future physical adapter.

**No GPIO, relay, serial or network device is opened.** The abstraction is not
wired into the dashboard.

### Stage 15 — Documentation and release

- `docs/limitations.md` — full limitation and non-claim register.
- `docs/research_report.md` — final research report with verified results,
  status labels and CV-ready bullets.
- `docs/release_notes.md` — this file.
- Updates to `README.md`, `docs/architecture.md`, `docs/testing_report.md`,
  `docs/dataset_report.md`, `docs/BASELINE_VS_IMPROVED.md` and
  `docs/ADVANCED_DEVELOPMENT_ROADMAP.md`.
- Package version bumped from `0.2.0` to `0.3.0`.

---

## 3. Verification

Run on 2026-11-12, Python 3.12.5, CPU-only Windows:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

```text
443 passed, 1 warning in 40.20s
```

Coverage:

```powershell
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

```text
443 passed, 1 warning in 66.40s
TOTAL   3259 statements   255 missed   92%
```

The warning is the existing Librosa `audioread` deprecation notice. The 16
legacy tests remain unchanged. Timing is load-sensitive: a later identical plain
run on the same machine took 63.05 s, with the same pass count and warning.

Synthetic localisation benchmark:

```powershell
.\venv\Scripts\python.exe scripts\benchmark_localisation.py
```

```text
single-source synthetic: 280 trials, delay MAE 1.16 us, bearing MAE 0.498 deg, within 2 deg 100.0%
two-source synthetic: 15/15 both peaks detected (100.0%)
```

---

## 4. Upgrade notes

- The new modules are additive; no existing public interface was removed.
- `ambulance_system.__version__` is now `0.3.0`.
- The localisation and hardware packages are **not** imported by the dashboard
  runtime. Existing launch commands are unchanged.
- `benchmarks/results/localisation_synthetic_20261111T191746Z.json` is published
  through a `.gitignore` negation, following the existing result-provenance
  pattern.

---

## 5. Known limitations at release

See [`limitations.md`](limitations.md) for the full register. The headline items:

- Audio is **content-group-held-out**, not source-independent. Only 3 of 400
  files have a confirmed source and licence.
- Vision is a 117-image dataset with a 5-image test split; the checkpoint was not
  retrained.
- No physical signal, microphone array, live camera or road-side trial was
  validated.
- The localisation result is synthetic and does not produce a geographic
  position.
- The hardware layer is a software boundary with safe defaults, not a certified
  fail-safe system.

---

## 6. Release checklist

| Item | Status |
|---|---|
| Full test suite passes | ✅ 443 passed, 1 warning |
| Coverage recorded | ✅ 92% (255 missed of 3259 statements) |
| Legacy 16 tests unchanged | ✅ |
| Shipped model hashes unchanged | ✅ |
| No secrets, device libraries or credentials added | ✅ |
| Localisation isolated from priority path | ✅ source-scan test |
| Hardware package imports no device library | ✅ source-scan test |
| Stored results and docs updated with measured values | ✅ |
| Limitations and non-claims documented | ✅ |
| Roadmap status updated | ✅ |
| Vision retraining and source-independent audio evaluation | ❌ blocked, documented |

---

## 7. Do not claim

- Source-independent audio accuracy.
- Improved YOLO weights.
- Real-time performance.
- Traffic-control benefit.
- Geographic localisation from a single microphone.
- Field localisation accuracy from the synthetic GCC-PHAT experiment.
- Physical traffic-signal or GPIO validation.
- Road-side safety or production readiness.
