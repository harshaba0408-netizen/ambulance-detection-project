# GCC-PHAT Direction Research (Stage 13)

**Status: Experimental — offline research only.**
**Last updated:** 2026-11-12
**Code:** `src/ambulance_system/localisation/`, `tests/test_gcc_phat.py`,
`scripts/benchmark_localisation.py`

This document describes an optional, self-contained direction-of-arrival (DoA)
experiment. It is **not** part of the runtime detection or priority decision. The
project still classifies siren presence from a single audio stream; that
classifier cannot and does not produce a direction or a geographic position.

> **Non-claims.** No microphone array was available. Every number below comes
> from synthetic free-field signals with a known delay. No real siren, room,
> vehicle or intersection was measured. The output is a bearing relative to an
> assumed array geometry, never a GPS position.

---

## 1. Why GCC-PHAT

Two synchronised microphones \(m_0\) and \(m_1\) receive the same far-field
source at slightly different times. If the source is at bearing \(\theta\) and
the sensors are \(d\) metres apart, the time difference of arrival (TDOA) is

\[
\tau = -\frac{d \sin \theta}{c},
\]

with \(\theta\) measured from broadside (\(+y\)) towards \(+x\) and \(c\) the
speed of sound. The minus sign appears because the sensor on the side the source
comes from receives the wave **earlier**.

A plain cross-correlation finds \(\tau\) by aligning the waveforms, but its peak
is broad when the source spectrum is coloured (a siren is strongly harmonic).
GCC-PHAT whitens the cross-spectrum:

\[
R_{01}(f) = \frac{X_0(f) X_1^{*}(f)}
                 {\left|X_0(f) X_1^{*}(f)\right|},
\qquad
r_{01}(\tau) = \mathcal{F}^{-1}\{R_{01}\}(\tau).
\]

The inverse transform keeps only the phase difference, so the peak is sharp and
independent of the source magnitude. The estimate is

\[
\hat{\tau} = \arg\max_{\tau} \left| r_{01}(\tau) \right|.
\]

The implementation zero-pads to the linear correlation length and optionally
oversamples the inverse FFT. A peak is reported with its delay in seconds, its
delay in samples and the relative peak height. The peak height is a sharpness
indicator, **not** a calibrated probability.

### Sign convention

`estimate_delay(reference, moving, fs)` returns a positive delay when `moving`
arrives later than `reference`:

| Situation | Sign |
|---|---|
| `moving` is a delayed copy of `reference` | positive |
| `moving` leads `reference` | negative |
| identical channels | zero |

The two-source helper returns one estimate per local maximum, but a single
GCC-PHAT run still cannot label which peak belongs to which physical source.

---

## 2. From delay to bearing

For a linear pair at \((-d/2, 0)\) and \((+d/2, 0)\), the delay is defined as
\(t(+x) - t(-x)\). Inverting the geometry gives

\[
\theta = \arcsin\left(-\frac{c \tau}{d}\right).
\]

A delay larger than \(d/c\) is physically impossible for that pair. By default
the solver clips it to the endfire direction and sets `clipped=True`; with
`clip=False` it raises `DoAError`. Clipping is a diagnostic, not a correction.

For a two-dimensional array, every pair contributes one linear equation

\[
(p_b - p_a) \cdot u = -c\,\tau_{ab},
\]

where \(u = (\sin\theta, \cos\theta)\) points from the array towards the source.
`estimate_array_bearing` solves the overdetermined system by least squares,
normalises \(u\), and reports the residual in seconds. At least two
non-collinear pairs are required. A single pair or a collinear array cannot
resolve a two-dimensional bearing.

```python
from ambulance_system.localisation import (
    TdoaObservation,
    estimate_array_bearing,
    estimate_delay,
    estimate_pair_bearing,
)

tau = estimate_delay(left_channel, right_channel, sample_rate=16000)
bearing = estimate_pair_bearing(tau.delay_s, sensor_spacing_m=0.12)
print(bearing.bearing_deg, bearing.clipped)

estimate = estimate_array_bearing(
    {"m0": (0.0, 0.0), "m1": (0.2, 0.0), "m2": (0.0, 0.2)},
    [
        TdoaObservation("m0", "m1", tau_01),
        TdoaObservation("m0", "m2", tau_02),
    ],
)
print(estimate.bearing_deg, estimate.residual_rms_s)
```

Nothing in the runtime imports this package. A source-scan test enforces that
the localisation package does not import fusion, the signal controller, the
dashboard or any audio-device library.

---

## 3. Synthetic validation

`scripts/benchmark_localisation.py` generates a white source, applies a known
fractional delay with an FFT phase shift, adds independent Gaussian sensor noise
and runs the estimator. Configuration: 16 kHz, 0.25 s windows, 0.12 m spacing,
16× IFFT oversampling, 7 bearings from −75° to +75°, 10 trials per condition.
The two-source check uses 0.5 s windows, two independent sources and 5 trials per
bearing pair.

Result file:
[`benchmarks/results/localisation_synthetic_20261111T191746Z.json`](../benchmarks/results/localisation_synthetic_20261111T191746Z.json)
(run on 2026-11-12 local time; SHA-256 `93ec41ab86d2cb719e300718e3c622f363c8e339fdee4f4a31f236df1f0dc284`).

| Sensor noise σ | Trials | Delay MAE | Bearing MAE | Within 2° | Clipped |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 70 | 1.15 µs | 0.500° | 100% | 0 |
| 0.01 | 70 | 1.15 µs | 0.499° | 100% | 0 |
| 0.05 | 70 | 1.16 µs | 0.498° | 100% | 0 |
| 0.10 | 70 | 1.16 µs | 0.495° | 100% | 0 |
| **All** | **280** | **1.16 µs** | **0.498°** | **100%** | **0** |

The error floor is set by the 16× delay grid (one native sample is 62.5 µs; the
interpolated grid is 3.9 µs), not by noise. Bearing error grows towards endfire,
where a fixed delay error maps to a larger angle. The largest observed bearing
error was 1.26°.

Two simultaneous sources:

| Bearings | Trials | Both peaks detected | Matched delay MAE |
|---|---:|---:|---:|
| −40°, +35° | 5 | 5/5 | 1.84 µs |
| −60°, +20° | 5 | 5/5 | 2.23 µs |
| +45°, −50° | 5 | 5/5 | 1.94 µs |
| **All** | **15** | **15/15** | **2.00 µs** |

“Both peaks detected” means two reported peaks matched the two known delays
within 0.5 native samples. The tested pairs differed by roughly 4–6 native
samples. It does not mean the peaks were correctly assigned to sources, and the
result does not hold for closely spaced or unequal sources in general.

For the 0.12 m pair, the largest physically representable TDOA is
\(0.12 / 343 = 349.9\) µs (5.60 samples at 16 kHz). That bound is a property of
the geometry, not an accuracy figure.

### Reproduce

```powershell
.\venv\Scripts\python.exe -m pytest -q tests\test_gcc_phat.py
.\venv\Scripts\python.exe scripts\benchmark_localisation.py
```

---

## 4. What this experiment establishes

- The GCC-PHAT implementation recovers known integer and fractional delays under
  the stated synthetic model, including the sign convention.
- The geometry conversion from a pair delay to a far-field bearing is correct for
  a linear pair, and a non-collinear three-sensor array resolves a 2-D bearing.
- Two independent synthetic sources can produce two detectable delay peaks when
  they are well separated.
- The code is deterministic with a seeded generator and has no runtime
  dependency on the decision path.

## 5. Limitations

1. **No real array.** There is no multi-channel recording, no synchronised
   hardware and no measured array response. All results are synthetic.
2. **Free-field model.** Reverberation, walls, diffraction, sensor directivity,
   gain mismatch, clock drift and microphone self-noise are not modelled.
3. **Far-field assumption.** A plane wave is valid only when the source is much
   farther away than the array aperture. A nearby siren violates it.
4. **Linear-pair ambiguity.** A single pair gives a bearing from broadside and
   cannot distinguish front from back or left from right without a known sensor
   order. A collinear array has the same limitation.
5. **Two-source limits.** GCC-PHAT returns peaks, not source identities. Closely
   spaced sources merge; a much louder source can hide a quieter one; moving
   sources smear the peak over the analysis window.
6. **Speed of sound.** The solver uses a fixed 343 m/s. Temperature, humidity
   and wind change the real value and therefore the bearing.
7. **Spatial aliasing.** A large spacing relative to the shortest wavelength
   creates grating lobes. The reported bound \(d/c\) is the unambiguous *delay*
   limit, not a complete array-design analysis.
8. **No geographic position.** Bearings from one array are relative angles. A
   position requires at least two bearing baselines with known geometry, or
   another localisation method, and was not implemented.
9. **No siren link.** The experiment never reads `siren_classifier.pkl` and does
   not claim that a siren was detected, tracked or assigned to a road.
10. **Not wired into priority.** Fusion, the signal state machine and the
    dashboard do not import this package. No priority decision can depend on it.

## 6. What a real deployment would require

- A synchronised multi-channel audio interface and a calibrated array geometry.
- Measured channel delays, gain and frequency response, plus clock-drift
  handling.
- Field recordings with ground-truth bearings across distance, weather,
  reverberation and multiple simultaneous vehicles.
- A tracking layer that associates peaks over time before any fusion decision.
- An explicit safety review before any direction estimate influences a signal
  phase. This experiment is not that system.
