# Video Source Abstraction and Pipeline

Stage 5 replaces the single inline webcam/file loop in `new.py` with a reusable
source abstraction and a two-thread acquisition/processing pipeline.

This is **software simulation only**. Nothing in this stage controls a real
traffic signal, and no live camera or RTSP stream has been validated on this
machine. All measurements below come from the running process, not from model
metadata or estimates.

---

## 1. Components

```
src/ambulance_system/video/source.py     VideoSource protocol + implementations
src/ambulance_system/video/pipeline.py   acquisition/processing workers + stats
src/ambulance_system/monitoring/metrics.py   RateMeter, LatencyTracker, ResourceMonitor
```

### Source types

| Class | Input | Status |
|---|---|---|
| `FileVideoSource` | Recorded video file | Used by `new.py`, tested with injected captures |
| `WebcamVideoSource` | Local camera index | Configurable, not exercised on hardware |
| `RTSPVideoSource` | Network stream URL | **Configurable but unverified** — no authorised test stream |
| `SyntheticVideoSource` | In-memory frames | Test/demo only |

All sources implement:

```python
source.open()      # raise VideoSourceError if the device/stream cannot open
source.read()      # next BGR frame, or None at end-of-stream / read failure
source.close()     # idempotent release
source.name        # log-safe name; RTSP passwords are redacted
source.is_open
source.exhausted   # True only for finite sources that reached the end
```

`create_source(config.video)` builds the source selected by
`video.source_type`. Capture creation is injectable (`capture_factory=`), so the
tests never touch a camera, file codec or network.

### Pipeline

`VideoPipeline` runs two daemon threads:

1. **Acquire** — reads frames, assigns a monotonically increasing index, and
   puts each frame into a bounded queue. If the queue is full it drops the
   frame and increments `frames_dropped`. This keeps preview latency bounded
   when inference is slower than decoding.
2. **Process** — removes frames and calls `on_frame(FramePacket)`.

`FramePacket.sampled` is `True` when `index % frame_skip == 0`. Inference runs
only on sampled frames; preview uses every delivered frame. A callback exception
is logged and counted in `errors`; it does not stop the pipeline.

```
read() ──► bounded queue ──► on_frame(packet)
             │                    │
        full? drop            sampled? run detector
             │
      end-of-stream ──► sentinel ──► processing thread exits
```

## 2. Reconnection

For file sources, `read() == None` with `exhausted == True` means normal end of
stream. For live sources, a `None` frame that is not exhaustion triggers:

```
close() -> wait(reconnect_delay_s) -> open() -> retry
```

`max_reconnect_attempts=None` retries indefinitely (suitable for a camera that
must stay up); an integer bounds the attempts and records `last_error`.
Reconnection has been tested with injected sources only.

**Limitation:** OpenCV's `read()` can block on a dead RTSP connection. The
pipeline stop timeout releases the UI, but the acquisition thread may stay
blocked inside OpenCV until the socket times out. A production deployment would
need an OS-level read timeout or a subprocess boundary.

## 3. Measured metrics

`VideoPipeline.stats` returns a `PipelineStats` snapshot:

| Field | Definition |
|---|---|
| `frames_read` | Frames returned by the source |
| `frames_delivered` | Frames passed to the callback |
| `frames_dropped` | Frames discarded because the queue was full |
| `samples_processed` | Delivered frames marked `sampled` |
| `errors` | Callback exceptions |
| `reconnects` | Reconnect attempts |
| `fps` | `frames_delivered / elapsed` since `start()` |
| `avg_latency_ms`, `p95_latency_ms` | Rolling callback duration (256 samples) |
| `running`, `last_error` | Lifecycle state and last failure |

`LatencyTracker` measures **time spent inside the callback**, not inference
latency in isolation and not end-to-end camera latency. `RateMeter` and
`ResourceMonitor` (CPU / system RAM / process RSS through `psutil`) report only
what the OS and process expose.

## 4. Configuration

```yaml
video:
  source_type: file          # file | webcam | rtsp
  file_path: null            # chosen at runtime for file sources
  webcam_index: 0
  rtsp_url: null             # required when source_type is rtsp
  frame_skip: 5              # infer on every Nth frame
  read_delay_s: 0.01         # pause between reads; 0.0 decodes as fast as possible
  reconnect_delay_s: 2.0
  read_timeout_s: 5.0
  display_width: 320
  display_height: 180
```

`read_delay_s` and `reconnect_delay_s` must be `>= 0`; `frame_skip` must be
`>= 1`.

## 5. Integration with `new.py`

`start_video(idx, road, path)` now:

1. stops any pipeline already attached to that road,
2. builds a `FileVideoSource`,
3. starts a `VideoPipeline` with the configured skip/delay values,
4. converts each delivered frame to a Tk image on the existing display size,
5. runs `detector.detect_best()` only on sampled frames while that road has not
   yet been flagged.

`stop_video(road)` pops the pipeline and calls `stop(timeout=0.5)`, which sets
the stop event, closes the source and joins both workers. Visible behaviour is
unchanged: same preview, same five-frame inference cadence, same result queue.
The difference is that capture is released deterministically and UI blocking is
bounded.

## 6. Usage outside the dashboard

```python
from ambulance_system.video import FileVideoSource, VideoPipeline

source = FileVideoSource("clip.mp4", loop=True)
pipeline = VideoPipeline(source, handle_packet, frame_skip=5, read_delay_s=0.01)
pipeline.start()
...
pipeline.stop()
print(pipeline.stats.to_dict())
```

## 7. Tests

Stage 5 adds 46 tests (178 total, all passing):

| File | Coverage |
|---|---|
| `tests/test_video_source.py` | open/read/close, exhaustion, looping, open failure, webcam/RTSP validation, URL redaction, factory |
| `tests/test_video_pipeline.py` | delivery order, sampling, drops, callback errors, reconnection and limit, start failure, restart, shutdown |
| `tests/test_metrics.py` | rate measurement, latency average/p95/rolling window, resource sampling and fallback |
| `tests/test_config.py` | `read_delay_s` default and validation |

No test opens a real camera or network stream. No performance number in this
document is presented as a real-world FPS claim.

## 8. Measured integration run

One real pipeline run was executed on this machine to confirm that the counters
are populated from live data rather than model metadata:

```powershell
.\venv\Scripts\python.exe scripts\benchmark_video_pipeline.py
```

Settings: `data/Video/1.mp4` (96 frames, 900×474, 23.98 fps nominal),
`frame_skip=5`, `queue_size=2`, `read_delay_s=0.01`, YOLO `imgsz=640` on CPU.

| Measured value | Result |
|---|---:|
| Wall time | 11.36 s |
| Frames read | 96 |
| Frames delivered (preview + sampled) | 21 |
| Frames dropped (queue full) | 75 |
| Sampled frames that reached inference | 4 |
| Delivered FPS (`frames_delivered / wall`) | 1.84 |
| Average callback latency (all deliveries) | 525.3 ms |
| p95 callback latency | 2719.0 ms |
| Errors / reconnects | 0 / 0 |

Interpretation:

* The bounded queue did its job: decoding ran far ahead of CPU inference, so 75
  of 96 frames were dropped and only 4 sampled frames reached YOLO.
* The average mixes fast preview-only callbacks with slow inference callbacks.
  p95 (~2.7 s) is the more representative figure for a sampled frame on this CPU.
  It is not a hardware-independent number and not a claim that the dashboard
  runs in real time on this machine.
* These figures describe one clip on one CPU. They are a reproducibility check,
  not a benchmark of the model or of traffic-control performance.

`scripts/benchmark_video_pipeline.py --no-detection` measures the transport path
without loading YOLO; `--queue-size`, `--frame-skip` and `--clip` allow other
configurations. The script prints JSON and can write it with `--output`.
