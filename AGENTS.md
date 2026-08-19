# AGENTS.md

Instructions for AI agents working in this repository.

Read this file, then `PLAN.md` (what we're building and why), then `MISTAKES.md`
(what not to do, and what has already gone wrong). All three are short on purpose.

---

## What this is

A personal tool, for one user. Phone goes on the ground behind the golfer, they hit
five balls, and afterwards they get the swing back with a skeleton drawn on it plus
a short written note about what changed between their setup position and their swing.

It is not a product. There is no user base, no monetization, no retention target. The
only success criterion is whether the owner looks at the output and learns something
true about their swing. Optimize for **inspectability and honesty**, not features.

---

## Hard constraints

These are settled decisions, each made for a documented reason in `PLAN.md`. Do not
quietly reverse them. If you think one is wrong, say so and stop — don't implement
around it.

1. **Down-the-line camera only.** Never add metrics that require depth along the
   camera axis: shoulder turn, X-factor, weight shift, hip sway, pelvis rotation.
   They are not measurable from this view and a monocular depth estimate of them is
   noise. This is permanent, not deferred.
2. **No 3D pose lift.** Out-of-the-box monocular 3D models sit around 145 mm MPJPE on
   golf footage. Do not use MediaPipe's `z` coordinate for anything.
3. **No club or shaft tracking** without an explicit decision to take it on. It needs
   a hand-labeled dataset and is larger than the rest of the project combined. That
   means **no swing plane metric**, however much it seems like the obvious next thing.
4. **No invented thresholds presented as verdicts.** Output ranks deviations by
   magnitude and shows the frame. It does not emit pass/fail. See "Rank, don't judge"
   in `PLAN.md`.
5. **No LLM in the feedback path.** Feedback text comes from hand-written templates.
   A model writing coaching prose over noisy measurements produces confident,
   fluent, unfalsifiable wrong answers.
6. **No app, no server, no cloud.** Local Python pipeline, files in and files out.
7. **Never fabricate a measurement to fill a slot.** If a landmark's visibility is
   too low or an event wasn't detected, the metric is `None` and the report says so.
   A missing number is fine; a made-up one poisons the whole tool.

---

## Vocabulary

Use these terms exactly; they map to identifiers in the code.

| Term | Meaning |
|---|---|
| **DTL** | Down-the-line. The camera view: behind the golfer, on the target line, hitting away from the camera. |
| **address** | The still setup position immediately before the swing starts. The reference datum for every metric. |
| **top** | Transition. The moment the backswing ends and the downswing begins. |
| **impact** | Club meets ball. Approximated by the low point of the hand path. |
| **early extension** | Pelvis moving *toward the ball* during the downswing. The most common amateur fault, and the one DTL shows best. |
| **loss of posture** | Spine angle straightening away from its address value. Co-occurs with early extension but measured separately. |
| **butt line** | A vertical line dropped at the golfer's hip line at address. Crossing it is early extension made visible. |
| **deviation** | A measured difference from the address frame, normalized by shoulder width. The unit of output. |
| **scale** | Shoulder width in pixels at address. Every distance is divided by it so camera distance doesn't matter. |
| **ball direction** | Signed image-x direction from pelvis toward the hands at address. Self-calibrating; do not hardcode by handedness. |

---

## Conventions

**Coordinates.** Image space, OpenCV convention: origin top-left, `x` right,
**`y` increases downward**. Convert MediaPipe's normalized `[0,1]` output to pixels
immediately at the pose boundary, multiplying `x` by width and `y` by height
separately — normalized coordinates are not square and treating them as such
silently distorts every angle.

**The lowest physical point is the maximum `y`.** This catches people constantly.

**Units.** Distances are in shoulder-widths (dimensionless), never pixels, once they
leave `metrics.py`. Angles in degrees. Time in seconds; frame indices stay `int`.

**Handedness** is inferred from the address pose, not configured. See "ball
direction" above.

---

## Architecture

```
swingcoach/
  ingest.py      video file -> frames + true fps
  pose.py        frames -> landmark array + visibility
  smooth.py      zero-phase low-pass filtering of landmark traces
  segment.py     swing splitting; address / top / impact detection
  metrics.py     the four measurements, normalization, ranking
  feedback.py    templates and drills
  render/
    overlay.py       annotated MP4
    velocity_plot.py wrist-speed PNG w/ detected events -- the segmentation debug view
    report.py        self-contained HTML (wraps overlay + velocity_plot + metrics)
  cli.py         the only place that parses args or touches argv
tests/
```

`swingcoach/` core modules (`pose`, `smooth`, `segment`, `metrics`, `feedback`) are
**pure**: array in, array or dataclass out. No file reads, no writes, no printing, no
global state. All I/O lives in `ingest`, `render/`, and `cli`. This is what keeps a
future app port a UI project rather than a rewrite — do not erode it for convenience.

### Module contracts

```python
# ingest.py
def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    """Frames and TRUE capture fps. See MISTAKES.md on slo-mo fps."""


# pose.py
@dataclass(frozen=True)
class Landmarks:
    xy: np.ndarray  # (n_frames, 33, 2) float32, pixels
    visibility: np.ndarray  # (n_frames, 33) float32, 0..1


# smooth.py
def lowpass(traces: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth (filtfilt). Never lfilter — it introduces lag."""


# segment.py
@dataclass(frozen=True)
class Swing:
    start: int
    address: int
    top: int
    impact: int
    end: int


def find_swings(lm: Landmarks, fps: float) -> list[Swing]: ...


# metrics.py
@dataclass(frozen=True)
class Deviation:
    name: str
    value: float | None  # None when not measurable
    unit: str  # "shoulder-widths" | "degrees"
    peak_frame: int | None
    confidence: float  # driven by landmark visibility
    note: str = ""  # why it's None, when it is


def measure(lm: Landmarks, swing: Swing, fps: float) -> list[Deviation]:
    """Ranked by |value| descending. Unmeasurable deviations sort last."""


# feedback.py
def describe(devs: list[Deviation]) -> SwingFeedback:
    """Pure template substitution. No branching cleverness, no generation."""
```

---

## Tooling

Python 3.11+. `mediapipe`, `opencv-python`, `numpy`, `scipy`, `jinja2`. `pytest` for
tests, `ruff` for lint and format.

```
ruff format . && ruff check . && pytest -q
```

Run that before claiming anything works.

---

## How to work here

**Validate against real footage, not synthetic data.** This is a CV project — a
passing unit test on a hand-made array tells you almost nothing about whether the
pipeline works on an actual swing. Every metric change needs a look at the rendered
overlay on a real clip. Keep a couple of sample clips out of git (they're large) and
reference them by path.

**When a number looks wrong, plot it before theorizing.** The wrist-velocity trace
with detected events marked answers most segmentation questions in five seconds.
`render/report.py` includes that plot for exactly this reason — keep it working.

**Prefer a threshold you can see on a plot over a model you can't.** This is a
one-user tool; debuggability beats accuracy every time.

**Uncertainty is reportable.** Confidence fields and `None` values exist so the tool
can say "I couldn't see your head on swing 3." Use them rather than degrading
silently.

**Don't grow the fault list.** Four faults: early extension, loss of posture, head
movement, hand path. Adding a fifth is a decision for the owner, not a refactor.
Hand path is on probation and may be cut — see `PLAN.md`.

**Keep the plan honest.** If a decision in `PLAN.md` turns out to be wrong once real
footage exists, update the file in the same change that acts on it. If you hit a trap
worth remembering, append to `MISTAKES.md`.
