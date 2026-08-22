# swingcoach

A personal down-the-line golf swing analyzer. Point a phone at yourself from behind,
hit some balls, and get back an HTML report: your swing with a skeleton drawn on it,
plus a short note on what changed between your address position and the swing itself.

It's a local Python pipeline for one user, not a product. See `PLAN.md` for the
reasoning behind that and `AGENTS.md` for the hard constraints it's built on.

## Setup

Requires Python 3.11+.

```
pip install -e ".[dev]"
```

The pose model isn't committed (9.4 MB binary). Fetch it once:

```
curl -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Capture

- Camera **behind you, on the target line**, at roughly hand height. This is the only
  supported angle — see `PLAN.md` §1 for why.
- 240 fps slo-mo if your phone supports it, landscape, braced so it doesn't move
  mid-session, framed full-body with headroom for the club at the top.
- Hit as many balls as you like in one clip; the pipeline splits it into individual
  swings on its own.

## Usage

```
python -m swingcoach.cli report swing.mov report.html
```

That's the actual thing: one self-contained HTML file with the annotated video,
address/top/impact stills, and your measurements (early extension, loss of posture,
head sway, head lift) ranked by size, largest first. No pass/fail — see `AGENTS.md`
on "no invented thresholds." A fifth fault, hand path, is planned but not yet built —
see `metrics.py`.

Two more commands help while debugging a clip:

```
python -m swingcoach.cli segment swing.mov --plot velocity.png
python -m swingcoach.cli overlay swing.mov overlay.mp4
```

`segment` prints the detected swings and events without rendering anything;
`--plot` writes the wrist-speed trace that segmentation is based on, useful for
seeing why a swing was or wasn't found. `overlay` renders just the skeleton video,
no metrics. All three accept `--cache pose.npz` to skip re-running pose estimation
(the slow step, ~10fps on CPU) on repeat runs against the same clip.

## Development

```
ruff format . && ruff check . && pytest -q
```

Run that before claiming anything works. Then read `AGENTS.md`'s "How to work here" —
in short: validate against real footage, not just synthetic tests, and plot a signal
before theorizing about why it looks wrong.

## Project layout

```
swingcoach/
  ingest.py      video file -> frames + true fps
  pose.py        frames -> landmark array + visibility
  smooth.py      zero-phase low-pass filtering of landmark traces
  segment.py     swing splitting; address / top / impact detection
  metrics.py     the four measurements, normalization, ranking
  feedback.py    templates and drills
  render/        overlay video, velocity plot, HTML report, PDF export
  cli.py         the only place that parses args
tests/
```

`AGENTS.md` has the full module contracts and coding conventions if you're changing
any of this. `MISTAKES.md` is a log of real bugs found against real footage — read it
before touching `segment.py` or `pose.py` in particular; several non-obvious traps
(fps that lies about itself, image-space `y` pointing down, MediaPipe's blind spots
from behind) are already documented there.
