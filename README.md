# swingcoach

A down-the-line golf swing analyzer. Point a phone at yourself from behind, hit some
balls, and get back an HTML report: your swing with a skeleton drawn on it, plus a
short note on what changed between your address position and the swing itself.

It's a Python pipeline, not a product — no competitive positioning, no
monetization, no retention metrics. See `PLAN.md` for the reasoning behind that and
`AGENTS.md` for the hard constraints it's built on. Use it two ways:

- **CLI**, locally, on your own machine — see [Usage](#usage) below.
- **Web**, hosted, no install — see [Web](#web) below. Same pipeline either way.

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

## Web

A hosted version of the same pipeline: upload a clip through a page instead of
running the CLI. See PLAN.md §10 for why this exists and how it's built.

- No account, no sign-up. Upload a `.mov`/`.mp4`, wait (pose estimation takes a
  couple of minutes for a typical clip), get a report link.
- The report link is private (an unguessable id) but not password-protected —
  don't share it anywhere you wouldn't share the video itself.
- Video and report are deleted automatically ~48h after processing. There's no
  account to notify when that happens, so save/download anything you want to keep.
- Same caps apply as anywhere else CPU-bound and public: upload size, clip length,
  and uploads-per-hour are capped (limits are shown on the upload page).

Run it locally the same way you'd run any FastAPI app:

```
pip install -e ".[dev,web]"
uvicorn swingcoach.web.app:app --reload
```

Then open `http://127.0.0.1:8000`.

### Deploy

The included `render.yaml` deploys this as one [Render](https://render.com) web
service — no separate worker or database, see PLAN.md §10 for why one process is
enough here. No third-party API keys or secrets are needed.

1. Push this repo (with `render.yaml` and `Dockerfile` at the root) to GitHub.
2. In the Render dashboard: **New** -> **Blueprint**, point it at the repo.
   Render reads `render.yaml` and provisions the service and its persistent disk.
3. Persistent disks need a paid plan (`render.yaml` requests `starter`, currently
   Render's cheapest plan with disk support) — free-tier services have no disk.
4. Deploy. The pose model (9.4 MB) is fetched during the Docker build, so first
   deploy takes a few minutes longer than a code-only change would.
5. Once live, the service's `.onrender.com` URL (or a custom domain you attach) is
   the link anyone can use.

Everything is configurable via the `SWINGCOACH_*` environment variables in
`swingcoach/web/config.py` (upload caps, rate limit, retention window, worker
concurrency) if the defaults don't fit — set overrides as Render environment
variables, no code change needed.

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
  web/           optional FastAPI front end -- upload page, in-process job
                 queue, same pipeline as the CLI (PLAN.md §10)
tests/
```

`AGENTS.md` has the full module contracts and coding conventions if you're changing
any of this. `MISTAKES.md` is a log of real bugs found against real footage — read it
before touching `segment.py` or `pose.py` in particular; several non-obvious traps
(fps that lies about itself, image-space `y` pointing down, MediaPipe's blind spots
from behind) are already documented there.
