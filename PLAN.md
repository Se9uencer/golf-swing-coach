# Golf Swing Coach — Plan

A personal tool. I set my phone up behind me, hit five balls, and afterwards get my
swing back with a skeleton drawn on it and a short written note about what changed
between my setup position and my swing.

Not a product. One user. No competitive positioning, no monetization, no retention
metrics. The only success criterion is whether I look at the output and learn
something true about my swing.

See `AGENTS.md` for conventions and module contracts, `MISTAKES.md` for the traps
already ruled out and the ones waiting in the implementation.

---

## 1. Down-the-line only

Camera behind me on the target line, at roughly hand height, hitting away from it.
This decides what is measurable and what isn't, and it is not negotiable later
without a second camera.

**Visible from DTL:** spine angle and posture, early extension, head movement, hand
path and arm depth, swing plane and shaft angle (with club tracking).

**Not visible from DTL:** shoulder turn, X-factor, weight shift, hip sway. These
rotate in the axis pointing away from the camera. A monocular depth estimate along
that axis is noise, and anything computed from it is theater. Out of scope
permanently, not deferred — see `MISTAKES.md`.

## 2. Python pipeline on a laptop; the phone is a dumb camera

Record slo-mo with the stock camera app, transfer the clip, run the pipeline. No app
in v1. The interactive "take 5 swings" flow is the most expensive part of the idea
and the least valuable — it is app plumbing, not swing analysis.

The analysis core is I/O-free from the start (contracts in `AGENTS.md`), so porting
to an app later is a UI project rather than a rewrite.

## 3. Capture spec

- **240 fps** slo-mo. The downswing is ~0.25 s: seven frames at 30 fps, sixty at 240.
- Landscape, phone braced on something that will not move mid-session.
- Positioned down the target line at roughly hand height.
- Framed full-body with headroom for the club at the top.
- Fast shutter if the camera app exposes it — motion blur on the hands degrades the
  landmark that drives segmentation.
- Camera does not move during a session.

Rolling shutter is a non-issue here: it distorts fast objects, and the fastest thing
being measured is a pelvis. It would matter for clubface-at-impact, which is out of
scope.

Note that the exported file will probably claim 30 fps. See `MISTAKES.md` — this is
the most likely source of silently wrong output in the whole pipeline.

## 4. Four faults, body only, no club detection

All four come from MediaPipe landmarks alone. No training data, no labeling, no
custom models.

Shared setup, computed once per swing at the address frame:

```
pelvis(t)   = midpoint(LEFT_HIP, RIGHT_HIP)
shoulder(t) = midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
wrist(t)    = midpoint(LEFT_WRIST, RIGHT_WRIST)

scale     = |shoulder(address) - pelvis(address)|  (torso length, in pixels)
ball_dir  = sign(wrist.x[address] - pelvis.x[address])
```

`scale` makes every distance dimensionless, so the numbers mean the same thing at any
camera distance. It's torso length, **not shoulder width** — shoulder width was the
original choice and is wrong for this camera angle: from DTL the two shoulders sit
almost in a line away from the camera, so their 2D separation collapses toward zero
and every distance divided by it blows up (real clip: shoulder width measured 20px at
one address frame vs. 134px at another, same golfer, same fixed camera — see
MISTAKES.md). Torso length stays large and stable across viewing angle instead.
`ball_dir` self-calibrates handedness from the address pose — the hands hang out over
the ball, so the pelvis→wrist direction *is* the ball direction. No configuration
flag, nothing to get wrong when switching sides.

Search windows below say `[top, impact]` because that was the original design intent
— in the current implementation they're `[address, impact]` instead, because
top-of-backswing detection isn't reliable yet (MISTAKES.md). This doesn't corrupt
these particular metrics — pelvis and spine motion during the backswing runs, if
anything, opposite the downswing fault each one is meant to catch — but revisit once
top detection is fixed.

### Early extension — pelvis moving toward the ball

```
ee(t)  = ball_dir * (pelvis.x[t] - pelvis.x[address]) / scale
report = max(ee(t)) for t in [top, impact],  with argmax as peak_frame
```

Positive means toward the ball. Negative means the pelvis stayed back or moved away,
which is fine and reported as such. **Reliability: high.** This is the most common
amateur fault and DTL is the best view of it.

### Loss of posture — spine straightening

```
tilt(t) = angle between (shoulder(t) - pelvis(t)) and image-vertical, in degrees
lop(t)  = tilt(address) - tilt(t)
report  = lop at top, lop at impact, and max|lop| with its frame
```

Positive means standing up out of the address posture. Co-occurs with early extension
but is a distinct measurement and reported separately. **Reliability: high.**

### Head movement — displacement from address

```
head(t) = midpoint(LEFT_EAR, RIGHT_EAR)   if visibility ok
          else NOSE                        if visibility ok
          else None                        -> metric reports None with a note

sway(t) = ball_dir * (head.x[t] - head.x[address]) / scale
lift(t) = (head.y[address] - head.y[t]) / scale     # +ve = up, y is down
report  = peak |sway| and peak |lift| separately, each with its frame
```

Split rather than combined into one magnitude: lifting and swaying are different
faults with different fixes, and a single Euclidean number hides which one happened.
**Reliability: medium** — the ear landmarks are the weak link from directly behind.

### Hand path — trajectory shape

```
trace  = wrist(t) for t in [address, impact], resampled to normalized swing time
report = visual overlay; optionally the normalized offset of the hands at top
```

This is the odd one out: a shape judgment, not a deviation from address, so it does
not inherit the self-referential robustness the other three get. **Reliability: low.**
On probation — likely reduced to a purely visual trace with no number, or cut.

### What is excluded

**Swing plane is deliberately absent.** It is the thing everyone wants from a DTL
camera, and it requires tracking the shaft — a hand-labeled dataset and a custom
detector, since motion blur makes the club the highest-error object even in published
fine-tuned models. That is more work than everything else here combined. It gets
considered only after the rest is built and proven useful.

## 5. Reference is my own address frame

Three of the four faults are intrinsically self-referential: early extension, loss of
posture and head movement are all literally "moved from where I was at address." No
population norms, no pro reference, no external data, and — critically — the
comparison happens within a single clip where the camera geometry is shared, so it
stays valid in a way cross-session angle comparisons do not.

## 6. Rank, don't judge

No pass/fail. No invented thresholds presented as verdicts. The output measures all
four deviations, normalizes them, sorts by magnitude, and reports the largest with
the frame where it peaks. "What I'm doing right" is whatever stayed small.

Advisory bands may be seeded from coaching heuristics and tuned against my own
footage, but they are labels on a continuum — never a machine telling me I'm broken
based on a number I made up. The eyeball stays in the loop.

## 7. Segmentation by wrist velocity

A golf swing is loud: hand speed during one is roughly an order of magnitude above
anything else done standing in a bay. Wrist-landmark speed over time gives a clean
spike train, one spike per swing.

```
smoothed = lowpass(wrist_trace, fps, cutoff_hz=20)
speed(t) = |d/dt smoothed(t)|

swing peaks:  find_peaks(speed, prominence = k * median(speed),
                                distance   = 2.0 * fps)

per peak:
  impact  = argmax(wrist.y) in a short window around the peak   # y is down
  address = end of the last sustained low-motion window (>= 0.3 s) before the peak
  top     = argmin(wrist.y) between address and the peak         # highest hand point
  start   = address - 0.2 s,  end = impact + 0.5 s   (for rendering)
```

Address is found by searching backward from the peak directly, not through an
intermediate top guess — a golfer who waggles or takes a forward press can have more
than one qualifying quiet window, and only the one closest to the peak is the real
address (MISTAKES.md, "top-of-backswing detection fixed"). Top then falls out as the
highest point the hands reach in that window — the same logic as impact, mirrored,
and needs no threshold at all.

Pure numpy and scipy over the landmark traces. Inspectable and plottable, which
matters more than accuracy here: a wrong threshold is a five-second fix on a plot, a
wrong neural-net prediction is not.

**Practice swings are indistinguishable from real ones in landmark space and this is
not being solved.** Every detected swing is surfaced and numbered; I discard the
practice ones by eye.

## 8. Hand-written templates for all feedback text

One template per fault, with slots for the measured value, the peak frame, and a
single curated drill. Deterministic, no dependencies, impossible to hallucinate, and
every sentence traces to the number that produced it.

No LLM in v1. The entire English surface is about a dozen sentences and four drills.
For four known faults, four *curated* drills beat generated ones.

## 9. Annotated MP4 wrapped in a self-contained HTML report

The overlay's core idea falls out of §5: **draw the address position ghosted onto
every live frame.** Translucent skeleton and reference lines frozen at address, solid
ones tracking live. Early extension stops being a number and becomes a visible gap
between the ghost hips and the real hips — the visual argument *is* the measurement.

Burned into the video:

- live skeleton
- ghosted address skeleton
- vertical butt line at the hip line at address — early extension is crossing it
- spine-angle line, address vs. live
- head marker at address position
- hand-path trace accumulating through the swing
- swing number and current phase label

The HTML report embeds that video, plus per swing: the three key frames as stills,
the ranked deviations table with confidences, the templated feedback, and the
wrist-velocity plot with detected events marked. One self-contained file, opens in a
browser, readable on a phone in the car after the range.

The velocity plot is not decoration — it is how a bad segmentation gets diagnosed in
five seconds instead of five minutes. Keep it working.

---

## Out of scope

- 3D pose lift. Monocular models sit around 145 mm MPJPE on golf, unusable for
  kinematics; fixing it needs proprietary mocap data.
- Shoulder turn, X-factor, weight shift, hip sway — not visible from DTL (§1).
- Club and clubface tracking, swing plane (§4).
- Cross-session trend lines. Five swings cannot support a trend claim, and raw angle
  values are contaminated by camera placement anyway.
- Practice-swing detection (§7).
- Any mobile app, server, or cloud component.

---

## Risks and unvalidated assumptions

**MediaPipe may not track a golfer well from directly behind.** BlazePose is trained
heavily on front-facing poses; a golfer bent at address, arms extended down,
substantially self-occluding, is out of distribution. This is the single largest
assumption in the plan. M0 exists to test it and gates everything else.

**Head landmarks are the weak link.** From directly behind, the nose is often not
visible at all. Ear-midpoint with visibility gating, nose as fallback, `None` when
neither is usable — and be prepared to drop the metric.

**Landmark traces are jittery** and must be low-pass filtered before differentiation,
or velocity is dominated by noise and segmentation fails outright.

**Hand path is the softest metric** and does not inherit the robustness of the other
three. Likely cut.

**Thresholds are guesses until tuned** against ~50 real swings. Until then the output
ranks and shows; it does not diagnose.

---

## Build order

Each milestone has an acceptance test that involves looking at real footage. A green
unit test on a synthetic array does not count.

### M0 — Does pose work at all? *(gates everything)*

Film five swings DTL. Run MediaPipe. Dump a skeleton-only overlay video. Watch it.

**Accept when:** hips, shoulders and wrists track plausibly from address through
impact on all five swings, without the skeleton collapsing or limb-swapping during
the downswing.

**If it fails:** stop. Options are a different pose model, a less extreme camera
angle, or a rethink. Do not build on top of bad landmarks.

### M1 — Ingest, pose, smoothing, skeleton overlay

`ingest` + `pose` + `smooth`, and `render/overlay.py` drawing the live skeleton.

**Accept when:** true fps is correctly resolved on a real iPhone slo-mo file (verified
against clip duration, not the header), and the overlay renders smoothly with no
visible landmark jitter.

### M2 — Segmentation and event detection

`segment.py`, plus the velocity plot in the report.

**Accept when:** a five-swing clip splits into exactly five swings, and the detected
address / top / impact frames survive eyeball inspection on every one.

### M3 — Metrics

`metrics.py`: the four measurements, torso-length normalization, ranking,
confidence propagation.

**Accept when:** numbers come out per swing, `None` appears where landmarks were
unusable rather than a fabricated value, and the ranking matches what the overlay
shows.

### M4 — The actual thing

Ghosted-address overlay, `feedback.py` templates, `render/report.py`.

**Accept when:** one command turns a range-session clip into a single HTML file that
is worth reading in the car afterwards.

### M5 — Tune

Fifty swings. Calibrate the advisory bands. Cut what doesn't work — hand path first
in line.

Only after M5: consider a shaft detector, or an app.

---

## Open items

- **Minimum swings per session** for the output to be worth reading. Five is what I
  want; it is not enough for any statistical claim, which is fine, because there are
  no statistical claims.
- **Head-movement fallback landmark**, pending what M0 shows.
- **Where sample clips live.** Too large for git; keep out of the repo and reference
  by path.
- **Rename "address" to "setup" in report-facing text.** "Address" is golf jargon
  that a first-time reader doesn't recognize (flagged directly by a real recipient).
  Code identifiers, `Swing.address`, and internal docs can keep "address" — it's an
  established term in the codebase and in golf instruction. Only the HTML report's
  labels (frame captions, table headers) need the friendlier word.
