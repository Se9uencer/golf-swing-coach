# Golf Swing Coach — Plan

A personal tool. I set my phone up behind me, hit five balls, and afterwards get my
swing back with a skeleton drawn on it and a short written note about what changed
between my setup position and my swing.

Not a product. One user. No competitive positioning, no monetization, no retention
metrics. The only success criterion is whether I look at the output and learn
something true about my swing.

---

## Locked decisions

### 1. Down-the-line only

Camera behind me on the target line, at roughly hand height, hitting away from it.
This decides what is measurable and what isn't, and it is not negotiable later
without a second camera.

**Visible from DTL:** spine angle and posture, early extension, head movement,
hand path and arm depth, swing plane and shaft angle (with club tracking).

**Not visible from DTL:** shoulder turn, X-factor, weight shift, hip sway. These
rotate in the axis pointing away from the camera. A monocular depth estimate along
that axis is noise, and anything computed from it is theater. They are out of scope
permanently, not deferred.

### 2. Python pipeline on a laptop; the phone is a dumb camera

Record slo-mo with the stock camera app, transfer the clip, run the pipeline. No
app in v1. The interactive "take 5 swings" flow is the most expensive part of the
idea and the least valuable — it is app plumbing, not swing analysis.

The analysis core is structured as a clean, I/O-free module from the start, so that
porting to an app later is a UI project rather than a rewrite.

### 3. Capture spec

- **240 fps** slo-mo. The downswing is ~0.25 s: seven frames at 30 fps, sixty at 240.
- Landscape, phone braced on something that will not move mid-session.
- Framed full-body with headroom for the club at the top.
- Fast shutter if the camera app exposes it.
- Camera does not move during a session.

Rolling shutter is a non-issue here: it distorts fast objects, and the fastest thing
being measured is a pelvis. It would matter for clubface-at-impact, which is out of
scope.

### 4. Four faults, body only, no club detection

| Fault | Measurement | Reliability |
|---|---|---|
| Early extension | Horizontal pelvis displacement toward the ball, address → impact | High |
| Loss of posture | Spine angle (pelvis→shoulder midpoint vs. image vertical), delta from address | High |
| Head movement | Head marker displacement from address, peak over swing | Medium — see risks |
| Hand path | Wrist-midpoint trajectory shape through the backswing | Low — most likely to be cut |

All four come from MediaPipe landmarks alone. No training data, no labeling, no
custom models.

**Swing plane is deliberately excluded.** It is the thing everyone wants from a DTL
camera and it requires tracking the shaft, which means a hand-labeled dataset and a
custom detector — motion blur makes the club the highest-error object even in
published fine-tuned models. That is most of the total effort of the project. It
gets considered only after everything else works.

### 5. Reference is my own address frame

Three of the four faults are intrinsically self-referential: early extension, loss
of posture, and head movement are all literally "moved from where I was at address."
No population norms, no pro reference, no external data.

All distances are normalized by **shoulder width in pixels at address**, so results
don't change when the phone sits six feet back instead of eight.

### 6. Rank, don't judge

No pass/fail. No invented thresholds presented as verdicts. The output measures all
four deviations, normalizes them, sorts by magnitude, and reports the largest with
the frame where it peaks. "What I'm doing right" is whatever stayed small.

Rough advisory bands may be seeded from coaching heuristics, but they are labels on
a continuum, tuned against my own footage — never a machine telling me I'm broken
based on a number I made up. The eyeball stays in the loop.

### 7. Segmentation by wrist velocity, all swings surfaced

A golf swing is loud: hand speed during one is roughly an order of magnitude above
anything else done standing in a bay. Wrist-landmark speed over time gives a clean
spike train.

- **Address** — last sustained low-motion window before the spike
- **Top** — velocity minimum between the backswing and downswing peaks
- **Impact** — hand-path low point, coincident with peak hand speed

Pure numpy over the landmark traces. Inspectable and plottable, which matters far
more than accuracy here, because a wrong threshold is a five-second fix and a wrong
neural-net prediction is not.

**Practice swings are indistinguishable from real ones in landmark space and this is
not being solved.** Every detected swing is surfaced and numbered; I discard the
practice ones by eye.

### 8. Hand-written templates for all feedback text

One template per fault, with slots for the measured value, the peak frame, and a
single curated drill. Deterministic, no dependencies, impossible to hallucinate, and
every sentence traces to the number that produced it.

No LLM in v1. The entire English surface is about a dozen sentences and four drills
— writing them by hand takes an afternoon, and adding an API dependency and
nondeterminism while thresholds are still un-tuned is a bad trade. A phrasing layer
can be added later once the numbers are trustworthy.

For four known faults, four *curated* drills beat generated ones.

### 9. Annotated MP4 wrapped in a self-contained HTML report

The overlay's core idea falls out of decision 5: **draw the address position ghosted
onto every live frame.** Translucent skeleton and reference lines frozen at address,
solid ones tracking live. Early extension stops being a number and becomes a visible
gap between the ghost hips and the real hips.

Burned into the video:
- live skeleton
- ghosted address skeleton
- vertical "butt line" at the hip line at address — early extension is crossing it
- spine-angle line, address vs. live
- head marker at address position
- hand-path trace accumulating through the swing

The HTML report embeds that video, plus the three key frames per swing as stills,
the ranked deviations table, and the wrist-velocity plot with detected events marked.
One file, opens in a browser, readable on a phone in the car after the range.

---

## Explicitly out of scope

- 3D pose lift. Out-of-the-box monocular models sit around 145 mm MPJPE on golf,
  which is unusable for kinematics. Fixing that needs proprietary mocap data.
- Shoulder turn, X-factor, weight shift, hip sway — not visible from DTL (§1).
- Club and clubface tracking, swing plane (§4).
- Cross-session trend lines. Five swings cannot support a trend claim, and raw angle
  values are contaminated by camera placement anyway.
- Any mobile app.

---

## Risks and unvalidated assumptions

**MediaPipe may not track a golfer well from directly behind.** BlazePose is trained
heavily on front-facing poses. A golfer viewed from behind — bent at address, arms
extended down, substantial self-occlusion — is out of distribution. This is the
single largest assumption in the plan and it is tested first, before anything else
is built.

**Head landmarks are the weak link.** From directly behind, the nose landmark is
often not visible at all. Fall back to the midpoint of the ear landmarks, gate on
the visibility score, and be prepared to drop the head-movement metric.

**Landmark traces are jittery** and need low-pass filtering before differentiation —
otherwise velocity is dominated by noise and segmentation fails. At 240 fps there is
plenty of headroom to filter aggressively.

**Hand path is the softest of the four metrics.** It is a shape judgment rather than
a deviation from address, so it doesn't inherit the self-referential robustness the
other three get. Likely to be cut or reduced to a purely visual trace with no number.

**Thresholds are guesses until tuned** against ~50 of my own swings. Until then the
output ranks and shows; it does not diagnose.

---

## Build order

**M0 — Does pose work at all?** Film five swings DTL, run MediaPipe, dump a
skeleton-only overlay video, watch it. If the landmarks are garbage through the
downswing, everything above needs rethinking. *This gates the entire project.*

**M1 — Ingest, pose, smoothing, skeleton overlay.** Deliverable: my swing back with
a skeleton on it.

**M2 — Segmentation and event detection.** Wrist velocity, swing splitting, address
/ top / impact, plus the diagnostic velocity plot. Deliverable: correctly splits five
swings and marks the three events.

**M3 — Metrics.** The four measurements, shoulder-width normalization, ranking.
Deliverable: numbers per swing.

**M4 — The actual thing.** Ghosted-address overlay, templated feedback, HTML report.
Deliverable: what I asked for.

**M5 — Tune.** Fifty swings, calibrate the advisory bands, cut what doesn't work.

Only after M5: consider a shaft detector, or an app.

---

## Proposed layout

```
swingcoach/
  ingest.py      video -> frames, fps
  pose.py        frames -> landmark array (n_frames, 33, 3) + visibility
  smooth.py      low-pass filtering of landmark traces
  segment.py     swing splitting, address/top/impact detection
  metrics.py     the four measurements, normalization, ranking
  feedback.py    templates and drills
  render/
    overlay.py   annotated MP4
    report.py    self-contained HTML
  cli.py
tests/
```

`swingcoach/` stays free of I/O and presentation concerns so the app port stays a
UI project.

---

## Open items

- **Handedness** is a config flag — it sets the sign convention for "toward the
  ball." Defaulting to right-handed.
- **Minimum swings per session** for the output to be worth reading. Five is what I
  want; it is not enough for any statistical claim, which is fine because there are
  no statistical claims.
- **Head-movement fallback** landmark choice, pending what M0 shows.
