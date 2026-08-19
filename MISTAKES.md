# MISTAKES.md

Two things live here: **traps we already decided to avoid** (so nobody spends a
weekend re-discovering them), and a **running log** of what actually went wrong
during the build.

If you catch yourself proposing something in Part 1, stop and read the reason.
If you burn more than an hour on a bug, add it to Part 3.

---

## Part 1 — Rejected approaches

These were considered and rejected with reasons. Re-proposing one is fine, but only
with an argument that engages the reason.

### Measuring rotation from a down-the-line camera

**Tempting because:** X-factor, shoulder turn and hip rotation are the metrics that
sound most like real biomechanics, and every competing app displays them.

**Why it fails:** they rotate in the axis pointing away from the lens. A DTL camera
has essentially no information about that axis. Monocular 3D lift doesn't rescue it —
the golf-specific benchmark puts out-of-the-box models around 145 mm joint error with
depth error dominating, which is far larger than the effect being measured.

**Worse:** the X-factor literature is itself contested. Kwon et al. found the
X-factor parameters were *not* directly related to maximum clubhead velocity, and
that the conventional computation inflates values. So we'd be measuring a disputed
quantity, badly, along the one axis we can't see.

**Instead:** stick to in-plane measurements. If rotation genuinely matters later,
that's a second camera, not a better model.

### Trusting absolute angle values across sessions

**Tempting because:** "your spine angle improved 4° since March" is a great sentence.

**Why it fails:** 2D angles are projections. Where the camera sits relative to the
stance line changes the measured value by several degrees, and real swing-to-swing
variation is the same magnitude. Glazier (2025) makes the general point: markerless
measurement error often exceeds real swing-to-swing variability. A cross-session
trend line built on raw angles plots camera placement.

**Instead:** every metric is a deviation from the address frame *within the same
clip*, where the camera geometry is shared. That comparison is valid; the
cross-session one is not.

### Inventing thresholds and calling faults

**Tempting because:** "Early extension: DETECTED" feels like a real diagnosis, and a
continuum feels wishy-washy.

**Why it fails:** no source gives you a defensible cutoff in normalized image units.
Whatever number you pick is a guess, and a binary verdict hides that it was a guess.
You end up with a machine that confidently tells you you're broken based on a
constant someone typed.

**Instead:** rank deviations by magnitude, show the peak frame, let the eye confirm.
Advisory bands are allowed as labels on a continuum once tuned against ~50 real
swings — never as verdicts before then.

### Letting an LLM write the coaching feedback

**Tempting because:** it reads far better than templates and handles fault
combinations gracefully.

**Why it fails twice:**
1. *Noise laundering.* A chart showing `84.6° ± 4.2°` wears its uncertainty on its
   face. The sentence "your hips are sliding toward the target" reads as established
   fact. Generating prose from an uncertain measurement strips the error bars and
   adds authority — strictly worse than the dashboard we're trying to beat.
2. *It's twelve sentences.* Four faults, four drills. The entire English surface can
   be hand-written in an afternoon. Trading determinism, and the ability to trace a
   sentence back to the number that caused it, for that is a bad deal — especially
   while thresholds are still un-tuned.

**Instead:** templates. Revisit only after the numbers are trustworthy, and then only
as a rephrasing layer that cannot introduce new claims.

### Building the "take 5 swings" app first

**Tempting because:** it's the experience the owner actually described.

**Why it fails:** real-time capture, on-device ML deployment, camera session
management and playback UI are weeks of work that produce zero swing analysis. The
phone can be a dumb camera and the whole loop still works.

**Instead:** Python pipeline. Build the app once the file-shuffling is genuinely the
most annoying part of using the tool.

### Learned event detection (SwingNet / GolfDB) before trying heuristics

**Tempting because:** it's the published approach and gives 8 events instead of 3.

**Why it fails here:** it lands around 76% on all-8-events even on its own training
distribution of pro swings, and when it's wrong you have no idea why. A golf swing is
enormously loud in wrist-speed terms — an order of magnitude above anything else you
do standing in a bay — so a threshold on a curve you can plot handles it.

**Instead:** heuristics first. Revisit only if they measurably fail on real footage.

### Detecting practice swings

**Tempting because:** they pollute the session.

**Why it fails:** practice swings and real swings are identical in landmark space.
Distinguishing them means detecting the ball, which from directly behind the golfer is
a handful of white pixels occluded by their own body.

**Instead:** surface every detected swing, numbered. The owner discards practice
swings by eye. Do not build ball detection for this.

### A cross-session progress graph off five swings

**Tempting because:** progress tracking is satisfying.

**Why it fails:** with n=5 the uncertainty on a dispersion estimate is roughly ±35%.
You cannot distinguish improvement from noise. (Even at n=20 it's ~±16%.)

**Instead:** the tool analyzes *this* session. No trend lines.

---

## Part 2 — Traps waiting in the implementation

Not mistakes yet. Things that will bite if you're not expecting them.

### iPhone slo-mo files lie about their frame rate

The single most likely source of silently wrong results. A 240 fps clip exported from
an iPhone is often already time-stretched for playback, so `cv2.CAP_PROP_FPS` reports
**30**. Every velocity you compute is then 8× too small, segmentation thresholds
don't fire, and nothing obviously looks broken.

`ingest.load_video` must return the *true capture rate*, not the container's playback
rate. Cross-check against clip duration and known swing timing (a downswing is
~0.25 s; if it spans 60 frames you are at 240 fps whatever the header says), and log
the resolved fps loudly.

### `y` increases downward

Image coordinates, not maths coordinates. The lowest point of the hand path is the
**maximum** `y`. Every "is this above or below" comparison is inverted from intuition,
and the bug it causes looks like a plausible-but-wrong answer rather than a crash.

### MediaPipe normalized coordinates are not square

Landmarks come back in `[0, 1]` on both axes, but the frame isn't square. Multiply `x`
by width and `y` by height *separately*, at the pose boundary, before anything
computes a distance or an angle. Skipping this distorts every angle by the aspect
ratio and the result still looks reasonable.

### Never differentiate unsmoothed landmark traces

Raw landmark output jitters frame to frame. Differentiating amplifies exactly that,
so velocity comes out dominated by noise and peak-finding falls apart. Low-pass first:
zero-phase Butterworth (`scipy.signal.filtfilt`, order 4), ~12 Hz for body segments,
~20 Hz for the wrist trace used in segmentation.

Use `filtfilt`, not `lfilter`. `lfilter` introduces phase lag, which shifts every
detected event by a variable amount — a subtle, systematic, hard-to-spot error.

### The nose landmark is frequently invisible from behind

From directly behind the golfer, MediaPipe's face landmarks are unreliable or absent.
Head movement should use the midpoint of the ear landmarks, gated on visibility, with
the nose as fallback — and report `None` rather than a garbage number when neither is
visible.

### MediaPipe is out of distribution here

BlazePose is trained heavily on front-facing poses. A golfer bent at the hips, seen
from behind, arms extended down, heavily self-occluding, is not typical training data.
This is the largest unvalidated assumption in the project and milestone M0 exists to
test it. Do not build on top of pose output before looking at a rendered overlay of a
real swing.

### Impact is not exactly the hand-path low point

It's close, and close is fine for body metrics at 240 fps (one frame is 4.2 ms). Do
not extend this approximation to anything club-related — that's the regime where the
error matters, and it's out of scope anyway.

### Confidence must propagate, not evaporate

A metric computed from landmarks with 0.3 visibility is not the same as one computed
from landmarks at 0.95, and the output must not present them identically. Carry
`confidence` through to the report.

---

## Part 3 — Log

Append real mistakes here as they happen. Newest first. Keep entries short — the
value is the pattern, not the narrative.

```markdown
### YYYY-MM-DD — One-line summary

**Symptom:** what was observed.
**Cause:** what was actually wrong.
**Fix:** what changed.
**Lesson:** the generalizable bit, or "none — one-off."
```

### 2026-08-19 — M0 spike: pose tracking holds up, but the test clip isn't pure DTL

**Symptom:** none — this is a finding, not a bug. Ran MediaPipe Pose Landmarker
(Tasks API, `pose_landmarker_full.task`) over a real swing clip and inspected the
overlay frame by frame.

**Finding 1 — pose tracking is solid.** 100% detection across all 96 frames,
including the high-motion-blur frames right at impact. Hip, shoulder, and head
landmarks averaged ~1.0 visibility throughout. This clears the single largest risk
in the plan (`PLAN.md` §"Risks": "MediaPipe may not track a golfer from directly
behind"). M0 gate passed.

**Finding 2 — the clip isn't actually down-the-line.** The camera shows the golfer's
face and front torso at address and their back at follow-through — it was positioned
off to the side, not braced directly behind on the target line. `PLAN.md` §1 and §4
assume pure DTL geometry (ball-direction axis roughly perpendicular to the camera).
At an angle, early-extension and head-sway measurements pick up a foreshortening
bias. Not a blocker — the self-calibrating `ball_dir` in §4 still gets the sign
right — but the *magnitude* of lateral measurements will be off until capture angle
is standardized or the geometry accounts for bearing. Revisit once there's a
genuinely braced, on-the-line clip to compare against.

**Finding 3 — wrist landmarks are the weak joint.** Left wrist averaged 0.37
visibility, below 0.5 on 58% of frames, versus ~1.0 for hips/shoulders/head. This is
exactly the landmark `segment.py` (§7) depends on most. Confirms the plan's existing
call to average both wrists — but suggests that alone may not be enough; a
hold-last-good-value fallback when both dip low is probably needed too. Revisit
during M2 once real segmentation is running.

**Finding 4 — downswing frame count didn't match the literature figure.** Top→impact
spanned ~13 frames at the header-reported 29.1fps (~0.45s), against the ~0.25s figure
`PLAN.md` cites from the biomechanics literature. Two explanations, not yet
distinguished: genuinely slower swing tempo on this rep, or the fps-lying trap
already documented above (this environment has no `ffprobe`/`exiftool` to check true
capture metadata against the container header). Not urgent — doesn't block M1 — but
`ingest.load_video` should still cross-check header fps against expected swing timing
once real segmentation exists, per the existing guidance above.

**Lesson:** the plan's `MISTAKES.md` entry on fps-lying was written from research,
not measurement — this is the first real clip and it already raised a flag on
exactly that axis. Trust the plan's *shape* (check fps against known timing) even
when the specific mechanism isn't confirmed yet.
