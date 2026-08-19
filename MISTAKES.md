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

### 2026-08-19 — M2 first run against real multi-swing footage: three bugs found and fixed, one left open

Ran `segment.py` against a real 30fps, 34.6s, ~6-swing range session (the first
multi-swing clip available). `PLAN.md` section 7's algorithm, implemented literally,
found *something* on the first try but got the events wrong in ways that only real
footage exposed. In order of discovery:

**Bug 1 — the 20Hz wrist cutoff violates Nyquist below 50fps.** `PLAN.md`'s cutoff
assumed true 240fps capture (Nyquist 120Hz). Both real clips available so far are
~30fps (Nyquist 15Hz) — the capture spec (240fps slo-mo) hasn't actually been
followed by either test clip. `smooth.lowpass` correctly raised rather than doing
something undefined with an invalid cutoff. **Fix:** `_wrist_cutoff_hz(fps)` scales
the cutoff to `min(20Hz, 0.45 * Nyquist)` instead of hardcoding 20. Verified: no
longer crashes on either real clip.

**Bug 2 (the big one) — "quiet" was calibrated per-swing instead of per-clip.**
Original: `quiet_threshold = max(0.2 * peak_speed, median_speed)`. On the real clip
this evaluated to ~577-580px/s — but the clip's actual stillness floor (25th
percentile of the whole session's speed) was ~60px/s. A threshold 10x too loose
grabbed noise dips *inside* the downswing itself as "the top of the backswing":
address and top collapsed to 4 frames apart (0.13s) instead of the ~9 frames a real
transition takes at this fps. **Fix:** `quiet_threshold = percentile(speed_whole_clip,
25) * 1.5`, computed once from the whole clip's resting-speed distribution instead of
a fraction of each swing's own peak. Verified by hand against the raw per-frame speed
trace before and after (documented address→top gaps of 4 frames became 9 frames,
matching what the frame stills actually show).

**Bug 3 — wrist midpoint was an unweighted average of two confidences.** Real clip
had a stretch (frames ~980-1010) where the camera was pointed at a golf-simulator
screen, not the golfer — MediaPipe's visibility on both wrists sat at ~0.1 there, yet
`wrist_midpoint` averaged the (meaningless) positions anyway, and segmentation
confidently reported a swing from it. Separately, in a real swing, right-wrist
visibility fell from 0.69 to 0.39 across 10 frames during follow-through (natural
occlusion), producing a spurious ~2900px/s "speed" spike from the unweighted average
that outcompeted the real swing's own, smaller peak and got it suppressed by the
`distance` parameter. **Fix, two parts:**
1. `wrist_midpoint` is now visibility-weighted (`(L*visL + R*visR) / (visL+visR)`),
   NaN below `MIN_WRIST_VISIBILITY=0.4` combined, rather than a plain average.
2. That alone wasn't enough: `smooth.lowpass` linearly interpolates across the NaN
   gap to keep *positions* continuous (correct, documented behavior), but filtfilt
   then *rings* at the boundary where the flat interpolated stretch meets real data
   again — producing a fabricated dip that read as genuine stillness and a
   fabricated 9600px/s spike that read as a genuine swing, both landing inside a
   region the wrists were never actually seen. Interpolating a *position* for
   continuity is one thing; treating its *derivative* as observed motion is another.
   Added `wrist_trustworthy(lm)`, a boolean mask of frames where the wrists were
   actually tracked, and `find_swings` now rejects any candidate whose address, top,
   impact, or peak frame lands outside it. Verified: the simulator-screen false
   positive is gone; the two real swings elsewhere in the clip are unaffected.

**Open issue — NOT fixed, needs a different algorithm, not a threshold tweak.**
Even after all three fixes, 1 of 3 remaining detected swings has an incorrect `top`.
Confirmed by hand on two separate real swings:
- One golfer waggles/rehearses before actually swinging, creating a *second*
  legitimate stillness period (frames 648-654: speed ~20-100px/s) between address
  and the real backswing (which doesn't start until ~658). `_find_top`'s "last local
  minimum below the quiet threshold, searched backward up to 1.5s" logic has no way
  to distinguish "settled again after a waggle" from "the actual top of backswing" --
  it grabbed the deeper, more obvious dip at 653, which the frame stills confirm is
  still an address-like pose, not arms-raised-at-the-top. The real top is somewhere
  in the actually-elevated stretch closer to impact that the current logic never
  considers, because it stops at the first/last qualifying threshold-crossing rather
  than reasoning about the swing's structure.
- A related but distinct failure appeared on a different swing in the same clip,
  where a small, low-confidence peak next to a much larger one got discarded by the
  `distance` parameter in `find_peaks`, when the smaller peak was arguably the real
  swing and the larger one coincided suspiciously with degrading wrist visibility.

**Likely fix, not yet implemented:** stop searching backward from the peak for *a*
qualifying minimum. Work forward from a confirmed address instead, and find "top" as
the point where wrist velocity *direction* reverses (backswing and downswing go
opposite ways), not just where its *magnitude* dips below a threshold. Direction
reversal is a structural property of the swing itself and shouldn't be fooled by a
waggle, which doesn't reverse in the same way. This is a real algorithm change, not
a constant to retune, so it's flagged here rather than silently patched.

**Lesson:** three real, verifiable bugs surfaced from a single clip that a synthetic
test never would have caught -- every one of them was invisible until frame stills
and raw per-frame values were checked by hand, exactly per `AGENTS.md`'s "validate
against real footage" and "plot it before theorizing" rules. The remaining open
issue is a reminder that a threshold fix which resolves the *symptom* on the clip in
hand can still leave the underlying algorithm wrong in a way the next clip exposes
differently -- don't declare a milestone's acceptance criteria met on partial
evidence just because the loudest bugs are gone.

### 2026-08-19 — M3 first run: `scale` used shoulder width, which collapses toward zero from a DTL angle

Built `metrics.py` (early extension, loss of posture, head sway/lift) and ran it
against the three swings M2 had already found in the real multi-swing clip. Two of
the three swings produced physically absurd numbers: head movement of 1.8-3.2
*torso-lengths* (the head would have to travel several body-lengths sideways), pelvis
shifts of 1.4-2.7 shoulder-widths. Real faults are a fraction of one body-length, not
multiples.

**Root cause: `scale` was defined as shoulder-to-shoulder width at address**
(`PLAN.md` section 4's original formula, carried into `AGENTS.md`'s vocabulary table
without question). That's the standard normalization in face-on pose-fitness apps,
where shoulder width is a large, stable reference. It is exactly the wrong axis for a
DTL camera: viewed from the side, a golfer's two shoulders sit almost in a line away
from the camera, so their 2D separation is a foreshortened projection that collapses
toward zero and is highly sensitive to small variations in exact body orientation.
Confirmed directly: drawing the raw shoulder landmarks on the source frame showed
MediaPipe correctly placing them (visibility 1.0, not a tracking error) almost on top
of each other -- shoulder width measured 20px and 35px at two real address frames,
versus 134px at a third, same golfer, same fixed camera, the only difference being
which way the golfer happened to be oriented relative to it. Dividing real pixel
displacements by a ~20-35px denominator instead of the true ~130px inflated every
distance-based metric by 4-6x.

**Fix:** `scale` is now torso length -- the pelvis-to-shoulder-midpoint distance at
address -- instead of shoulder width. Torso length is dominated by standing height,
which barely foreshortens under left-right camera rotation the way shoulder width
does, so it stays large and stable regardless of the golfer's exact orientation to a
fixed DTL camera. Updated everywhere the old definition was documented: `PLAN.md`
section 4's formula block, and `AGENTS.md`'s vocabulary table and unit list (the unit
string changed from `"shoulder-widths"` to `"torso-lengths"` throughout).
Re-verified against the same real clip: the two swings already trusted from M2's
validation now produce sane numbers (early extension 0.23-0.28 torso-lengths, posture
change 16.6-26 degrees); the one swing M2 had already flagged as suspect (its
address/impact don't correspond to a real swing sequence) still produces inflated
numbers, which is expected and consistent -- a metric is only as trustworthy as the
event indices it's fed, and this wasn't a new failure, it was the same known-bad
input surfacing through a different module.

**Second, smaller finding from the same session: ranking compared raw values across
different units.** With `scale` fixed, a real ~40px pelvis shift (1.0 shoulder-widths
under the old, buggy `scale`) still lost a magnitude-ranking comparison to the
~22-degree spine-tilt change it incidentally caused, purely because "degrees" produces
numerically bigger values than a normalized-distance unit for physically comparable
deviations. `PLAN.md` section 6 says deviations should be "normalized" before ranking
but the implementation had only normalized *within* a unit (by `scale`), not *across*
units. Fixed with a per-unit rank-scale divisor (`_RANK_SCALE` in `metrics.py`),
explicitly documented as a rough, uncalibrated placeholder pending M5 -- it only
changes ranking order, never the reported value.

**Lesson:** a definition carried over unreflectively from a different domain (face-on
pose apps) can be wrong for this project's camera angle in a way that produces
plausible-looking code and catastrophically wrong numbers -- and the bug hides behind
"high confidence" (visibility 1.0), so a confidence check alone would never have
caught it. The only thing that did was comparing the output against physical
plausibility (a torso-length-scale fault should never be several body-lengths large)
and then drawing the actual landmark positions on the actual frame.

### 2026-08-19 — M4: ghost skeleton was invisible against a bright sky

**Symptom:** the ghosted-address overlay (PLAN.md section 9) ran without error and
confirmably modified real pixels (checked via frame diff against the unmodified
source), but was not visible by eye in the rendered video -- looked like plain live
tracking with no ghost at all.

**Cause:** `GHOST_COLOR` was light gray `(200, 200, 200)` at 35% alpha. Golf is
outdoors under open sky; light gray at low opacity against a bright blue-white sky is
close to invisible, even though the pixels really were being blended.

**Fix:** bold red-orange `(0, 80, 255)` at 55% alpha. Verified by eye afterward, not
just by re-running the frame-diff check -- the diff check only proves pixels changed,
not that a human can see them, which is the actual requirement for a visual coaching
aid.

**Lesson:** "the code ran and pixels changed" is not the same acceptance bar as "a
person can actually see this." For anything whose entire job is being looked at, the
only real test is looking at it.

### 2026-08-19 — First real user review of the M4 report: feedback text was not helpful

Sent the actual owner the generated report. Three findings, all real, all useful:

**Swing 0 confirmed as not a real swing** ("I just turn to face the camera and turn
back") -- working as designed. `find_swings` surfaces every candidate and relies on
the human to discard the wrong ones by eye (PLAN.md section 7, MISTAKES.md
"detecting practice swings" -- deliberately not solved). This is the first real
confirmation that design choice holds up in actual use, not just in theory.

**Swing 2's top frame confirmed wrong** by the person who was actually there. Same
already-documented open issue from the M2 session (waggle/resettle confusing
top-of-backswing detection) -- not a new bug, but real-world corroboration that it
matters enough to eventually fix.

**The feedback text itself: "not really helpful."** The specific ask was for
concrete coaching language ("you need to stand more straight," "bend more at the
knees"), not numbers in invented units. Root problem: `feedback.py`'s original
templates put the raw measurement ("Hips moved 0.28 torso-lengths toward the ball,
peaking at frame 394") front and center as the primary sentence -- correct and
honest, but not what a person asking "what should I do differently" wants to read
first.

**Fix:** restructured every `FaultLine` into a `cue` (an actionable, coach-toned
sentence -- "Your hips are pushing toward the ball on the way down. Try to keep them
back...") plus a separate `detail` (the number, now secondary, tucked into a
collapsible `<details>` section in the report). Still 100% grounded in the same
self-referential deviation-from-address data; this was a wording/hierarchy problem,
not a measurement problem, so it required no new data and no new invented threshold.

**Explicitly NOT done, and flagged as a real design fork:** the user's own examples
("stand up straighter," "bend more at the knees") are a *different kind of claim*
than anything currently measured -- they judge address posture itself against an
implied ideal, not deviation from the golfer's own address. Every existing metric is
self-referential by deliberate design (PLAN.md section 5, AGENTS.md hard constraint
4: no invented thresholds). Building "your knee flex is wrong" would mean inventing
an external reference this project was specifically built to avoid. Did not decide
this unilaterally either way -- raised it back to the user explicitly rather than
silently picking a side.

**Lesson:** correct data in the wrong presentation register reads as "not helpful"
even when nothing about the underlying measurement is wrong. The fix for "the
feedback isn't helpful" was not better math -- it was better sentences over the same
math. Don't reach for a data/model fix when the actual complaint is about voice.
