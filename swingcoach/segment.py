"""Swing splitting and event detection from wrist speed. Pure numpy/scipy.

See PLAN.md section 7 for the algorithm and its reasoning, and MISTAKES.md
for why this is a heuristic on a plottable signal rather than a learned
event detector.

All thresholds below are first-pass values, not calibrated against real
footage yet -- that's M5 (PLAN.md build order). Expect to retune them once
there's ~50 real swings to check against.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from swingcoach.pose import LEFT_WRIST, RIGHT_WRIST, Landmarks, weighted_midpoint
from swingcoach.smooth import lowpass

WRIST_SMOOTH_CUTOFF_HZ = 20.0  # target cutoff at capture rates >= ~50fps
WRIST_SMOOTH_NYQUIST_MARGIN = 0.45  # cap cutoff to this fraction of Nyquist
MIN_SWING_SEPARATION_S = 2.0
PEAK_PROMINENCE_FACTOR = 3.0  # prominence = factor * median(speed)
QUIET_PERCENTILE = 25  # "quiet" = below this percentile of the WHOLE CLIP's speed
QUIET_MARGIN = 1.5  # safety margin over that percentile, for jitter
MIN_ADDRESS_HOLD_S = 0.3
IMPACT_SEARCH_WINDOW_S = 0.15
TOP_SEARCH_BACK_S = 1.5
ADDRESS_SEARCH_BACK_S = 1.5
PRE_ROLL_S = 0.2
POST_ROLL_S = 0.5
MIN_WRIST_VISIBILITY = 0.4  # combined L+R; below this the frame is NaN, not guessed


@dataclass(frozen=True)
class Swing:
    start: int
    address: int
    top: int
    impact: int
    end: int


def wrist_midpoint(lm: Landmarks) -> np.ndarray:
    """(n_frames, 2) pixel trace of the hands, used as the segmentation signal.

    Visibility-weighted, not a plain average: an unweighted average lets a
    confident wrist get dragged around by an unreliable one (real clip: right
    wrist visibility fell from 0.69 to 0.39 across 10 frames during a
    follow-through, right where an unweighted average produced a spurious
    ~2900px/s "speed" spike that outcompeted the real swing's peak and got it
    dropped by the min-separation rule). When both wrists are near-invisible
    (e.g. the camera pointed at a simulator screen instead of the golfer --
    this happened on real footage, visibility ~0.1 throughout), the frame is
    NaN, which smooth.lowpass already knows how to interpolate across -- never
    a fabricated position. See MISTAKES.md. The weighted-average math itself
    lives in pose.weighted_midpoint, shared with metrics.py.
    """
    midpoint, total_vis = weighted_midpoint(lm, LEFT_WRIST, RIGHT_WRIST)
    midpoint = midpoint.copy()
    midpoint[total_vis < MIN_WRIST_VISIBILITY] = np.nan
    return midpoint


def wrist_trustworthy(lm: Landmarks) -> np.ndarray:
    """Boolean mask: was this frame's wrist position actually observed?

    smooth.lowpass linearly interpolates across NaN gaps so positions stay
    continuous, which is the right call for keeping a plottable trace -- but
    a long gap (real clip: 30 frames, ~1s, camera pointed at a simulator
    screen instead of the golfer) interpolates to a straight line, and
    filtfilt then RINGS at the boundary where that flat interpolated stretch
    meets real data again -- producing a fabricated dip that looks like
    genuine stillness and a fabricated spike that looks like a genuine swing
    (real clip: a 9600px/s spike one frame after tracking resumed). Position
    interpolation is fine; treating its *derivative* as observed motion is
    not. find_swings uses this mask to refuse to place an event on a frame
    that was never actually seen. See MISTAKES.md.
    """
    _, total_vis = weighted_midpoint(lm, LEFT_WRIST, RIGHT_WRIST)
    return total_vis >= MIN_WRIST_VISIBILITY


def _wrist_cutoff_hz(fps: float) -> float:
    """20Hz assumes true 240fps capture (PLAN.md section 3), where it's well
    inside the 120Hz Nyquist limit. Real footage is frequently much slower
    than that -- both clips used to build this were ~30fps, where 20Hz would
    violate Nyquist outright -- so this scales the cutoff down for lower
    capture rates instead of hardcoding a value that only works at the
    capture spec's target rate. See MISTAKES.md."""
    nyquist = fps / 2.0
    return min(WRIST_SMOOTH_CUTOFF_HZ, WRIST_SMOOTH_NYQUIST_MARGIN * nyquist)


def wrist_speed(lm: Landmarks, fps: float) -> np.ndarray:
    """Smoothed wrist speed in pixels/second. Exposed separately from
    find_swings so the diagnostic plot can show exactly what segmentation
    saw, not a re-derived approximation of it."""
    smoothed = lowpass(wrist_midpoint(lm), fps=fps, cutoff_hz=_wrist_cutoff_hz(fps))
    velocity = np.gradient(smoothed, axis=0) * fps
    return np.linalg.norm(velocity, axis=1)


def _smoothed_wrist_y(lm: Landmarks, fps: float) -> np.ndarray:
    smoothed = lowpass(wrist_midpoint(lm), fps=fps, cutoff_hz=_wrist_cutoff_hz(fps))
    return smoothed[:, 1]


def _find_top(speed: np.ndarray, peak_idx: int, fps: float, quiet_threshold: float) -> int | None:
    """Last local minimum of speed before the peak, below the quiet threshold."""
    lo = max(0, peak_idx - int(TOP_SEARCH_BACK_S * fps))
    window = speed[lo:peak_idx]
    if len(window) < 3:
        return None
    minima = [
        i
        for i in range(1, len(window) - 1)
        if window[i] <= window[i - 1] and window[i] <= window[i + 1] and window[i] < quiet_threshold
    ]
    if not minima:
        return None
    return lo + minima[-1]


def _find_address(
    speed: np.ndarray, top_idx: int, fps: float, quiet_threshold: float
) -> int | None:
    """End of the last sustained (>= MIN_ADDRESS_HOLD_S) quiet window before top."""
    hold_frames = int(MIN_ADDRESS_HOLD_S * fps)
    lo = max(0, top_idx - int(ADDRESS_SEARCH_BACK_S * fps))
    window = speed[lo:top_idx]
    if len(window) < hold_frames:
        return None

    quiet = window < quiet_threshold
    run_end = None
    run_len = 0
    for i, is_quiet in enumerate(quiet):
        if is_quiet:
            run_len += 1
            if run_len >= hold_frames:
                run_end = i
        else:
            run_len = 0
    if run_end is None:
        return None
    return lo + run_end


def _find_impact(wrist_y: np.ndarray, peak_idx: int, fps: float) -> int:
    """Lowest point of the hand path near the speed peak. y increases downward,
    so the lowest point is the max, not the min -- see MISTAKES.md."""
    half = int(IMPACT_SEARCH_WINDOW_S * fps)
    lo = max(0, peak_idx - half)
    hi = min(len(wrist_y), peak_idx + half + 1)
    return lo + int(np.argmax(wrist_y[lo:hi]))


def find_swings(lm: Landmarks, fps: float) -> list[Swing]:
    """Split a clip into candidate swings from wrist speed.

    A golf swing is loud in wrist-speed terms -- roughly an order of
    magnitude above anything else done standing in a bay -- so peaks in
    |d(wrist)/dt| mark swings directly, with no ML model needed.

    Every detected peak is returned as a candidate, including practice
    swings: this function cannot and does not try to tell them apart from
    real ones (MISTAKES.md, "detecting practice swings"). A candidate whose
    top or address frame can't be located confidently -- e.g. a swing too
    close to the start of the clip to have a pre-roll, or one that lands on
    a frame the wrists genuinely weren't tracked on -- is dropped rather
    than filled in with a guessed frame index.

    "Quiet" (used to find top and address) is calibrated against the whole
    clip's resting-speed floor, not a fraction of each swing's own peak --
    a fraction of peak speed is 10x too loose in practice (real clip: ~580
    px/s vs. a true stillness floor of ~10-50 px/s) and grabs noise dips
    inside the downswing itself instead of genuine stillness. See
    MISTAKES.md.
    """
    speed = wrist_speed(lm, fps)
    wrist_y = _smoothed_wrist_y(lm, fps)
    trustworthy = wrist_trustworthy(lm)

    median_speed = float(np.median(speed))
    prominence = max(PEAK_PROMINENCE_FACTOR * median_speed, 1e-6)
    distance = max(1, int(MIN_SWING_SEPARATION_S * fps))
    quiet_threshold = float(np.percentile(speed, QUIET_PERCENTILE)) * QUIET_MARGIN

    peaks, _ = find_peaks(speed, prominence=prominence, distance=distance)

    swings: list[Swing] = []
    for peak_idx in peaks:
        peak_idx = int(peak_idx)
        if not trustworthy[peak_idx]:
            continue

        top_idx = _find_top(speed, peak_idx, fps, quiet_threshold)
        if top_idx is None or not trustworthy[top_idx]:
            continue
        address_idx = _find_address(speed, top_idx, fps, quiet_threshold)
        if address_idx is None or not trustworthy[address_idx]:
            continue
        impact_idx = _find_impact(wrist_y, peak_idx, fps)
        if not trustworthy[impact_idx]:
            continue

        start = max(0, address_idx - int(PRE_ROLL_S * fps))
        end = min(lm.n_frames - 1, impact_idx + int(POST_ROLL_S * fps))

        swings.append(
            Swing(start=start, address=address_idx, top=top_idx, impact=impact_idx, end=end)
        )

    return swings
