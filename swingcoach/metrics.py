"""The four planned fault measurements -- three implemented for now. Pure numpy.

See PLAN.md section 4 for the formulas, section 5 for why the address frame
is a safe reference, and section 6 for why this ranks deviations instead of
emitting a pass/fail verdict.

Search windows here use [address, impact], not [top, impact] as PLAN.md
section 4 originally specified -- top-of-backswing detection is not yet
reliable (MISTAKES.md, "M2 first run against real multi-swing footage").
Using the wider window doesn't corrupt these particular metrics: during the
backswing the pelvis and spine move, if anything, in the OPPOSITE direction
from the downswing fault each metric is meant to catch, so the true peak
still wins the |value|-max search. Revisit once top-of-backswing detection
is fixed.

hand_path is not implemented here. PLAN.md already flagged it as the
softest of the four metrics, "likely reduced to a purely visual trace with
no number, or cut" -- and it's the one metric that actually needs a
reliable top, which M2 doesn't have. Nothing to adapt around; it's simply
not built yet.
"""

from dataclasses import dataclass

import numpy as np

from swingcoach.pose import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmarks,
    weighted_midpoint,
)
from swingcoach.segment import Swing
from swingcoach.smooth import lowpass

BODY_SMOOTH_CUTOFF_HZ = 12.0  # PLAN.md section 7: ~12Hz for body segments
BODY_SMOOTH_NYQUIST_MARGIN = 0.45  # same Nyquist-safety reasoning as segment.py
MIN_SCALE_VISIBILITY = 0.5  # pelvis+shoulders must be this visible at address to trust `scale`
MIN_HEAD_VISIBILITY = 0.3  # combined L+R ear (or nose) visibility to trust the head trace

_UNMEASURABLE_FAULTS = (
    ("early_extension", "torso-lengths"),
    ("loss_of_posture", "degrees"),
    ("head_sway", "torso-lengths"),
    ("head_lift", "torso-lengths"),
)

# Ranking key = |value| / _RANK_SCALE[unit]. Without this, "1.0 torso-lengths"
# and "1.0 degrees" would compare as equal in magnitude when they are wildly
# different in how notable they are -- a rough real example: a 40px pelvis
# shift (1.0 shoulder-widths, back when `scale` was shoulder width -- an
# enormous fault) numerically lost a ranking to the ~22 degree spine-tilt
# change it incidentally caused, simply because "degrees" produces bigger raw
# numbers than the distance unit for physically comparable deviations. These
# divisors are a rough placeholder -- "roughly how big a value in this unit is
# already a large, clearly-real fault" -- not a calibrated equivalence. Like
# segment.py's thresholds, expect to retune once there's real footage with
# known-bad swings to rank against (PLAN.md M5). They affect ranking ORDER
# only; Deviation.value stays the honest, un-rescaled number in its natural
# unit.
_RANK_SCALE = {
    "torso-lengths": 0.15,
    "degrees": 15.0,
}


@dataclass(frozen=True)
class Deviation:
    name: str
    value: float | None  # None when not measurable
    unit: str  # "torso-lengths" | "degrees"
    peak_frame: int | None
    confidence: float  # 0..1, driven by landmark visibility
    note: str = ""  # why it's None, when it is


def _body_cutoff_hz(fps: float) -> float:
    """Same Nyquist-safety reasoning as segment._wrist_cutoff_hz -- 12Hz
    assumes reasonably high capture rates; scaled down below that so this
    never hands smooth.lowpass an invalid cutoff. See MISTAKES.md."""
    nyquist = fps / 2.0
    return min(BODY_SMOOTH_CUTOFF_HZ, BODY_SMOOTH_NYQUIST_MARGIN * nyquist)


def _smooth_point(xy: np.ndarray, fps: float) -> np.ndarray:
    return lowpass(xy, fps=fps, cutoff_hz=_body_cutoff_hz(fps))


def _tilt_from_vertical_deg(vec: np.ndarray) -> np.ndarray:
    """Unsigned angle between `vec` (n,2) and the image's vertical axis, in
    degrees. 0 = perfectly vertical, 90 = perfectly horizontal. Unsigned is
    deliberate: loss-of-posture only cares how far from vertical the spine
    is, not which way it leans, and standing up always decreases this
    number regardless of lean direction."""
    dx, dy = vec[:, 0], vec[:, 1]
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def _peak_in_window(values: np.ndarray, lo: int, hi: int) -> tuple[float, int]:
    """The |value|-max within [lo, hi] inclusive. Returns (signed value, frame)."""
    window = values[lo : hi + 1]
    local_idx = int(np.argmax(np.abs(window)))
    return float(window[local_idx]), lo + local_idx


def _head_trace(lm: Landmarks) -> tuple[np.ndarray, np.ndarray, str]:
    """(xy, visibility, source). Ear midpoint first, nose as fallback -- the
    nose is frequently invisible from directly behind (MISTAKES.md, "the
    nose landmark is frequently invisible from behind")."""
    ear_xy, ear_vis = weighted_midpoint(lm, LEFT_EAR, RIGHT_EAR)
    if np.nanmean(ear_vis) / 2.0 >= MIN_HEAD_VISIBILITY:
        return ear_xy, ear_vis, "ears"

    nose_xy = lm.xy[:, NOSE]
    nose_vis = lm.visibility[:, NOSE]
    if np.nanmean(nose_vis) >= MIN_HEAD_VISIBILITY:
        return nose_xy, nose_vis, "nose"

    return ear_xy, ear_vis, "none"


def _all_unmeasurable(note: str) -> list[Deviation]:
    return [
        Deviation(name, None, unit, None, 0.0, note=note) for name, unit in _UNMEASURABLE_FAULTS
    ]


def _rank(deviations: list[Deviation]) -> list[Deviation]:
    """|value|/_RANK_SCALE[unit] descending, so faults in different units are
    ranked on a roughly comparable footing instead of comparing raw numbers
    across degrees and torso-lengths. None (unmeasurable) sorts last, per
    AGENTS.md."""

    def key(d: Deviation) -> float:
        if d.value is None:
            return float("-inf")
        return abs(d.value) / _RANK_SCALE[d.unit]

    return sorted(deviations, key=key, reverse=True)


def measure(lm: Landmarks, swing: Swing, fps: float) -> list[Deviation]:
    """Measure early extension, loss of posture, and head sway/lift for one
    swing, each as a deviation from the swing's own address frame (PLAN.md
    section 5). Ranked by |value| descending; unmeasurable deviations sort
    last rather than being omitted, so the caller always sees why.
    """
    address, impact = swing.address, swing.impact
    if impact <= address:
        return _all_unmeasurable(f"impact frame ({impact}) not after address frame ({address})")

    pelvis_xy, pelvis_vis = weighted_midpoint(lm, LEFT_HIP, RIGHT_HIP)
    shoulder_xy, shoulder_vis = weighted_midpoint(lm, LEFT_SHOULDER, RIGHT_SHOULDER)
    wrist_xy, _ = weighted_midpoint(lm, LEFT_WRIST, RIGHT_WRIST)

    # `scale` is torso length (pelvis-to-shoulder distance at address), NOT
    # shoulder width. Shoulder width was the original choice and is wrong for
    # this camera: from DTL the two shoulders sit almost in a line away from
    # the camera, so their 2D separation collapses toward zero and every
    # distance divided by it blows up (real clip: 20px at one address frame,
    # 134px at another, same golfer, same fixed camera). Torso length stays
    # large and stable regardless of viewing angle. See MISTAKES.md and
    # PLAN.md section 4.
    scale_vis = (
        min(float(pelvis_vis[address]), float(shoulder_vis[address])) / 2.0
    )  # weighted_midpoint visibility is summed (0..2); rescale to 0..1
    scale = float(np.linalg.norm(shoulder_xy[address] - pelvis_xy[address]))
    if scale_vis < MIN_SCALE_VISIBILITY or not np.isfinite(scale) or scale < 1e-3:
        return _all_unmeasurable(
            f"hips/shoulders not confidently visible at address (visibility={scale_vis:.2f})"
        )

    pelvis_xy = _smooth_point(pelvis_xy, fps)
    shoulder_xy = _smooth_point(shoulder_xy, fps)

    # Self-calibrating handedness (AGENTS.md "ball direction"): the hands
    # hang out over the ball at address, so pelvis-to-wrist IS ball direction.
    ball_dir = float(np.sign(wrist_xy[address, 0] - pelvis_xy[address, 0])) or 1.0

    deviations: list[Deviation] = []

    ee_series = ball_dir * (pelvis_xy[:, 0] - pelvis_xy[address, 0]) / scale
    ee_value, ee_frame = _peak_in_window(ee_series, address, impact)
    ee_conf = float(np.mean(pelvis_vis[address : impact + 1])) / 2.0
    deviations.append(Deviation("early_extension", ee_value, "torso-lengths", ee_frame, ee_conf))

    spine_vec = shoulder_xy - pelvis_xy
    tilt = _tilt_from_vertical_deg(spine_vec)
    lop_series = tilt[address] - tilt
    lop_value, lop_frame = _peak_in_window(lop_series, address, impact)
    posture_vis = np.minimum(pelvis_vis, shoulder_vis)
    lop_conf = float(np.mean(posture_vis[address : impact + 1])) / 2.0
    deviations.append(Deviation("loss_of_posture", lop_value, "degrees", lop_frame, lop_conf))

    head_xy, head_vis, head_source = _head_trace(lm)
    if head_source == "none":
        note = "head not confidently visible (neither ears nor nose)"
        deviations.append(Deviation("head_sway", None, "torso-lengths", None, 0.0, note=note))
        deviations.append(Deviation("head_lift", None, "torso-lengths", None, 0.0, note=note))
    else:
        head_xy = _smooth_point(head_xy, fps)
        sway_series = ball_dir * (head_xy[:, 0] - head_xy[address, 0]) / scale
        lift_series = (head_xy[address, 1] - head_xy[:, 1]) / scale  # y is down; +ve = up

        sway_value, sway_frame = _peak_in_window(sway_series, address, impact)
        lift_value, lift_frame = _peak_in_window(lift_series, address, impact)
        head_norm = 2.0 if head_source == "ears" else 1.0
        head_conf = float(np.mean(head_vis[address : impact + 1])) / head_norm

        note = f"from {head_source}"
        deviations.append(
            Deviation("head_sway", sway_value, "torso-lengths", sway_frame, head_conf, note=note)
        )
        deviations.append(
            Deviation("head_lift", lift_value, "torso-lengths", lift_frame, head_conf, note=note)
        )

    return _rank(deviations)
