"""Unit tests for metrics.py's math on synthetic landmark data. Verifies
sign conventions, normalization, ranking, and the None/unmeasurable path --
NOT whether the formulas are the right ones for a real swing. Per AGENTS.md
that still needs a look at real footage; see MISTAKES.md for what that
turned up."""

import numpy as np
import pytest

from swingcoach.metrics import measure
from swingcoach.pose import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    N_LANDMARKS,
    NOSE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmarks,
)
from swingcoach.segment import Swing

FPS = 30.0
N_FRAMES = 60
ADDRESS = 10
IMPACT = 40
SHOULDER_SPACING = 40.0  # pixels, L-to-R shoulder gap -- NOT `scale` (see TORSO_LENGTH)
TORSO_LENGTH = 100.0  # pixels, pelvis-to-shoulder distance at address -> `scale`


def _blank_landmarks(fps: float = FPS, n: int = N_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    xy = np.full((n, N_LANDMARKS, 2), 500.0, dtype=np.float32)  # arbitrary, off in a corner
    visibility = np.zeros((n, N_LANDMARKS), dtype=np.float32)
    return xy, visibility


def _set_still(xy, vis, idx, pos, frames=slice(None), visibility=1.0):
    xy[frames, idx] = pos
    vis[frames, idx] = visibility


def _base_swing_landmarks():
    """A geometrically boring golfer: hips at (200,300), shoulders at
    (200,200) -- pelvis-to-shoulder distance (TORSO_LENGTH) is 100px,
    which is `scale`. Wrists at (220,320), so ball_dir is +1 (wrists are
    to the +x side of the hips at address). Nothing moves unless a test
    perturbs it."""
    xy, vis = _blank_landmarks()
    _set_still(xy, vis, LEFT_HIP, (200 - 15, 300))
    _set_still(xy, vis, RIGHT_HIP, (200 + 15, 300))
    _set_still(xy, vis, LEFT_SHOULDER, (200 - SHOULDER_SPACING / 2, 200))
    _set_still(xy, vis, RIGHT_SHOULDER, (200 + SHOULDER_SPACING / 2, 200))
    _set_still(xy, vis, LEFT_WRIST, (220 - 5, 320))
    _set_still(xy, vis, RIGHT_WRIST, (220 + 5, 320))
    _set_still(xy, vis, LEFT_EAR, (195, 190))
    _set_still(xy, vis, RIGHT_EAR, (205, 190))
    _set_still(xy, vis, NOSE, (200, 185))
    return xy, vis


def _swing():
    return Swing(start=0, address=ADDRESS, top=25, impact=IMPACT, end=N_FRAMES - 1)


def test_early_extension_sign_and_magnitude():
    xy, vis = _base_swing_landmarks()
    # pelvis moves 20px toward the ball (ball is +x, per ball_dir) and STAYS
    # there -- a step, not a single-frame blip, matching how a real fault
    # persists through impact. A single-frame impulse gets attenuated by the
    # 12Hz smoothing filter and undershoots this test's tolerance.
    xy[25:, LEFT_HIP, 0] += 20
    xy[25:, RIGHT_HIP, 0] += 20
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    ee = next(d for d in devs if d.name == "early_extension")

    assert ee.value is not None
    assert ee.value == pytest.approx(20.0 / TORSO_LENGTH, abs=0.02)
    assert 20 <= ee.peak_frame <= IMPACT


def test_early_extension_negative_when_pelvis_moves_away():
    xy, vis = _base_swing_landmarks()
    xy[25:, LEFT_HIP, 0] -= 20
    xy[25:, RIGHT_HIP, 0] -= 20
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    ee = next(d for d in devs if d.name == "early_extension")
    assert ee.value < 0


def test_loss_of_posture_positive_when_standing_up():
    xy, vis = _base_swing_landmarks()
    # Address: shoulders offset sideways from hips (a real tilt from vertical).
    xy[:, LEFT_SHOULDER, 0] -= 30
    xy[:, RIGHT_SHOULDER, 0] -= 30
    # From frame 25 on, shoulders move back over the hips -- posture straightens.
    xy[25:, LEFT_SHOULDER, 0] += 30
    xy[25:, RIGHT_SHOULDER, 0] += 30
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    lop = next(d for d in devs if d.name == "loss_of_posture")
    assert lop.value is not None
    assert lop.value > 0  # standing up = positive, per PLAN.md convention


def test_unmeasurable_when_shoulders_not_visible_at_address():
    xy, vis = _base_swing_landmarks()
    vis[ADDRESS, LEFT_SHOULDER] = 0.0
    vis[ADDRESS, RIGHT_SHOULDER] = 0.0
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    assert len(devs) == 4
    assert all(d.value is None for d in devs)
    assert all("not confidently visible" in d.note for d in devs)


def test_head_falls_back_to_nose_when_ears_unreliable():
    xy, vis = _base_swing_landmarks()
    vis[:, LEFT_EAR] = 0.05
    vis[:, RIGHT_EAR] = 0.05
    vis[:, NOSE] = 0.9
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    head_sway = next(d for d in devs if d.name == "head_sway")
    assert head_sway.value is not None
    assert "nose" in head_sway.note


def test_head_none_when_neither_ears_nor_nose_visible():
    xy, vis = _base_swing_landmarks()
    vis[:, LEFT_EAR] = 0.0
    vis[:, RIGHT_EAR] = 0.0
    vis[:, NOSE] = 0.0
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    head_sway = next(d for d in devs if d.name == "head_sway")
    head_lift = next(d for d in devs if d.name == "head_lift")
    assert head_sway.value is None
    assert head_lift.value is None


def test_ranked_by_magnitude_descending_with_none_last():
    xy, vis = _base_swing_landmarks()
    # A big, unambiguous early-extension step; everything else stays flat.
    xy[25:, LEFT_HIP, 0] += 40
    xy[25:, RIGHT_HIP, 0] += 40
    vis[:, LEFT_EAR] = 0.0
    vis[:, RIGHT_EAR] = 0.0
    vis[:, NOSE] = 0.0  # head deviations become unmeasurable -> should sort last
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)

    devs = measure(lm, _swing(), FPS)
    assert devs[0].name == "early_extension"
    assert devs[-1].value is None
    assert devs[-2].value is None


def test_impact_not_after_address_is_unmeasurable():
    xy, vis = _base_swing_landmarks()
    lm = Landmarks(xy=xy, visibility=vis, fps=FPS)
    bad_swing = Swing(start=0, address=40, top=45, impact=40, end=59)

    devs = measure(lm, bad_swing, FPS)
    assert all(d.value is None for d in devs)
