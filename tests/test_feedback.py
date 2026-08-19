"""Unit tests for feedback.py's templating. Pure string substitution, so
this is fully testable without real footage -- unlike the CV modules, there
is no "does this look right on a real swing" question here, just "does the
right template get picked and filled in correctly"."""

from swingcoach.feedback import describe
from swingcoach.metrics import Deviation


def _dev(name, value, unit, frame=10, confidence=1.0, note=""):
    return Deviation(
        name=name, value=value, unit=unit, peak_frame=frame, confidence=confidence, note=note
    )


def test_biggest_is_top_ranked_measurable():
    devs = [
        _dev("early_extension", 0.5, "torso-lengths", frame=20),
        _dev("loss_of_posture", 10.0, "degrees", frame=15),
        _dev("head_sway", None, "torso-lengths", frame=None, confidence=0.0, note="not visible"),
        _dev("head_lift", None, "torso-lengths", frame=None, confidence=0.0, note="not visible"),
    ]
    fb = describe(devs)
    assert fb.biggest is not None
    assert fb.biggest.name == "early_extension"
    assert fb.biggest_drill is not None
    assert "wall" in fb.biggest_drill.lower() or "chair" in fb.biggest_drill.lower()


def test_steadiest_is_smallest_measurable():
    devs = [
        _dev("early_extension", 0.5, "torso-lengths"),
        _dev("loss_of_posture", 3.0, "degrees"),
    ]
    fb = describe(devs)
    assert fb.steadiest is not None
    assert fb.steadiest.name == "loss_of_posture"


def test_no_steadiest_with_only_one_measurable():
    devs = [
        _dev("early_extension", 0.5, "torso-lengths"),
        _dev("loss_of_posture", None, "degrees", frame=None, confidence=0.0, note="x"),
    ]
    fb = describe(devs)
    assert fb.steadiest is None
    assert fb.biggest is not None


def test_no_biggest_when_nothing_measurable():
    devs = [
        _dev("early_extension", None, "torso-lengths", frame=None, confidence=0.0, note="x"),
        _dev("loss_of_posture", None, "degrees", frame=None, confidence=0.0, note="x"),
    ]
    fb = describe(devs)
    assert fb.biggest is None
    assert fb.biggest_drill is None
    assert "not measurable" in fb.lines[0].cue
    assert fb.lines[0].detail == ""


def test_direction_cue_matches_sign():
    toward = describe([_dev("early_extension", 0.4, "torso-lengths", frame=5)]).lines[0]
    assert "toward the ball" in toward.cue

    away = describe([_dev("early_extension", -0.4, "torso-lengths", frame=5)]).lines[0]
    assert "behind the ball" in away.cue


def test_detail_shows_absolute_value_and_frame():
    line = describe([_dev("loss_of_posture", -12.221, "degrees", frame=7)]).lines[0]
    assert "12.2" in line.detail
    assert "-12.2" not in line.detail
    assert "frame 7" in line.detail


def test_cue_is_actionable_not_a_bare_measurement():
    # The whole point of this rewrite: the cue should read like coaching
    # advice, not "X moved Y units at frame Z" -- the number belongs in
    # `detail`, not `cue`.
    line = describe([_dev("head_lift", 0.5, "torso-lengths", frame=12)]).lines[0]
    assert "torso-lengths" not in line.cue
    assert "frame" not in line.cue
    assert "torso-lengths" in line.detail
    assert "frame 12" in line.detail
