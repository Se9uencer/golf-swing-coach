"""Feedback text. Pure template substitution -- see PLAN.md section 8 and
AGENTS.md hard constraint 5 for why there is no LLM anywhere in this file.

Every cue traces to a number `metrics.measure` produced, and every cue
compares you only to your OWN address position (PLAN.md section 5) --
never to an external "correct" posture or population norm. That's a
deliberate, documented boundary (AGENTS.md hard constraint 4: no invented
thresholds presented as verdicts). A cue like "you're standing up out of
your posture" is honest under that boundary, because it's describing
motion away from where *you* started. A cue like "you should stand
straighter" or "bend your knees more" would require an external idea of
correct posture this project has deliberately never built -- see
MISTAKES.md for the real user feedback that drew this line explicitly.

No pass/fail verdicts: each deviation gets a plain, continuum cue stating
direction and (in the detail line) magnitude, never "good" or "bad". The
one piece of judgment this module makes is which single deviation to call
out as the "biggest" thing to work on -- that's just `deviations[0]` from
metrics.measure's existing magnitude ranking, with one curated drill
attached.
"""

from dataclasses import dataclass

from swingcoach.metrics import Deviation

_PRETTY_NAMES = {
    "early_extension": "Early extension",
    "loss_of_posture": "Loss of posture",
    "head_sway": "Head sway",
    "head_lift": "Head lift",
}

# Each fault has a `cue` (an actionable, coach-toned sentence -- what you'd
# actually want to hear) and a `detail` (the number behind it, for anyone
# who wants to see the raw measurement). Selected by sign only, never by
# magnitude -- no threshold is being invented, just which of two true,
# neutral descriptions of the same number applies.
_TEMPLATES = {
    "early_extension": {
        "toward": {
            "cue": (
                "Your hips are pushing toward the ball on the way down. Try "
                "to keep them back and rotate around your spine instead of "
                "thrusting forward -- that's classic early extension."
            ),
            "detail": (
                "Hips moved {value:.2f} torso-lengths toward the ball, peaking at frame {frame}."
            ),
        },
        "away": {
            "cue": (
                "Your hips stayed behind the ball through the downswing -- "
                "no forward hip slide here."
            ),
            "detail": (
                "Hips moved {value:.2f} torso-lengths away from the ball "
                "(or stayed back), peaking at frame {frame}."
            ),
        },
    },
    "loss_of_posture": {
        "up": {
            "cue": (
                "You're standing up out of your posture as you swing. Focus "
                "on keeping the same forward bend from address all the way "
                "through impact."
            ),
            "detail": (
                "Spine angle straightened by {value:.1f} degrees from address, "
                "peaking at frame {frame}."
            ),
        },
        "down": {
            "cue": (
                "You're bending forward more than you started at address -- "
                "make sure you're not losing your spine angle downward "
                "through the swing."
            ),
            "detail": (
                "Spine angle bent {value:.1f} degrees more than address, peaking at frame {frame}."
            ),
        },
    },
    "head_sway": {
        "toward": {
            "cue": (
                "Your head is drifting toward the ball during the swing. "
                "Try to keep it steady over the ball rather than sliding "
                "forward."
            ),
            "detail": (
                "Head moved {value:.2f} torso-lengths toward the ball, peaking at frame {frame}."
            ),
        },
        "away": {
            "cue": (
                "Your head is drifting away from the ball during the swing. "
                "Try to keep it centered rather than sliding back."
            ),
            "detail": (
                "Head moved {value:.2f} torso-lengths away from the ball, peaking at frame {frame}."
            ),
        },
    },
    "head_lift": {
        "up": {
            "cue": (
                "You're lifting your head and upper body up through the "
                "swing. Try to keep your eyes on the ball and stay down "
                "through impact."
            ),
            "detail": (
                "Head rose {value:.2f} torso-lengths above its address "
                "height, peaking at frame {frame}."
            ),
        },
        "down": {
            "cue": (
                "Your head dropped lower than address through the swing -- "
                "less common, but make sure you're not diving down at the "
                "ball."
            ),
            "detail": (
                "Head dropped {value:.2f} torso-lengths below its address "
                "height, peaking at frame {frame}."
            ),
        },
    },
}

# Standard instructional drills, one per fault -- not generated, not tuned
# to any individual golfer. PLAN.md section 8: for four known faults, four
# curated drills beat generated ones.
_DRILLS = {
    "early_extension": (
        "Wall/chair drill: address the ball with your rear end lightly touching "
        "a wall or chair back behind you. Make slow swings keeping that contact "
        "all the way to impact -- the moment you lose contact is the moment "
        "early extension is creeping in."
    ),
    "loss_of_posture": (
        "Stick-in-the-ground drill: plant an alignment stick in the ground just "
        "outside your trail hip, angled to match your spine tilt at address. "
        "Make swings keeping your chest from rising above the stick until "
        "after impact."
    ),
    "head_sway": (
        "Head-behind-the-ball drill: set a headcover or ball just outside your "
        "trail ear at address. Make backswings without bumping it -- that caps "
        "how far your head is allowed to slide."
    ),
    "head_lift": (
        "Stay-down drill: place a tee a few inches in front of your ball. Keep "
        "your eyes on that tee's spot on the ground through impact, resisting "
        "the urge to look up early."
    ),
}


@dataclass(frozen=True)
class FaultLine:
    name: str
    pretty_name: str
    cue: str  # actionable, coach-toned sentence -- or an explanation if unmeasurable
    detail: str  # the supporting number, empty string if unmeasurable
    value: float | None
    unit: str
    peak_frame: int | None
    confidence: float


@dataclass(frozen=True)
class SwingFeedback:
    lines: list[FaultLine]  # every fault, in metrics.measure's ranked order
    biggest: FaultLine | None  # top-ranked MEASURABLE fault, or None
    biggest_drill: str | None
    steadiest: FaultLine | None  # smallest-magnitude MEASURABLE fault


def _direction_key(name: str, value: float) -> str:
    if name in ("early_extension", "head_sway"):
        return "toward" if value > 0 else "away"
    if name in ("loss_of_posture", "head_lift"):
        return "up" if value > 0 else "down"
    raise ValueError(f"unknown fault name: {name}")


def _line_for(d: Deviation) -> FaultLine:
    pretty = _PRETTY_NAMES.get(d.name, d.name)
    if d.value is None:
        cue = f"{pretty}: not measurable ({d.note})."
        detail = ""
    else:
        direction = _direction_key(d.name, d.value)
        pair = _TEMPLATES[d.name][direction]
        cue = pair["cue"]
        detail = pair["detail"].format(value=abs(d.value), frame=d.peak_frame)
    return FaultLine(
        name=d.name,
        pretty_name=pretty,
        cue=cue,
        detail=detail,
        value=d.value,
        unit=d.unit,
        peak_frame=d.peak_frame,
        confidence=d.confidence,
    )


def describe(deviations: list[Deviation]) -> SwingFeedback:
    """Render metrics.measure's output into coach-toned cues plus one
    prescribed drill for the single largest deviation. `deviations` is
    expected in metrics.measure's ranked order (|value| descending, None
    last) -- this function trusts that ordering rather than re-deriving it.
    """
    lines = [_line_for(d) for d in deviations]
    measurable = [line for line in lines if line.value is not None]

    biggest = measurable[0] if measurable else None
    biggest_drill = _DRILLS[biggest.name] if biggest is not None else None
    steadiest = measurable[-1] if len(measurable) >= 2 else None

    return SwingFeedback(
        lines=lines, biggest=biggest, biggest_drill=biggest_drill, steadiest=steadiest
    )
