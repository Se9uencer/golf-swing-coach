"""Feedback text. Pure template substitution -- see PLAN.md section 8 and
AGENTS.md hard constraint 5 for why there is no LLM anywhere in this file.

Every sentence traces to a number `metrics.measure` produced. No pass/fail
verdicts (AGENTS.md constraint 4): each deviation gets a plain, continuum
sentence stating direction and magnitude, never "good" or "bad". The one
piece of judgment this module makes is which single deviation to call out
as the "biggest" thing to work on -- that's just `deviations[0]` from
metrics.measure's existing magnitude ranking, with one curated drill
attached. This mirrors PLAN.md's original differentiation idea (a single
prescribed drill, not a wall of numbers) without inventing a threshold for
what counts as a "real" fault.
"""

from dataclasses import dataclass

from swingcoach.metrics import Deviation

_PRETTY_NAMES = {
    "early_extension": "Early extension",
    "loss_of_posture": "Loss of posture",
    "head_sway": "Head sway",
    "head_lift": "Head lift",
}

# {value} and {frame} are substituted; both branches for a fault are chosen
# by sign only, never by magnitude -- no threshold is being invented here,
# just which of two true, neutral descriptions of the same number applies.
_TEMPLATES = {
    "early_extension": {
        "toward": (
            "Hips moved toward the ball by {value:.2f} torso-lengths, peaking at frame {frame}."
        ),
        "away": (
            "Hips moved away from the ball (or stayed back) by {value:.2f} "
            "torso-lengths, peaking at frame {frame}."
        ),
    },
    "loss_of_posture": {
        "up": (
            "Spine angle straightened by {value:.1f} degrees from address, "
            "peaking at frame {frame} -- standing up out of the shot."
        ),
        "down": (
            "Spine angle bent {value:.1f} degrees more than at address, peaking at frame {frame}."
        ),
    },
    "head_sway": {
        "toward": (
            "Head moved {value:.2f} torso-lengths toward the ball, peaking at frame {frame}."
        ),
        "away": (
            "Head moved {value:.2f} torso-lengths away from the ball, peaking at frame {frame}."
        ),
    },
    "head_lift": {
        "up": (
            "Head rose {value:.2f} torso-lengths above its address height, "
            "peaking at frame {frame}."
        ),
        "down": (
            "Head dropped {value:.2f} torso-lengths below its address height, "
            "peaking at frame {frame}."
        ),
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
    text: str  # rendered sentence, or an explanation if unmeasurable
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
        text = f"{pretty}: not measurable ({d.note})."
    else:
        direction = _direction_key(d.name, d.value)
        template = _TEMPLATES[d.name][direction]
        text = template.format(value=abs(d.value), frame=d.peak_frame)
    return FaultLine(
        name=d.name,
        pretty_name=pretty,
        text=text,
        value=d.value,
        unit=d.unit,
        peak_frame=d.peak_frame,
        confidence=d.confidence,
    )


def describe(deviations: list[Deviation]) -> SwingFeedback:
    """Render metrics.measure's output into plain sentences plus one
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
