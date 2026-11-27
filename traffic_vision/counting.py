"""Count tracked objects that cross a line, with their direction.

Counting detections frame by frame overcounts: a car seen in 90 frames is
still one car. Here an object is counted once, when the bottom-centre of its
box (where it touches the road) crosses a counting line drawn across the lanes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

Point = tuple[float, float]


def _side(a: Point, b: Point, p: Point) -> float:
    """Sign of the cross product (b - a) x (p - a): >0 on one side, <0 on the other."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _within_segment(a: Point, b: Point, p: Point, q: Point) -> bool:
    """True if the move p -> q crosses the segment a-b itself, not its extension."""
    d1, d2 = _side(p, q, a), _side(p, q, b)
    return d1 * d2 <= 0


@dataclass
class CrossingEvent:
    track_id: int
    road_class: str
    direction: str  # one of CountingLine.directions
    frame: int
    video_time_s: float


@dataclass
class CountingLine:
    """A counting line between points a and b, in pixels.

    An object moving from the negative side of a->b to the positive side is
    counted in ``directions[0]``, the opposite move in ``directions[1]``.
    """

    a: Point
    b: Point
    name: str = "line"
    directions: tuple[str, str] = ("in", "out")
    _last: dict[int, Point] = field(default_factory=dict, repr=False)
    _counted: set[tuple[int, str]] = field(default_factory=set, repr=False)
    counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)), repr=False
    )

    @classmethod
    def from_relative(cls, a: Point, b: Point, width: int, height: int, **kw) -> "CountingLine":
        """Build a line from coordinates given as fractions of the frame size."""
        return cls((a[0] * width, a[1] * height), (b[0] * width, b[1] * height), **kw)

    def update(self, track_id: int, road_class: str, anchor: Point,
               frame: int, video_time_s: float) -> CrossingEvent | None:
        """Feed the new position of a track; return an event if it just crossed."""
        prev = self._last.get(track_id)
        self._last[track_id] = anchor
        if prev is None:
            return None
        s_prev, s_now = _side(self.a, self.b, prev), _side(self.a, self.b, anchor)
        if s_prev == 0 or s_prev * s_now >= 0:
            return None
        if not _within_segment(self.a, self.b, prev, anchor):
            return None
        direction = self.directions[0] if s_prev < 0 < s_now else self.directions[1]
        key = (track_id, direction)
        if key in self._counted:  # a box jittering on the line counts once
            return None
        self._counted.add(key)
        self.counts[road_class][direction] += 1
        return CrossingEvent(track_id, road_class, direction, frame, round(video_time_s, 3))

    def totals(self) -> dict[str, dict[str, int]]:
        """Counts as plain dicts: {class: {direction: n}}."""
        return {cls: dict(d) for cls, d in self.counts.items()}
