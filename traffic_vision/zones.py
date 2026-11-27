"""Count tracked objects that use a zone of the road, such as a reserved lane.

A counting line answers "how many passed here". Many mobility measures are
about *where* vehicles drive instead: a carriageway kept for accredited buses,
a motorway lane kept for Olympic traffic, a shuttle-only lane. For those, the
useful number is how many vehicles of each class drove inside the lane.

A zone is a polygon drawn on the image. An object "uses" the zone when its
contact point with the road (bottom-centre of its box) stays inside it for at
least ``min_frames`` processed frames; each object is counted once per zone.
The minimum avoids counting a car that only clips the lane edge for one frame
while changing lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Even-odd rule (ray casting). Points on an edge may fall either way."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


@dataclass
class Zone:
    name: str
    polygon: list[Point]  # pixels

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Zone '{self.name}' needs at least 3 points.")

    @classmethod
    def from_relative(cls, name: str, points: list[Point], width: int, height: int) -> "Zone":
        """Build a zone from points given as fractions of the frame size."""
        for x, y in points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Zone '{name}': points are fractions of the frame, between 0 and 1.")
        return cls(name, [(x * width, y * height) for x, y in points])

    def contains(self, p: Point) -> bool:
        return point_in_polygon(p, self.polygon)


@dataclass
class ZoneEvent:
    zone: str
    track_id: int
    road_class: str
    frame: int
    video_time_s: float


@dataclass
class ZoneCounter:
    """Count each track once per zone, after ``min_frames`` frames inside."""

    zones: list[Zone]
    min_frames: int = 2
    _frames_inside: dict[tuple[str, int], int] = field(default_factory=dict, repr=False)
    _counted: set[tuple[str, int]] = field(default_factory=set, repr=False)
    counts: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    def update(self, track_id: int, road_class: str, anchor: Point,
               frame: int, video_time_s: float) -> list[ZoneEvent]:
        events = []
        for z in self.zones:
            key = (z.name, track_id)
            if key in self._counted or not z.contains(anchor):
                continue
            n = self._frames_inside.get(key, 0) + 1
            self._frames_inside[key] = n
            if n >= self.min_frames:
                self._counted.add(key)
                per_zone = self.counts.setdefault(z.name, {})
                per_zone[road_class] = per_zone.get(road_class, 0) + 1
                events.append(ZoneEvent(z.name, track_id, road_class, frame, round(video_time_s, 3)))
        return events

    def totals(self) -> dict[str, dict[str, int]]:
        return {z: dict(c) for z, c in self.counts.items()}
