"""Check mobility measures against what the video shows.

A traffic plan, a regulation or a new infrastructure is a list of rules: this
lane for buses, no trucks between 6:00 and 21:00, no commercial motorbikes in
these districts, this road closed on Sunday mornings. Each rule implies
something that can be counted. Five kinds of rules are covered:

- ``reserved_lane``: who actually drove in the lane (zone events), and the
  share of the classes the lane was meant for;
- ``class_ban``: how many vehicles of a banned class still passed, during the
  ban hours against outside them, on a line or inside a zone (a pedestrian
  street is a zone where vehicles are banned);
- ``movement_ban``: how many vehicles still made a banned turn, from one
  counting line to another;
- ``service_frequency``: whether the buses promised per hour actually pass;
- ``closure``: a closed road moves traffic elsewhere. Its effect is measured on
  the diversion routes with a difference-in-differences
  (:func:`traffic_vision.evaluation.diff_in_diff`).

Measures are described, with their sources, in registries under
``measures/``: the Dakar 2026 Youth Olympic Games, the Dakar BRT corridor, the
okada ban in Lagos and the car-free days of Kigali are worked examples, and
``measures/TEMPLATE.yaml`` starts a new one.

Video sees classes, not permits: a car in a bus lane may be an accredited
vehicle, and a "truck" may weigh less than the 3.5 tonnes of the rule. Results
describe use of the road, they are not lists of offences, and no plate or face
is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("reserved_lane", "class_ban", "movement_ban", "service_frequency", "closure")
Z975 = 1.959963984540054


def _chi2_quantile(p_z: float, k: float) -> float:
    """Chi-square quantile by the Wilson-Hilferty approximation (no scipy)."""
    return k * (1 - 2 / (9 * k) + p_z * math.sqrt(2 / (9 * k))) ** 3


def poisson_ci(n: int, z: float = Z975) -> tuple[float, float]:
    """Approximate 95% interval for a Poisson mean after observing ``n`` events.

    Garwood's exact interval with chi-square quantiles approximated by
    Wilson-Hilferty, accurate to about 1% for this use. For n = 0 the upper
    bound is about 3.7: seeing no truck in one hour is compatible with up to
    3.7 trucks per hour.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    lo = 0.0 if n == 0 else _chi2_quantile(-z, 2 * n) / 2
    hi = _chi2_quantile(z, 2 * n + 2) / 2
    return lo, hi


@dataclass
class TimeWindow:
    """Daily hours during which a rule applies, e.g. 07:00-19:00 or 18:00-07:00."""

    start: time
    end: time

    @classmethod
    def parse(cls, text: str) -> "TimeWindow":
        try:
            a, b = (s.strip() for s in text.split("-"))
            return cls(time.fromisoformat(a), time.fromisoformat(b))
        except ValueError as e:
            raise ValueError(f"Time window must look like 07:00-19:00, got '{text}'") from e

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end  # window over midnight

    def __str__(self) -> str:
        return f"{self.start:%H:%M}-{self.end:%H:%M}"


@dataclass
class Measure:
    id: str
    name: str
    kind: str
    place: str
    sources: list[str]
    period: list[str] | None = None  # [first day, last day], ISO dates
    window: TimeWindow | None = None
    allowed_classes: list[str] = field(default_factory=list)
    banned_classes: list[str] = field(default_factory=list)
    zone: str | None = None
    lines: list[str] | None = None          # movement_ban: [from_line, to_line]
    expected_per_hour: float | None = None  # service_frequency: promised vehicles per hour
    recurrence: str = ""                    # e.g. "first and third Sunday of the month"
    indicator: str = ""
    observability: str = "partial"  # high | partial | low
    video_cannot_see: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind must be one of {KINDS}")
        if not self.sources:
            raise ValueError(f"{self.id}: every measure needs at least one source")
        if self.kind == "reserved_lane" and not (self.zone and self.allowed_classes):
            raise ValueError(f"{self.id}: a reserved lane needs a zone and allowed classes")
        if self.kind == "class_ban" and not self.banned_classes:
            raise ValueError(f"{self.id}: a class ban needs banned classes")
        if self.kind == "movement_ban" and not (self.lines and len(self.lines) == 2):
            raise ValueError(f"{self.id}: a movement ban needs lines: [from_line, to_line]")
        if self.kind == "service_frequency" and not (self.expected_per_hour
                                                     and self.allowed_classes):
            raise ValueError(f"{self.id}: a service frequency needs expected_per_hour "
                             "and allowed_classes (the vehicles of the service)")

    def in_period(self, day: datetime) -> bool | None:
        """Whether a day falls within the measure's dates; open-ended if no last day."""
        if not self.period:
            return None
        first = datetime.fromisoformat(str(self.period[0])).date()
        last = self.period[1] if len(self.period) > 1 else None
        if day.date() < first:
            return False
        if last in (None, "", "open"):
            return True
        return day.date() <= datetime.fromisoformat(str(last)).date()


def registry_info(path: str | Path) -> dict:
    """The header of a registry: city, country, context profile, policy type."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    return {k: v for k, v in data.items() if k != "measures"}


def load_registry(path: str | Path) -> dict[str, Measure]:
    """Read a measure registry (YAML) into Measure objects keyed by id."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    out = {}
    for m in data["measures"]:
        m = dict(m)
        if m.get("window"):
            m["window"] = TimeWindow.parse(m["window"])
        out[m["id"]] = Measure(**m)
    return out


# ---------------------------------------------------------------------------
# Reserved lanes
# ---------------------------------------------------------------------------

@dataclass
class LaneUse:
    zone: str
    allowed: list[str]
    n_vehicles: int
    n_allowed: int
    share_allowed: float | None
    share_ci: tuple[float, float] | None
    others_per_hour: float
    others_per_hour_ci: tuple[float, float]
    hours: float
    by_class: dict[str, int]

    def as_text(self) -> str:
        if not self.n_vehicles:
            return f"No vehicle used zone '{self.zone}' during {self.hours * 60:.0f} min."
        allowed = "/".join(self.allowed)
        ci = (f" (95% CI {self.share_ci[0]:.0%} to {self.share_ci[1]:.0%})"
              if self.share_ci else "")
        lo, hi = self.others_per_hour_ci
        detail = ", ".join(f"{k} {v}" for k, v in sorted(self.by_class.items()))
        return (f"{self.n_vehicles} vehicles used '{self.zone}' in {self.hours * 60:.0f} min "
                f"({detail}). {self.share_allowed:.0%} were {allowed}{ci}. Other vehicles: "
                f"{self.others_per_hour:.0f} per hour (95% CI {lo:.0f} to {hi:.0f}). "
                "Video cannot see permits: other vehicles may be accredited, "
                "so this measures use of the lane, not offences.")


def lane_use(zone_events: pd.DataFrame, zone: str, allowed: list[str], duration_s: float,
             interval_s: float = 60.0, n_boot: int = 5000, seed: int = 0) -> LaneUse:
    """Who used a reserved lane, with an interval on the allowed share.

    The share is a ratio of totals; its interval resamples time intervals
    (allowed count, total count) together, so busy and quiet minutes keep
    their weight. With fewer than five intervals a resampling interval means
    nothing, so the Wilson score interval is used instead; it treats vehicles
    as independent and is too narrow when buses come in platoons.
    """
    ev = zone_events[(zone_events["zone"] == zone) & (zone_events["road_class"] != "pedestrian")]
    hours = max(duration_s / 3600, 1e-9)
    n = len(ev)
    n_ok = int(ev["road_class"].isin(allowed).sum())
    others = n - n_ok
    o_lo, o_hi = poisson_ci(others)
    share = share_ci = None
    if n:
        share = n_ok / n
        k = max(int(math.ceil(duration_s / interval_s)), 1)
        idx = np.minimum((ev["video_time_s"].to_numpy() // interval_s).astype(int), k - 1)
        tot = np.bincount(idx, minlength=k).astype(float)
        ok = np.bincount(idx, weights=ev["road_class"].isin(allowed).to_numpy().astype(float),
                         minlength=k)
        if k >= 5:
            rng = np.random.default_rng(seed)
            pick = rng.integers(0, k, (n_boot, k))
            num, den = ok[pick].sum(axis=1), tot[pick].sum(axis=1)
            ratios = num[den > 0] / den[den > 0]
            if len(ratios):
                share_ci = tuple(float(v) for v in np.percentile(ratios, [2.5, 97.5]))
        else:
            z = Z975
            centre = (share + z * z / (2 * n)) / (1 + z * z / n)
            half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            share_ci = (max(0.0, centre - half), min(1.0, centre + half))
    return LaneUse(zone, list(allowed), n, n_ok, share, share_ci, others / hours,
                   (o_lo / hours, o_hi / hours), hours,
                   ev["road_class"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# Bans on a class of vehicle during some hours
# ---------------------------------------------------------------------------

def clock_times(df: pd.DataFrame, start_clock: str | None) -> pd.Series:
    """Clock time of each event, from a ``clock_time`` column or a start time."""
    if "clock_time" in df.columns and df["clock_time"].notna().all() and len(df):
        return pd.to_datetime(df["clock_time"])
    if start_clock is None:
        raise ValueError("Clock time unknown: run with --start-clock, or pass start_clock.")
    t0 = datetime.fromisoformat(start_clock)
    return pd.Series([t0 + timedelta(seconds=float(s)) for s in df["video_time_s"]],
                     index=df.index, dtype="datetime64[ns]")


@dataclass
class BanCheck:
    banned: list[str]
    window: str
    hours_in: float
    hours_out: float
    n_in: int
    n_out: int
    rate_in: float | None
    rate_in_ci: tuple[float, float] | None
    rate_out: float | None
    rate_out_ci: tuple[float, float] | None
    in_period: bool | None

    def as_text(self) -> str:
        cls = "/".join(self.banned)
        parts = []
        if self.hours_in > 0:
            lo, hi = self.rate_in_ci
            parts.append(f"during the ban ({self.window}): {self.n_in} {cls} in "
                         f"{self.hours_in * 60:.0f} min, {self.rate_in:.1f} per hour "
                         f"(95% CI {lo:.1f} to {hi:.1f})")
        if self.hours_out > 0:
            lo, hi = self.rate_out_ci
            parts.append(f"outside the ban: {self.n_out} in {self.hours_out * 60:.0f} min, "
                         f"{self.rate_out:.1f} per hour (95% CI {lo:.1f} to {hi:.1f})")
        period = {True: "", False: " Warning: the recording is outside the measure's dates.",
                  None: ""}[self.in_period]
        text = "; ".join(parts)
        return text[:1].upper() + text[1:] + "." + period + (
            " The class seen on video is not the legal category (a weight limit, a "
            "commercial licence, a permit), so read these as observed flows, not offences.")


def ban_check(crossings: pd.DataFrame, banned: list[str], window: TimeWindow,
              duration_s: float, start_clock: str | None = None,
              measure: Measure | None = None, step_s: float = 10.0) -> BanCheck:
    """Rate of a banned class during the ban hours and outside them.

    The recording time is split into time inside and outside the window, so a
    recording from 6:30 to 7:30 compares the half hour before a 7:00 ban with
    the half hour after it.
    """
    if (start_clock is None and "clock_time" in crossings.columns and len(crossings)
            and pd.notna(crossings.iloc[0]["clock_time"])):
        first = crossings.iloc[0]
        start_clock = (pd.to_datetime(first["clock_time"])
                       - timedelta(seconds=float(first["video_time_s"]))).isoformat()
    if start_clock is None:
        raise ValueError("Clock time unknown: run with --start-clock, or pass start_clock.")
    t0 = datetime.fromisoformat(start_clock)
    steps = np.arange(0, duration_s, step_s)
    inside = np.array([window.contains((t0 + timedelta(seconds=float(s))).time()) for s in steps])
    widths = np.minimum(step_s, duration_s - steps)
    h_in = float(widths[inside].sum()) / 3600
    h_out = float(widths[~inside].sum()) / 3600

    sub = crossings[crossings["road_class"].isin(banned)]
    if len(sub):
        ct = clock_times(sub, start_clock)
        flags = np.array([window.contains(t.time()) for t in ct])
    else:
        flags = np.array([], dtype=bool)
    n_in, n_out = int(flags.sum()), int((~flags).sum())

    def rate(n, h):
        if h <= 0:
            return None, None
        lo, hi = poisson_ci(n)
        return n / h, (lo / h, hi / h)

    r_in, ci_in = rate(n_in, h_in)
    r_out, ci_out = rate(n_out, h_out)
    in_period = measure.in_period(t0) if measure else None
    return BanCheck(list(banned), str(window), h_in, h_out, n_in, n_out, r_in, ci_in, r_out,
                    ci_out, in_period)


# ---------------------------------------------------------------------------
# Banned movements and promised service frequencies
# ---------------------------------------------------------------------------

@dataclass
class RateCheck:
    label: str
    n: int
    hours: float
    rate: float
    rate_ci: tuple[float, float]
    expected: float | None = None

    def as_text(self) -> str:
        lo, hi = self.rate_ci
        text = (f"{self.label}: {self.n} in {self.hours * 60:.0f} min, {self.rate:.1f} per hour "
                f"(95% CI {lo:.1f} to {hi:.1f})")
        if self.expected is not None:
            if hi < self.expected:
                verdict = "fewer than promised: the interval is below the promised rate"
            elif lo > self.expected:
                verdict = "more than promised"
            else:
                verdict = "compatible with the promised rate"
            text += f"; promised {self.expected:g} per hour, {verdict}"
        return text + "."


def movement_check(crossings: pd.DataFrame, from_line: str, to_line: str,
                   duration_s: float, classes: list[str] | None = None) -> RateCheck:
    """Vehicles per hour making the movement from one line to another."""
    from .analysis import movements

    mv = movements(crossings)
    if classes:
        mv = mv[mv["road_class"].isin(classes)]
    n = int((mv["movement"] == f"{from_line}>{to_line}").sum()) if len(mv) else 0
    hours = max(duration_s / 3600, 1e-9)
    lo, hi = poisson_ci(n)
    return RateCheck(f"movement {from_line} > {to_line}", n, hours, n / hours,
                     (lo / hours, hi / hours))


def frequency_check(crossings: pd.DataFrame, classes: list[str], expected_per_hour: float,
                    duration_s: float, line: str | None = None) -> RateCheck:
    """Observed vehicles of a service per hour against the promised rate."""
    sel = crossings[crossings["road_class"].isin(classes)]
    if line is not None and "line" in sel.columns:
        sel = sel[sel["line"] == line]
    n = len(sel)
    hours = max(duration_s / 3600, 1e-9)
    lo, hi = poisson_ci(n)
    return RateCheck("/".join(classes) + " passing", n, hours, n / hours,
                     (lo / hours, hi / hours), expected_per_hour)


def check_measure(m: Measure, run_dir: str | Path, zone: str | None = None,
                  interval_s: float = 60.0, start_clock: str | None = None) -> str:
    """Run the check that fits a measure on a run folder; return a report."""
    import json

    run = Path(run_dir)
    summary = json.loads((run / "summary.json").read_text())
    duration = summary["video"]["duration_s"]
    start = start_clock or summary["config"].get("start_clock")
    cx = pd.read_csv(run / "crossings.csv")
    zf = run / "zone_events.csv"
    zev = pd.read_csv(zf) if zf.exists() else None

    zones_in_run = sorted((summary["config"].get("zones") or {}).keys())
    wanted = zone or m.zone
    if wanted and m.kind in ("reserved_lane", "class_ban") and wanted not in zones_in_run:
        if m.kind == "reserved_lane" or m.zone:
            have = ", ".join(zones_in_run) or "none"
            raise ValueError(f"Zone '{wanted}' is not in this run (zones: {have}). Pass "
                             f"--zone <name> or run the video with a zone named '{wanted}'.")
    if m.kind == "class_ban" and m.window is not None and start is None and \
            ("clock_time" not in cx.columns or cx["clock_time"].isna().all()):
        raise ValueError(f"This ban applies {m.window}: give the clock time of the first "
                         "frame with --start-clock, e.g. 2026-11-07T22:30:00.")
    if m.kind == "reserved_lane":
        if zev is None:
            raise ValueError(f"No zone_events.csv in {run}: run the video with a zone "
                             f"named '{m.zone}'.")
        return lane_use(zev, zone or m.zone, m.allowed_classes, duration, interval_s).as_text()
    if m.kind == "class_ban":
        events = cx
        if m.zone or zone:
            if zev is None:
                raise ValueError(f"This ban applies inside a zone: run with a zone named "
                                 f"'{m.zone}'.")
            events = zev[zev["zone"] == (zone or m.zone)]
        if m.window is None:
            sel = events[events["road_class"].isin(m.banned_classes)]
            hours = max(duration / 3600, 1e-9)
            lo, hi = poisson_ci(len(sel))
            return (RateCheck("/".join(m.banned_classes), len(sel), hours, len(sel) / hours,
                              (lo / hours, hi / hours)).as_text()
                    + " The ban applies all day: compare with a recording from before the "
                    "ban, or from a control area, with `traffic-vision did`.")
        return ban_check(events, m.banned_classes, m.window, duration, start, m).as_text()
    if m.kind == "movement_ban":
        return movement_check(cx, m.lines[0], m.lines[1], duration,
                              m.banned_classes or None).as_text()
    if m.kind == "service_frequency":
        return frequency_check(cx, m.allowed_classes, m.expected_per_hour, duration).as_text()
    return ("A closure is measured on the diversion routes, against a control street, with "
            "`traffic-vision did` (four recordings: treated and control, before and during).")

