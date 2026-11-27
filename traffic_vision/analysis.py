"""Turn line crossings into traffic indicators and compare two periods.

The question behind this module: did a measure (a reserved lane, a road
closure, a shuttle service during the Dakar 2026 Youth Olympic Games) change
traffic at a given point? To answer it, film the same place, at the same time
of day, before and during the measure, then compare:

- flow per interval (vehicles per minute, by class and direction);
- flow in passenger car equivalents (PCE), so that a bus weighs more than a car;
- the share of public transport (buses) and two-wheelers in the flow;
- the difference between periods, with a bootstrap confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Illustrative weights. Traffic authorities publish their own values: set yours
# with the `pce` argument. Pedestrians are not vehicles and weigh 0.
DEFAULT_PCE = {
    "pedestrian": 0.0,
    "bicycle": 0.3,
    "motorcycle": 0.5,
    "car": 1.0,
    "bus": 2.5,
    "truck": 2.5,
}


def load_crossings(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Read crossings.csv from one or more run folders."""
    frames = []
    for d in run_dirs:
        p = Path(d) / "crossings.csv"
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise ValueError("No run folder given.")
    return pd.concat(frames, ignore_index=True)


def flows(crossings: pd.DataFrame, interval_s: float = 60.0, duration_s: float | None = None,
          by: tuple[str, ...] = ("road_class", "direction")) -> pd.DataFrame:
    """Count crossings per time interval.

    Intervals with no crossing are kept with a count of 0, which matters for
    averages. Pass ``duration_s`` (the video length) so the last empty
    intervals are kept too.
    """
    df = crossings.copy()
    df["interval"] = (df["video_time_s"] // interval_s).astype(int)
    last = int(duration_s // interval_s) if duration_s else int(df["interval"].max()) if len(df) else 0
    n_int = max(last, 0) + 1
    groups = [g for g in by if g in df.columns]
    counts = df.groupby(["interval", *groups]).size().rename("count")
    if groups:
        levels = [range(n_int)] + [sorted(df[g].unique()) for g in groups]
        idx = pd.MultiIndex.from_product(levels, names=["interval", *groups])
    else:
        idx = pd.Index(range(n_int), name="interval")
    out = counts.reindex(idx, fill_value=0).reset_index()
    out["start_s"] = out["interval"] * interval_s
    return out


def pce_flow(crossings: pd.DataFrame, interval_s: float = 60.0, duration_s: float | None = None,
             pce: dict[str, float] | None = None) -> pd.DataFrame:
    """Flow per interval in passenger car equivalents (all classes, all directions)."""
    weights = {**DEFAULT_PCE, **(pce or {})}
    f = flows(crossings, interval_s, duration_s, by=("road_class",))
    f["pce"] = f["count"] * f["road_class"].map(weights).fillna(0)
    return f.groupby(["interval", "start_s"], as_index=False)["pce"].sum()


def modal_shares(crossings: pd.DataFrame) -> pd.Series:
    """Share of each class among vehicles (pedestrians excluded)."""
    veh = crossings[crossings["road_class"] != "pedestrian"]
    if veh.empty:
        return pd.Series(dtype=float)
    return veh["road_class"].value_counts(normalize=True).sort_index()


@dataclass
class Comparison:
    label_a: str
    label_b: str
    mean_a: float
    mean_b: float
    difference: float  # mean_b - mean_a
    relative_change: float | None  # difference / mean_a
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int

    def as_text(self, unit: str = "vehicles/min") -> str:
        rel = f" ({self.relative_change:+.0%})" if self.relative_change is not None else ""
        sig = ("the interval excludes 0: the change is unlikely to be noise"
               if self.ci_low > 0 or self.ci_high < 0
               else "the interval includes 0: no clear change at this sample size")
        return (f"{self.label_a}: {self.mean_a:.2f} {unit} (n={self.n_a}) | "
                f"{self.label_b}: {self.mean_b:.2f} {unit} (n={self.n_b}) | "
                f"difference {self.difference:+.2f}{rel}, 95% CI [{self.ci_low:+.2f}, "
                f"{self.ci_high:+.2f}]; {sig}.")


def compare_periods(values_a, values_b, label_a: str = "before", label_b: str = "during",
                    n_boot: int = 10_000, seed: int = 0) -> Comparison:
    """Compare mean flow per interval between two periods.

    Uses a percentile bootstrap on the difference of means, which does not
    assume normal counts. Intervals are treated as independent, which is
    optimistic when traffic is autocorrelated: prefer intervals of 5 minutes
    or more over long recordings.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each period needs at least two intervals.")
    rng = np.random.default_rng(seed)
    boot = (rng.choice(b, (n_boot, len(b))).mean(axis=1)
            - rng.choice(a, (n_boot, len(a))).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    diff = b.mean() - a.mean()
    rel = diff / a.mean() if a.mean() else None
    return Comparison(label_a, label_b, float(a.mean()), float(b.mean()), float(diff),
                      None if rel is None else float(rel), float(lo), float(hi), len(a), len(b))


def exposure_per_hour(crossings: pd.DataFrame, duration_s: float,
                      classes: tuple[str, ...] = ("pedestrian", "motorcycle", "bicycle")) -> pd.Series:
    """Vulnerable road users crossing per hour, by class.

    Road-safety risk rates divide crashes by exposure (how many people use a
    road). Video counts give that denominator for the users most at risk.
    Short videos give rough rates: scale with care.
    """
    hours = max(duration_s / 3600, 1e-9)
    sub = crossings[crossings["road_class"].isin(classes)]
    counts = sub["road_class"].value_counts().reindex(list(classes), fill_value=0)
    return (counts / hours).round(0)


def aggregate_runs(run_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge several runs (cameras, sites, periods) into one dataset.

    Returns all crossings with their scene id, and one summary row per scene
    with duration, vehicles per minute and counts by class. All runs share the
    same schema, so the merged file can feed a city-wide dashboard.
    """
    import json

    rows, frames = [], []
    for d in run_dirs:
        d = Path(d)
        summary = json.loads((d / "summary.json").read_text())
        cx = pd.read_csv(d / "crossings.csv")
        frames.append(cx)
        dur = summary["video"]["duration_s"]
        veh = cx[cx["road_class"] != "pedestrian"]
        row = {"scene_id": summary["config"]["scene_id"], "duration_s": dur,
               "vehicles_per_min": round(len(veh) / max(dur / 60, 1e-9), 1)}
        row.update(cx["road_class"].value_counts().to_dict())
        rows.append(row)
    table = pd.DataFrame(rows).fillna(0)
    return pd.concat(frames, ignore_index=True), table
