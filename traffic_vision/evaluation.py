"""Estimate the effect of a measure from video counts, with honest uncertainty.

Comparing one site before and during a measure mixes two things: the measure
itself and everything else that changed in the city at the same time. During
the Dakar 2026 Youth Olympic Games, traffic changed everywhere (visitors,
shuttles, school calendars, security perimeters). A site that the measure did
not touch, filmed over the same periods, gives the counterfactual. This is a
difference-in-differences (DiD):

    effect = (treated_during - treated_before) - (control_during - control_before)

Two problems are specific to counts produced by a detector:

1. **The detector misses objects.** If it finds 85% of motorbikes, every
   difference in raw counts shrinks by 15%. A short manual count of the same
   video gives a calibration factor (manual total / automatic total) per class.
   :func:`calibrate` estimates it, and :func:`diff_in_diff` carries its
   uncertainty into the effect.
2. **Traffic is autocorrelated.** A jam lasts more than one interval, so
   intervals are not independent and an ordinary bootstrap gives intervals that
   are too narrow (Bertrand, Duflo and Mullainathan, 2004). The default
   interval cuts each recording into a few long batches of consecutive
   intervals (batch means), whose averages are close to independent, and uses a
   Student t critical value with Welch-Satterthwaite degrees of freedom. A
   moving block bootstrap (Kunsch, 1989) is available as an alternative.

``scripts/simulate_evaluation.py`` checks by simulation that this estimator
recovers a known effect and how often its 95% intervals contain the truth.
With two-hour recordings and autocorrelated traffic, the block bootstrap
covered the truth in only about 80% of simulated studies; batch means brought
coverage back to about 92% (two hours) and 95% (six hours). See
``docs/results/simulation.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t distribution (numerical inversion, no scipy)."""
    if not 0.5 < p < 1:
        raise ValueError("p must be between 0.5 and 1")
    if df > 1e6:
        df = 1e6
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)

    def cdf(x: float) -> float:
        grid = np.linspace(0.0, x, 4001)
        dens = np.exp(log_c - (df + 1) / 2 * np.log1p(grid ** 2 / df))
        h = grid[1] - grid[0]
        simpson = h / 3 * (dens[0] + dens[-1] + 4 * dens[1:-1:2].sum() + 2 * dens[2:-1:2].sum())
        return 0.5 + simpson

    lo, hi = 0.0, 1.0
    while cdf(hi) < p:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def batch_means(x, n_batches: int) -> np.ndarray:
    """Means of ``n_batches`` consecutive, equal-sized parts of a series."""
    x = np.asarray(x, dtype=float)
    if len(x) < n_batches:
        raise ValueError(f"Need at least {n_batches} intervals for {n_batches} batches.")
    return np.array([b.mean() for b in np.array_split(x, n_batches)])


def default_block(n: int) -> int:
    """Block length about n^(1/3), a common rule of thumb for the block bootstrap."""
    return max(1, int(round(n ** (1 / 3))))


def block_bootstrap_means(x, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Means of ``n_boot`` circular moving-block resamples of the series ``x``.

    ``block=1`` gives the ordinary (independent) bootstrap.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("Empty series.")
    block = max(1, min(block, n))
    k = math.ceil(n / block)
    starts = rng.integers(0, n, (n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)) % n
    idx = idx.reshape(n_boot, k * block)[:, :n]
    return x[idx].mean(axis=1)


# ---------------------------------------------------------------------------
# Calibration against a manual count
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    road_class: str
    manual: np.ndarray  # manual count per validation window
    auto: np.ndarray    # automatic count for the same windows
    factor: float       # sum(manual) / sum(auto)
    factor_ci: tuple[float, float]
    relative_error: float  # (sum(auto) - sum(manual)) / sum(manual)

    def as_text(self) -> str:
        return (f"{self.road_class}: manual {int(self.manual.sum())}, automatic "
                f"{int(self.auto.sum())} over {len(self.manual)} windows; error "
                f"{self.relative_error:+.0%}; correction factor {self.factor:.2f} "
                f"(95% CI {self.factor_ci[0]:.2f} to {self.factor_ci[1]:.2f})")


def calibrate(manual, auto, road_class: str = "all", n_boot: int = 5000,
              seed: int = 0) -> Calibration:
    """Correction factor for automatic counts, from paired validation windows.

    ``manual[i]`` and ``auto[i]`` are the manual and automatic counts of the
    same class in the same window of video (for example 10 windows of one
    minute). The factor is a ratio of totals; its interval resamples windows.
    """
    m = np.asarray(manual, dtype=float)
    a = np.asarray(auto, dtype=float)
    if len(m) != len(a) or len(m) < 2:
        raise ValueError("Need at least two paired validation windows.")
    if a.sum() <= 0 or m.sum() <= 0:
        raise ValueError(f"{road_class}: counts are zero, no factor can be estimated.")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(m), (n_boot, len(m)))
    den = a[pick].sum(axis=1)
    ok = den > 0
    boot = m[pick].sum(axis=1)[ok] / den[ok]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Calibration(road_class, m, a, float(m.sum() / a.sum()), (float(lo), float(hi)),
                       float((a.sum() - m.sum()) / m.sum()))


def calibration_from_files(manual_csv: str | Path, run_dir: str | Path) -> dict[str, Calibration]:
    """Calibrate a run from a manual count file.

    The file has one row per window and class: ``start_s, end_s, road_class,
    manual_count`` (an optional ``direction`` column restricts the comparison
    to one direction). Automatic counts are taken from the run's crossings in
    the same windows.
    """
    man = pd.read_csv(manual_csv)
    cx = pd.read_csv(Path(run_dir) / "crossings.csv")
    rows = []
    for _, r in man.iterrows():
        sel = ((cx["video_time_s"] >= r["start_s"]) & (cx["video_time_s"] < r["end_s"])
               & (cx["road_class"] == r["road_class"]))
        if "direction" in man.columns and pd.notna(r.get("direction")):
            sel &= cx["direction"] == r["direction"]
        rows.append({"road_class": r["road_class"], "manual": r["manual_count"],
                     "auto": int(sel.sum())})
    df = pd.DataFrame(rows)
    return {c: calibrate(g["manual"], g["auto"], c) for c, g in df.groupby("road_class")}


# ---------------------------------------------------------------------------
# Difference-in-differences
# ---------------------------------------------------------------------------

@dataclass
class DiDResult:
    effect: float                  # per interval, in (corrected) objects
    ci: tuple[float, float]
    relative: float | None         # effect / counterfactual mean of the treated site
    relative_ci: tuple[float, float] | None
    means: dict[str, float]
    method: str                    # "batch" or "block"
    detail: str                    # batches and degrees of freedom, or block length
    factor: float | None           # calibration factor applied, if any

    def as_text(self, unit: str = "objects per interval") -> str:
        lo, hi = self.ci
        rel = ""
        if self.relative is not None and self.relative_ci is not None:
            rel = (f" ({self.relative:+.0%} of what the site would have had without the measure, "
                   f"95% CI {self.relative_ci[0]:+.0%} to {self.relative_ci[1]:+.0%})")
        cal = f", counts corrected by a factor {self.factor:.2f}" if self.factor else ""
        verdict = ("the interval excludes 0" if lo > 0 or hi < 0
                   else "the interval includes 0: no clear effect at this sample size")
        m = self.means
        return (f"Means: treated {m['treated_before']:.1f} -> {m['treated_during']:.1f}, "
                f"control {m['control_before']:.1f} -> {m['control_during']:.1f} {unit}. "
                f"Effect of the measure: {self.effect:+.2f} {unit}{rel}; 95% CI "
                f"[{lo:+.2f}, {hi:+.2f}] ({self.detail}{cal}); {verdict}.")


def _ratio_variance(manual: np.ndarray, auto: np.ndarray) -> float:
    """Variance of a ratio-of-totals estimator, by linearisation (Cochran)."""
    n = len(manual)
    r = manual.sum() / auto.sum()
    resid = manual - r * auto
    return float(resid.var(ddof=1) / n / auto.mean() ** 2)


def diff_in_diff(treated_before, treated_during, control_before, control_during,
                 method: str = "batch", n_batches: int = 4, n_boot: int = 10_000,
                 block: int | None = None, seed: int = 0,
                 calibration: Calibration | None = None) -> DiDResult:
    """Difference-in-differences on flows per interval.

    ``method="batch"`` (default): each recording is cut into ``n_batches``
    consecutive batches; the variance of each mean comes from the batch means,
    and the interval uses a Student t value with Welch-Satterthwaite degrees
    of freedom. With ``calibration``, the uncertainty of the correction factor
    is added by the delta method.

    ``method="block"``: circular moving block bootstrap on each series (blocks
    of ``block`` intervals, default n^(1/3)); with ``calibration`` the
    validation windows are resampled too.

    The identifying assumption is that, without the measure, the treated site
    would have changed like the control site (parallel trends). Choose a
    control site on a similar road, outside the measure's reach.
    """
    series = [np.asarray(s, dtype=float) for s in
              (treated_before, treated_during, control_before, control_during)]
    if min(len(s) for s in series) < 2:
        raise ValueError("Each series needs at least two intervals.")
    tb, td, cb, cd = (s.mean() for s in series)
    raw = (td - tb) - (cd - cb)
    counterfactual = tb + (cd - cb)
    k = calibration.factor if calibration is not None else 1.0
    means = {"treated_before": tb * k, "treated_during": td * k,
             "control_before": cb * k, "control_during": cd * k}
    rel = raw / counterfactual if counterfactual > 0 else None  # the factor cancels

    if method == "batch":
        m = min(n_batches, min(len(s) for s in series))
        if m < 2:
            raise ValueError("Batch means need at least two batches.")
        v = [batch_means(s, m).var(ddof=1) / m for s in series]
        var_raw = sum(v)
        if var_raw > 0:
            df = var_raw ** 2 / sum(vi ** 2 / (m - 1) for vi in v)
        else:
            df = float(m - 1)
        var_eff = k ** 2 * var_raw
        if calibration is not None:
            var_eff += raw ** 2 * _ratio_variance(calibration.manual, calibration.auto)
        tq = t_quantile(0.975, df)
        half = tq * math.sqrt(var_eff)
        ci = (k * raw - half, k * raw + half)
        rel_ci = None
        if rel is not None:
            var_c = v[0] + v[2] + v[3]
            var_rel = (var_raw / counterfactual ** 2 + raw ** 2 * var_c / counterfactual ** 4
                       + 2 * raw * var_c / counterfactual ** 3)
            h = tq * math.sqrt(max(var_rel, 0.0))
            rel_ci = (rel - h, rel + h)
        detail = f"batch means, {m} batches per recording, t with {df:.1f} df"
        return DiDResult(float(k * raw), ci, rel, rel_ci, means, "batch", detail,
                         calibration.factor if calibration else None)

    if method != "block":
        raise ValueError("method must be 'batch' or 'block'")
    b = block or default_block(min(len(s) for s in series))
    rng = np.random.default_rng(seed)
    boots = [block_bootstrap_means(s, n_boot, b, rng) for s in series]
    raw_b = (boots[1] - boots[0]) - (boots[3] - boots[2])
    cf_b = boots[0] + (boots[3] - boots[2])
    k_b = np.ones(n_boot)
    if calibration is not None:
        mm, aa = calibration.manual, calibration.auto
        pick = rng.integers(0, len(mm), (n_boot, len(mm)))
        den = aa[pick].sum(axis=1)
        k_b = np.where(den > 0, mm[pick].sum(axis=1) / np.where(den > 0, den, 1), k)
    lo, hi = np.percentile(k_b * raw_b, [2.5, 97.5])
    rel_ci = None
    if rel is not None:
        ok = cf_b > 0
        rel_ci = tuple(float(x) for x in np.percentile(raw_b[ok] / cf_b[ok], [2.5, 97.5]))
    return DiDResult(float(k * raw), (float(lo), float(hi)), rel, rel_ci, means, "block",
                     f"block bootstrap, blocks of {b} intervals",
                     calibration.factor if calibration else None)
