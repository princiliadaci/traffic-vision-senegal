"""Estimate the effect of a measure from video counts, with honest uncertainty.

Comparing one site before and during a measure mixes two things: the measure
itself and everything else that changed in the city at the same time (during
the Dakar 2026 Youth Olympic Games: visitors, shuttles, school calendars,
security perimeters; after a ban: fuel prices, the season, the school year). A site that the measure did
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

3. **Cameras differ.** A correction factor estimated on one camera does not
   hold for another camera, another light or a crowd that hides vehicles. With
   a few hand-counted windows in *each* recording, prediction-powered
   inference (Angelopoulos et al., 2023; PPI++, Angelopoulos, Duchi and Zrnic,
   2023) combines the network's counts on every window with the human counts
   on a few, and gives valid intervals whatever the accuracy of the network
   (:func:`ppi_mean`, :func:`ppi_diff_in_diff`).

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


# ---------------------------------------------------------------------------
# Prediction-powered inference
# ---------------------------------------------------------------------------

def _welch_df(variances: list[float], dfs: list[float]) -> float:
    num = sum(variances) ** 2
    den = sum(v ** 2 / d for v, d in zip(variances, dfs) if d > 0)
    return num / den if den > 0 else float(min(dfs))


@dataclass
class PPIEstimate:
    mean: float          # estimated mean of the true count per window
    se: float
    df: float
    lam: float           # weight given to the network's counts (1 = plain PPI)
    n_labeled: int
    n_windows: int
    naive_mean: float    # mean of the network's counts alone


def ppi_mean(pred_all, pred_labeled, true_labeled, n_batches: int = 4,
             lam: float | str = "ratio") -> PPIEstimate:
    """Mean true count per window from network counts and a few human counts.

    ``pred_all`` holds the network's count in every window of a recording;
    ``pred_labeled`` and ``true_labeled`` the network's and a person's count in
    a few windows chosen at random. The estimate is

        mean(true_labeled) + lam * (mean(pred_all) - mean(pred_labeled))

    ``lam`` sets how much the network's counts are trusted:

    - ``"ratio"`` (default): ``lam`` is the ratio of human to network counts
      on the labelled windows, a correction proportional to the count, which
      fits a detector that misses a share of the vehicles. It is the ratio
      estimator of survey sampling, written in the PPI form. In the
      simulation study its intervals kept their 95% coverage when cameras
      differed (``docs/results/simulation.md``);
    - ``1``: plain prediction-powered inference (PPI), the network's average
      corrected by its average error on the labelled windows (slightly
      narrower intervals, coverage about 91-92% with six labels);
    - ``"tuned"``: the variance-minimising weight of PPI++, estimated on the
      labelled windows; unstable with only a few labels.

    The variance of the network's mean uses batch means, because traffic is
    autocorrelated; the variance of the labelled part treats the windows as a
    random sample of the recording.
    """
    f_all = np.asarray(pred_all, dtype=float)
    f_l = np.asarray(pred_labeled, dtype=float)
    y = np.asarray(true_labeled, dtype=float)
    n, n_all = len(y), len(f_all)
    if n < 3 or len(f_l) != n:
        raise ValueError("Need at least three labelled windows, with a network count for each.")
    m = min(n_batches, n_all)
    var_f_mean = batch_means(f_all, m).var(ddof=1) / m if m >= 2 else f_all.var(ddof=1) / n_all
    var_f = f_all.var(ddof=1)
    if lam == "tuned":
        cov = float(np.cov(y, f_l, ddof=1)[0, 1])
        lam = cov / ((1 + n / n_all) * var_f) if var_f > 0 else 0.0
        lam = float(np.clip(lam, 0.0, 3.0))
    elif lam == "ratio":
        lam = float(y.sum() / f_l.sum()) if f_l.sum() > 0 else 0.0
    else:
        lam = float(lam)
    theta = y.mean() + lam * (f_all.mean() - f_l.mean())
    v_unl = lam ** 2 * var_f_mean
    v_lab = (y - lam * f_l).var(ddof=1) / n
    df = _welch_df([v_unl, v_lab], [m - 1, n - 1])
    return PPIEstimate(float(theta), float(math.sqrt(v_unl + v_lab)), float(df), lam, n, n_all,
                       float(f_all.mean()))


def ppi_diff_in_diff(estimates: list[PPIEstimate]) -> DiDResult:
    """Difference-in-differences from four prediction-powered means.

    ``estimates`` are, in order: treated before, treated during, control
    before, control during. Each recording has its own labelled windows, so a
    camera that sees worse than another, or a crowd that hides vehicles
    during the measure, does not bias the effect.
    """
    if len(estimates) != 4:
        raise ValueError("Give four estimates: treated before, treated during, control "
                         "before, control during.")
    tb, td, cb, cd = (e.mean for e in estimates)
    effect = (td - tb) - (cd - cb)
    variances = [e.se ** 2 for e in estimates]
    df = _welch_df(variances, [e.df for e in estimates])
    half = t_quantile(0.975, df) * math.sqrt(sum(variances))
    counterfactual = tb + (cd - cb)
    rel = effect / counterfactual if counterfactual > 0 else None
    rel_ci = None
    if rel is not None:
        var_c = variances[0] + variances[2] + variances[3]
        var_rel = (sum(variances) / counterfactual ** 2 + effect ** 2 * var_c / counterfactual ** 4
                   + 2 * effect * var_c / counterfactual ** 3)
        h = t_quantile(0.975, df) * math.sqrt(max(var_rel, 0.0))
        rel_ci = (rel - h, rel + h)
    lams = ", ".join(f"{e.lam:.2f}" for e in estimates)
    detail = (f"prediction-powered inference, {sum(e.n_labeled for e in estimates)} labelled "
              f"windows, weights {lams}, t with {df:.1f} df")
    return DiDResult(float(effect), (effect - half, effect + half), rel, rel_ci,
                     {"treated_before": tb, "treated_during": td, "control_before": cb,
                      "control_during": cd}, "ppi", detail, None)


def ppi_from_files(labels_csv: str | Path, run_dirs: list[str | Path],
                   classes: list[str] | None = None, n_batches: int = 4,
                   lam: float | str = "ratio") -> list[PPIEstimate]:
    """Prediction-powered means for several runs from one label file.

    The label file has one row per hand-counted window: ``run, start_s, end_s,
    manual_count`` (``run`` is the run folder name). All windows of a run must
    have the same length; the network's counts are computed on consecutive
    windows of that length over the whole recording.
    """
    import json

    labels = pd.read_csv(labels_csv)
    out = []
    for rd in run_dirs:
        rd = Path(rd)
        lab = labels[labels["run"] == rd.name]
        if len(lab) < 3:
            raise ValueError(f"{rd.name}: need at least three labelled windows in {labels_csv}.")
        lengths = (lab["end_s"] - lab["start_s"]).round(3).unique()
        if len(lengths) != 1:
            raise ValueError(f"{rd.name}: labelled windows must all have the same length.")
        w = float(lengths[0])
        dur = json.loads((rd / "summary.json").read_text())["video"]["duration_s"]
        cx = pd.read_csv(rd / "crossings.csv")
        cx = cx[cx["road_class"] != "pedestrian"] if classes is None else \
            cx[cx["road_class"].isin(classes)]
        t = cx["video_time_s"].to_numpy()
        k = int(dur // w)
        pred_all = np.histogram(t, bins=np.arange(k + 1) * w)[0]
        pred_lab = [int(((t >= a) & (t < b)).sum()) for a, b in zip(lab["start_s"], lab["end_s"])]
        out.append(ppi_mean(pred_all, pred_lab, lab["manual_count"].to_numpy(), n_batches, lam))
    return out

