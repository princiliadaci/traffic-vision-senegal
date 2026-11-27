"""Simulated studies, to check the estimators and to plan a real study.

A study films a treated site and a control site, before and during a measure,
and counts vehicles per interval. Here the truth is known:

- traffic at each site has a level, a city-wide change during the measure
  (``city_shock``) and the effect of the measure on the treated site only;
- traffic is autocorrelated: a log-normal fluctuation that follows an AR(1)
  process (a jam lasts several intervals), with Poisson counts on top;
- the detector finds each vehicle with some probability (``recall``, which can
  differ between recordings: another camera, a crowd, the night) and adds a
  few false detections;
- a person counts some intervals by hand.

A second kind of study checks a ban on one class (trucks, motorbikes) when
the detector confuses another local vehicle with the banned class, as both
detectors of ``docs/results/open_vocabulary_probe.md`` did with the car
rapide, labelled a truck (``BanScenario``, ``ban_run``).

Used by ``scripts/simulate_evaluation.py`` (method comparison) and by
``traffic-vision design`` (how long to film).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import compare_periods
from .evaluation import calibrate, diff_in_diff, ppi_diff_in_diff, ppi_mean, t_quantile

SERIES = ("tb", "td", "cb", "cd")  # treated before/during, control before/during


@dataclass
class Scenario:
    hours: float = 2.0                # recording length per series
    interval_min: float = 5.0
    days: int = 1                     # independent recordings per series, concatenated
    base_treated: float = 60.0        # vehicles per interval before
    base_control: float = 50.0
    city_shock: float = 6.0           # change everywhere during the measure
    effect: float = 12.0              # true effect on the treated site
    rho: float = 0.6                  # autocorrelation of the fluctuation
    sigma: float = 0.15               # size of the fluctuation (log scale)
    recall: dict[str, float] = field(default_factory=lambda: dict.fromkeys(SERIES, 0.82))
    false_pos: float = 1.0            # false detections per interval
    n_ratio: int = 8                  # hand-counted intervals, treated before only
    n_ppi: int = 4                    # hand-counted intervals in each recording

    @property
    def per_day(self) -> int:
        return max(2, int(round(self.hours * 60 / self.interval_min)))

    def means(self) -> dict[str, float]:
        return {"tb": self.base_treated,
                "td": self.base_treated + self.city_shock + self.effect,
                "cb": self.base_control, "cd": self.base_control + self.city_shock}


def _ar1(rng, n, rho, sigma):
    e = np.empty(n)
    sd0 = sigma / np.sqrt(1 - rho ** 2)
    e[0] = rng.normal(0, sd0)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sigma)
    return e - sd0 ** 2 / 2  # mean of exp(e) close to 1


def series(rng, sc: Scenario, mean: float, recall: float):
    """True and detected counts for one site and period (all days concatenated)."""
    true = np.concatenate([rng.poisson(mean * np.exp(_ar1(rng, sc.per_day, sc.rho, sc.sigma)))
                           for _ in range(sc.days)])
    detected = rng.binomial(true, recall) + rng.poisson(sc.false_pos, len(true))
    return true, detected


def one_study(rng, sc: Scenario, n_boot: int = 999, which: tuple[str, ...] | None = None):
    """Estimates and 95% intervals of the effect for each estimator."""
    data = {k: series(rng, sc, m, sc.recall[k]) for k, m in sc.means().items()}
    obs = [data[k][1] for k in SERIES]
    seed = int(rng.integers(1 << 31))
    out = {}
    want = (lambda name: which is None or name in which)

    if want("before_after"):
        c = compare_periods(obs[0], obs[1], n_boot=n_boot, seed=seed)
        out["before_after"] = (c.difference, c.ci_low, c.ci_high)
    if want("did_raw"):
        r = diff_in_diff(*obs)
        out["did_raw"] = (r.effect, *r.ci)
    true_tb, obs_tb = data["tb"]
    pick = rng.choice(len(true_tb), size=min(sc.n_ratio, len(true_tb)), replace=False)
    cal = calibrate(true_tb[pick], obs_tb[pick], n_boot=200)
    if want("did_ratio_iid"):
        r = diff_in_diff(*obs, method="block", block=1, n_boot=n_boot, seed=seed, calibration=cal)
        out["did_ratio_iid"] = (r.effect, *r.ci)
    if want("did_ratio_block"):
        r = diff_in_diff(*obs, method="block", n_boot=n_boot, seed=seed, calibration=cal)
        out["did_ratio_block"] = (r.effect, *r.ci)
    if want("did_ratio_batch"):
        r = diff_in_diff(*obs, calibration=cal)
        out["did_ratio_batch"] = (r.effect, *r.ci)
    for name, lam in (("did_ppi", 1.0), ("did_ppi_ratio", "ratio"), ("did_ppi_tuned", "tuned")):
        if not want(name):
            continue
        ests = []
        for k in SERIES:
            true, det = data[k]
            idx = rng.choice(len(true), size=min(sc.n_ppi, len(true)), replace=False)
            ests.append(ppi_mean(det, det[idx], true[idx], lam=lam))
        r = ppi_diff_in_diff(ests)
        out[name] = (r.effect, *r.ci)
    return out


def run(sc: Scenario, n_studies: int = 1000, seed: int = 2026, n_boot: int = 999,
        which: tuple[str, ...] | None = None) -> dict[str, dict[str, float]]:
    """Bias, RMSE, coverage, interval width and detection rate per estimator."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {}
    for _ in range(n_studies):
        for k, v in one_study(rng, sc, n_boot, which).items():
            rows.setdefault(k, []).append(v)
    table = {}
    for k, r in rows.items():
        a = np.array(r)
        est, lo, hi = a[:, 0], a[:, 1], a[:, 2]
        table[k] = {"mean_estimate": float(est.mean()), "bias": float(est.mean() - sc.effect),
                    "rmse": float(np.sqrt(((est - sc.effect) ** 2).mean())),
                    "coverage": float(((lo <= sc.effect) & (sc.effect <= hi)).mean()),
                    "mean_ci_width": float((hi - lo).mean()),
                    "detects_effect": float(((lo > 0) | (hi < 0)).mean())}
    return table


def power_table(base: float, effect_share: float, hours_options=(2, 4, 8),
                days_options=(1, 3, 7), rho: float = 0.6, sigma: float = 0.15,
                recall: float = 0.82, interval_min: float = 5.0, n_studies: int = 300,
                seed: int = 7) -> list[dict]:
    """Chance of detecting an effect, by hours per day and days per period.

    ``base`` is the treated site's flow per interval before the measure and
    ``effect_share`` the expected change as a share of it (0.2 = +20%). The
    control site is assumed similar. Uses the default estimator (calibrated
    counts, batch means).
    """
    out = []
    for days in days_options:
        for hours in hours_options:
            sc = Scenario(hours=hours, days=days, interval_min=interval_min, base_treated=base,
                          base_control=base, city_shock=0.1 * base, effect=effect_share * base,
                          rho=rho, sigma=sigma, recall=dict.fromkeys(SERIES, recall),
                          false_pos=0.02 * base)
            t = run(sc, n_studies, seed, which=("did_ratio_batch",))["did_ratio_batch"]
            out.append({"days": days, "hours_per_day": hours,
                        "power": t["detects_effect"], "coverage": t["coverage"],
                        "ci_half_width": t["mean_ci_width"] / 2})
    return out


# ---------------------------------------------------------------------------
# A ban on one class, with a look-alike vehicle the detector confuses with it
# ---------------------------------------------------------------------------

@dataclass
class BanScenario:
    """Hours of a truck ban at one site.

    ``banned`` vehicles of the banned class still pass per interval (offences,
    exemptions); ``lookalikes`` are vehicles of another class that resemble
    it (car rapides for trucks); the detector labels a share ``confusion`` of
    them as the banned class.
    """
    hours: float = 4
    interval_min: float = 5
    days: int = 1
    banned: float = 1.0
    lookalikes: float = 8.0
    recall: float = 0.85
    confusion: float = 0.5
    false_pos: float = 0.1
    rho: float = 0.6
    sigma: float = 0.15
    n_labels: int = 6

    @property
    def per_day(self) -> int:
        return max(2, int(round(self.hours * 60 / self.interval_min)))


def ban_study(rng, sc: BanScenario) -> dict[str, tuple[float, float, float]]:
    """Mean banned vehicles per interval: network alone, and prediction-powered."""
    from .rules import poisson_ci

    n = sc.per_day * sc.days
    level = np.concatenate([np.exp(_ar1(rng, sc.per_day, sc.rho, sc.sigma))
                            for _ in range(sc.days)])
    true = rng.poisson(sc.banned * level)
    look = rng.poisson(sc.lookalikes * level)
    det = (rng.binomial(true, sc.recall) + rng.binomial(look, sc.confusion)
           + rng.poisson(sc.false_pos, n))
    lo, hi = poisson_ci(int(det.sum()))
    out = {"network": (float(det.mean()), lo / n, hi / n)}
    idx = rng.choice(n, size=min(sc.n_labels, n), replace=False)
    for name, lam in (("ppi", 1.0), ("ppi_ratio", "ratio")):
        e = ppi_mean(det, det[idx], true[idx], lam=lam)
        h = t_quantile(0.975, e.df) * e.se
        out[name] = (e.mean, e.mean - h, e.mean + h)
    return out


def ban_run(sc: BanScenario, n_studies: int = 1000, seed: int = 2026) -> dict:
    """Bias, coverage and width of each estimate of the banned-class rate."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {}
    for _ in range(n_studies):
        for k, v in ban_study(rng, sc).items():
            rows.setdefault(k, []).append(v)
    truth = sc.banned  # the AR(1) level has mean 1
    per_hour = 60 / sc.interval_min
    table = {}
    for k, r in rows.items():
        a = np.array(r)
        est, lo, hi = a[:, 0], a[:, 1], a[:, 2]
        table[k] = {"mean_per_hour": float(est.mean() * per_hour),
                    "true_per_hour": float(truth * per_hour),
                    "coverage": float(((lo <= truth) & (truth <= hi)).mean()),
                    "mean_ci_width_per_hour": float((hi - lo).mean() * per_hour)}
    return table
