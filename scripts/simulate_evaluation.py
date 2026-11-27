"""Simulation study: can video counts measure the effect of a traffic measure?

We simulate the situation of a closure during the Games: a diversion route
(treated) and a comparable street (control), each filmed before and during the
measure, two hours each, counted per 5 minutes. The truth is known:

- city-wide, traffic changes during the Games (+6 vehicles per 5 min on both
  streets: visitors, shuttles, school calendars);
- the measure adds ``EFFECT`` vehicles per 5 min on the diversion route only;
- traffic is autocorrelated (a jam lasts several intervals);
- the detector misses part of the vehicles and adds a few false detections;
- someone counts 8 intervals of one video by hand, for calibration.

Five estimators are compared on 1,000 simulated studies per scenario:

  E1 before/after on the treated street only, ordinary bootstrap
  E2 difference-in-differences (DiD), raw counts, ordinary bootstrap
  E3 DiD, calibrated counts, ordinary bootstrap
  E4 DiD, calibrated counts, moving block bootstrap
  E5 DiD, calibrated counts, batch means with a t interval (default of the tool)

For each: bias, root mean square error, coverage of the 95% interval (should
be close to 95%), mean interval width, and how often the interval excludes 0
(the chance of detecting the effect). Scenarios cross two recording
lengths (2 and 6 hours per period) with autocorrelated or independent
traffic. Run:

    python scripts/simulate_evaluation.py            # writes docs/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_vision.analysis import compare_periods  # noqa: E402
from traffic_vision.evaluation import calibrate, diff_in_diff  # noqa: E402

T = 24                # 5-minute intervals per recording (2 hours); changed per scenario
BASE_TREATED = 60.0   # vehicles per 5 min on the diversion route, before
BASE_CONTROL = 50.0   # on the control street, before
CITY_SHOCK = 6.0      # change everywhere during the Games
EFFECT = 12.0         # true effect of the measure on the treated street
RECALL = 0.82         # share of vehicles the detector finds
FALSE_POS = 1.0       # false detections per interval
SIGMA = 0.15          # size of traffic fluctuations (log scale)
N_MANUAL = 8          # intervals counted by hand


def ar1(rng, n, rho, sigma):
    e = np.empty(n)
    sd0 = sigma / np.sqrt(1 - rho ** 2) if rho < 1 else sigma
    e[0] = rng.normal(0, sd0)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sigma)
    return e - sd0 ** 2 / 2  # keeps the mean of exp(e) close to 1


def one_series(rng, mean, rho):
    true = rng.poisson(mean * np.exp(ar1(rng, T, rho, SIGMA)))
    observed = rng.binomial(true, RECALL) + rng.poisson(FALSE_POS, T)
    return true, observed


def one_study(rng, rho, n_boot):
    means = {"tb": BASE_TREATED, "td": BASE_TREATED + CITY_SHOCK + EFFECT,
             "cb": BASE_CONTROL, "cd": BASE_CONTROL + CITY_SHOCK}
    data = {k: one_series(rng, m, rho) for k, m in means.items()}
    obs = {k: v[1] for k, v in data.items()}
    true_tb, obs_tb = data["tb"]
    cal = calibrate(true_tb[:N_MANUAL], obs_tb[:N_MANUAL], n_boot=200)
    args = (obs["tb"], obs["td"], obs["cb"], obs["cd"])
    seed = int(rng.integers(1 << 31))
    out = {}
    c = compare_periods(obs["tb"], obs["td"], n_boot=n_boot, seed=seed)
    out["E1"] = (c.difference, c.ci_low, c.ci_high)
    for name, method, block, use_cal in (("E2", "block", 1, False), ("E3", "block", 1, True),
                                         ("E4", "block", None, True), ("E5", "batch", None, True)):
        r = diff_in_diff(*args, method=method, n_boot=n_boot, block=block, seed=seed,
                         calibration=cal if use_cal else None)
        out[name] = (r.effect, r.ci[0], r.ci[1])
    return out


def run(n_studies, rho, n_boot, seed, hours=2):
    global T
    T = int(hours * 12)
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("E1", "E2", "E3", "E4", "E5")}
    for _ in range(n_studies):
        for k, v in one_study(rng, rho, n_boot).items():
            res[k].append(v)
    table = {}
    for k, rows in res.items():
        a = np.array(rows)
        est, lo, hi = a[:, 0], a[:, 1], a[:, 2]
        table[k] = {"mean_estimate": float(est.mean()), "bias": float(est.mean() - EFFECT),
                    "rmse": float(np.sqrt(((est - EFFECT) ** 2).mean())),
                    "coverage": float(((lo <= EFFECT) & (EFFECT <= hi)).mean()),
                    "mean_ci_width": float((hi - lo).mean()),
                    "detects_effect": float(((lo > 0) | (hi < 0)).mean())}
    return table


LABELS = {
    "E1": "Before/after, treated street only",
    "E2": "DiD, raw counts, ordinary bootstrap",
    "E3": "DiD, calibrated counts, ordinary bootstrap",
    "E4": "DiD, calibrated counts, block bootstrap",
    "E5": "DiD, calibrated counts, batch means (default)",
}


def plot_coverage(results: dict, path: Path) -> None:
    """Small multiples: 95% interval coverage per estimator, autocorrelated scenarios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text, muted, grid, series = "#191919", "#666666", "#e5e5e5", "#257de0"
    panels = [k for k in results if "autocorrelated" in k]
    keys = list(LABELS)
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, panels):
        cov = [results[name][k]["coverage"] * 100 for k in keys]
        y = list(range(len(keys)))[::-1]
        ax.axvline(95, color=muted, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.hlines(y, 50, cov, color=grid, lw=2, zorder=1)
        ax.scatter(cov, y, s=70, color=series, edgecolor="white", linewidth=2, zorder=3)
        for xi, yi in zip(cov, y):
            left = xi > 88  # keep the label clear of the 95% reference line
            ax.text(xi - 1.4 if left else xi + 1.4, yi + (0.28 if left else 0), f"{xi:.0f}%",
                    va="center", ha="right" if left else "left", fontsize=9, color=text)
        ax.set_xlim(50, 102)
        ax.set_title(name.split(",")[0], fontsize=10, color=text, loc="left")
        ax.tick_params(colors=muted, labelsize=9, length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_xlabel("share of 95% intervals that contain the true effect", color=muted,
                      fontsize=9)
        ax.text(95.6, len(keys) - 0.45, "target 95%", color=muted, fontsize=8)
    axes[0].set_yticks(list(range(len(keys)))[::-1])
    axes[0].set_yticklabels([LABELS[k] for k in keys], color=text, fontsize=9)
    fig.suptitle("Honest uncertainty needs calibration and batch means when traffic is "
                 "autocorrelated (1,000 simulated studies each)", fontsize=10.5, color=text,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "2-hour recordings, autocorrelated traffic (rho = 0.6)": (0.6, 2),
        "6-hour recordings, autocorrelated traffic (rho = 0.6)": (0.6, 6),
        "2-hour recordings, independent intervals (rho = 0)": (0.0, 2),
        "6-hour recordings, independent intervals (rho = 0)": (0.0, 6),
    }
    results = {name: run(a.studies, rho, a.boot, a.seed, hours)
               for name, (rho, hours) in scenarios.items()}
    (out / "simulation.json").write_text(json.dumps(
        {"settings": {"recording_hours": [2, 6], "interval_minutes": 5,
                      "base_treated": BASE_TREATED, "base_control": BASE_CONTROL,
                      "city_shock": CITY_SHOCK, "true_effect": EFFECT, "recall": RECALL,
                      "false_positives_per_interval": FALSE_POS, "sigma": SIGMA,
                      "manual_intervals": N_MANUAL, "studies": a.studies,
                      "bootstrap_resamples": a.boot, "seed": a.seed},
         "results": results}, indent=2))

    lines = ["# Simulation study: results", "",
             f"{a.studies} simulated studies per scenario; true effect +{EFFECT:g} vehicles per "
             f"5 min; detector recall {RECALL:.0%}, {FALSE_POS:g} false detection per interval; "
             f"{N_MANUAL} intervals counted by hand. Produced by "
             "`scripts/simulate_evaluation.py`.", ""]
    for name, table in results.items():
        lines += [f"## {name[0].upper() + name[1:]}", "",
                  "| estimator | mean estimate | bias | RMSE | 95% CI coverage | mean CI width "
                  "| CI excludes 0 |",
                  "|---|---|---|---|---|---|---|"]
        for k, r in table.items():
            lines.append(f"| {k}. {LABELS[k]} | {r['mean_estimate']:.1f} | {r['bias']:+.1f} | "
                         f"{r['rmse']:.1f} | {r['coverage']:.1%} | {r['mean_ci_width']:.1f} | "
                         f"{r['detects_effect']:.0%} |")
        lines.append("")
    plot_coverage(results, out / "simulation_coverage.png")
    lines += ["![Coverage of the 95% intervals](simulation_coverage.png)", ""]
    (out / "simulation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
