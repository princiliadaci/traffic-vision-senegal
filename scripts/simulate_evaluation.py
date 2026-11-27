"""Simulation study: can video counts measure the effect of a traffic measure?

Three experiments. A and B simulate studies of a closure (a diversion route
and a comparable control street, filmed before and during the measure, counted
per 5 minutes, true effect +12 vehicles per interval, city-wide change +6):

A. Honest intervals. Traffic is autocorrelated and the detector finds 82% of
   vehicles. Compares a before/after comparison, a raw difference-in-differences
   (DiD) and a DiD on counts corrected by 8 hand-counted intervals, with three
   ways to build the interval (ordinary bootstrap, block bootstrap, batch
   means).
B. Cameras that do not see alike. The detector finds 82% of vehicles on the
   treated camera before the measure, 70% during it (a crowd, a security
   perimeter), 75% on the control camera. Compares raw counts, a correction
   estimated on one camera, and prediction-powered inference with 6
   hand-counted intervals in each recording.

C. A look-alike vehicle. A truck ban is checked on a road where one truck
   still passes per 5 minutes among 8 car rapides, and the detector labels a
   share of car rapides as trucks (both detectors did on the photos of
   docs/results/open_vocabulary_probe.md). Compares the network's count of
   trucks with prediction-powered estimates from a few hand-counted intervals.

    python scripts/simulate_evaluation.py         # writes docs/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_vision.simulate import (SERIES, BanScenario, Scenario, ban_run,  # noqa: E402
                                     power_table, run)

LABELS = {
    "before_after": "Before/after, treated street only",
    "did_raw": "DiD, raw counts",
    "did_ratio_iid": "DiD, counts corrected on one camera, ordinary bootstrap",
    "did_ratio_block": "DiD, counts corrected on one camera, block bootstrap",
    "did_ratio_batch": "DiD, counts corrected on one camera, batch means",
    "did_ppi": "DiD, prediction-powered, weight 1",
    "did_ppi_ratio": "DiD, prediction-powered, ratio weight (default)",
    "did_ppi_tuned": "DiD, prediction-powered, tuned weight (PPI++)",
}
TEXT, MUTED, GRID = "#191919", "#666666", "#e5e5e5"
BLUE, ORANGE = "#257de0", "#f47230"

EXP_A = ("before_after", "did_raw", "did_ratio_iid", "did_ratio_block", "did_ratio_batch")
EXP_B = ("did_raw", "did_ratio_batch", "did_ppi", "did_ppi_ratio", "did_ppi_tuned")


def table_md(title: str, table: dict, keys) -> list[str]:
    lines = [f"### {title}", "",
             "| estimator | mean estimate (truth 12) | bias | RMSE | 95% CI coverage | "
             "mean CI width | CI excludes 0 |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        r = table[k]
        lines.append(f"| {LABELS[k]} | {r['mean_estimate']:.1f} | {r['bias']:+.1f} | "
                     f"{r['rmse']:.1f} | {r['coverage']:.1%} | {r['mean_ci_width']:.1f} | "
                     f"{r['detects_effect']:.0%} |")
    return lines + [""]


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_a(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [k for k in results if "rho = 0.6" in k]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.6), sharey=True)
    y = list(range(len(EXP_A)))[::-1]
    for ax, name in zip(axes, panels):
        cov = [results[name][k]["coverage"] * 100 for k in EXP_A]
        ax.axvline(95, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.hlines(y, 50, cov, color=GRID, lw=2, zorder=1)
        ax.scatter(cov, y, s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        for xi, yi in zip(cov, y):
            left = xi > 88
            ax.text(xi - 1.4 if left else xi + 1.4, yi + (0.28 if left else 0), f"{xi:.0f}%",
                    va="center", ha="right" if left else "left", fontsize=9, color=TEXT)
        ax.set_xlim(50, 102)
        ax.set_title(name.split(",")[0], fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("share of 95% intervals that contain the true effect", color=MUTED,
                      fontsize=9)
        ax.text(95.6, len(EXP_A) - 0.45, "target 95%", color=MUTED, fontsize=8)
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[k] for k in EXP_A], color=TEXT, fontsize=9)
    fig.suptitle("A. With autocorrelated traffic, only batch means keep honest intervals "
                 "(1,000 simulated studies each)", fontsize=10.5, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_b(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    colours = {names[0]: BLUE, names[1]: ORANGE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    y = list(range(len(EXP_B)))[::-1]
    off = {names[0]: 0.14, names[1]: -0.14}
    for name in names:
        est = [results[name][k]["mean_estimate"] for k in EXP_B]
        cov = [results[name][k]["coverage"] * 100 for k in EXP_B]
        yy = [v + off[name] for v in y]
        a1.scatter(est, yy, s=60, color=colours[name], edgecolor="white", linewidth=2, zorder=3,
                   label=name)
        a2.scatter(cov, yy, s=60, color=colours[name], edgecolor="white", linewidth=2, zorder=3)
        for xi, yi in zip(cov, yy):
            a2.text(xi - 1.5, yi, f"{xi:.0f}%", va="center", ha="right", fontsize=8, color=TEXT)
    a1.axvline(12, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    a1.text(12.3, len(EXP_B) - 0.5, "true effect 12", color=MUTED, fontsize=8)
    a1.set_xlim(-1, 16)
    a1.set_xlabel("mean estimated effect (vehicles per 5 min)", color=MUTED, fontsize=9)
    a2.axvline(95, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    a2.text(95.6, len(EXP_B) - 0.5, "target 95%", color=MUTED, fontsize=8)
    a2.set_xlim(30, 102)
    a2.set_xlabel("share of 95% intervals that contain the true effect", color=MUTED, fontsize=9)
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        _style(ax)
    a1.set_yticks(y)
    a1.set_yticklabels([LABELS[k] for k in EXP_B], color=TEXT, fontsize=9)
    fig.suptitle("B. When the camera sees less during the measure, the effect vanishes unless "
                 "each recording is checked by hand", fontsize=10.5, color=TEXT, x=0.01,
                 ha="left", y=0.99)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=2, loc="upper left",
               bbox_to_anchor=(0.01, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_c(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ticks = []
    for i, name in enumerate(names):
        for j, (k, colour, lab) in enumerate((("network", BLUE, "Network count alone"),
                                              ("ppi_ratio", ORANGE, "Prediction-powered, "
                                               "ratio weight"))):
            r = results[name][k]
            m, w = r["mean_per_hour"], r["mean_ci_width_per_hour"] / 2
            y = i + (0.15 if j == 0 else -0.15)
            ax.errorbar(m, y, xerr=w, fmt="o", color=colour, ms=8, lw=2, capsize=0,
                        label=lab if i == 0 else None, mec="white", mew=1.5)
            ax.text(m + w + 1.5, y, f"{m:.0f} per hour, coverage {r['coverage']:.0%}",
                    va="center", fontsize=8, color=TEXT)
        conf = name.split(" of ")[0]
        hand = name.split(", ")[1].split(" intervals")[0]
        ticks.append(f"car rapides seen as trucks: {conf}\n{hand} intervals counted by hand")
    ax.axvline(12, color=MUTED, lw=1, ls="--")
    ax.text(13, -0.55, "true rate: 12 trucks per hour", fontsize=8, color=MUTED,
            ha="left", va="center")
    ax.set_yticks(range(len(names)), ticks, fontsize=8.5)
    ax.set_ylim(len(names) - 0.5, -0.85)
    ax.set_xlabel("Trucks per hour during the ban (mean estimate and mean 95% interval)",
                  fontsize=9, color=TEXT)
    ax.set_xlim(0, 105)
    _style(ax)
    fig.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
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

    scen_a = {
        "2-hour recordings, autocorrelated traffic (rho = 0.6)": Scenario(hours=2, rho=0.6),
        "6-hour recordings, autocorrelated traffic (rho = 0.6)": Scenario(hours=6, rho=0.6),
        "2-hour recordings, independent intervals (rho = 0)": Scenario(hours=2, rho=0.0),
        "6-hour recordings, independent intervals (rho = 0)": Scenario(hours=6, rho=0.0),
    }
    res_a = {n: run(sc, a.studies, a.seed, a.boot, EXP_A) for n, sc in scen_a.items()}
    scen_b = {
        "same detection everywhere (82%)": Scenario(hours=6, n_ppi=6),
        "detection 82% before, 70% during, 75% on the control camera": Scenario(
            hours=6, n_ppi=6, recall={"tb": 0.82, "td": 0.70, "cb": 0.75, "cd": 0.75}),
    }
    res_b = {n: run(sc, a.studies, a.seed + 1, a.boot, EXP_B) for n, sc in scen_b.items()}
    scen_c = {f"{c:.0%} of car rapides labelled truck, {n} intervals counted by hand":
              (c, n, BanScenario(confusion=c, n_labels=n))
              for c, n in ((0.0, 6), (0.25, 6), (0.5, 6), (0.5, 12))}
    res_c = {name: ban_run(sc, a.studies, a.seed + 2) for name, (_, _, sc) in scen_c.items()}
    design = power_table(base=60, effect_share=0.2)

    (out / "simulation.json").write_text(json.dumps(
        {"experiment_a": res_a, "experiment_b": res_b, "experiment_c": res_c,
         "design_example": design,
         "settings": {"studies": a.studies, "bootstrap_resamples": a.boot, "seed": a.seed,
                      "interval_minutes": 5, "true_effect": 12, "city_shock": 6,
                      "base_treated": 60, "base_control": 50, "false_pos_per_interval": 1,
                      "hand_counted_intervals": {"one_camera": 8, "ppi_per_recording": 6}}},
        indent=2))

    lines = ["# Simulation study: results", "",
             f"{a.studies} simulated studies per scenario. True effect +12 vehicles per 5 "
             "min, city-wide change +6, one false detection per interval. Produced by "
             "`scripts/simulate_evaluation.py`.", "",
             "## A. Honest intervals with autocorrelated traffic", "",
             "Detection 82% everywhere; 8 intervals counted by hand on one recording.", ""]
    for name, t in res_a.items():
        lines += table_md(name[0].upper() + name[1:], t, EXP_A)
    lines += ["![Experiment A](simulation_coverage.png)", "",
              "## B. Cameras that do not see alike", "",
              "6-hour recordings, autocorrelated traffic. Correction on one camera: 8 "
              "intervals counted by hand on the treated camera before the measure. "
              "Prediction-powered: 6 intervals counted by hand in each of the four "
              "recordings (30 minutes out of 6 hours).", ""]
    for name, t in res_b.items():
        lines += table_md(name[0].upper() + name[1:], t, EXP_B)
    lines += ["![Experiment B](simulation_cameras.png)", "",
              "## C. A vehicle that looks like the banned class", "",
              "A truck ban checked over 4 hours on a road where 1 truck still passes per 5 "
              "minutes (12 per hour) among 8 car rapides. The detector finds 85% of trucks "
              "and labels a share of car rapides as trucks. The question is the true rate of "
              "trucks during the ban.", "",
              "| scenario | estimator | mean estimate per hour (truth 12) | 95% CI coverage | "
              "mean CI width per hour |", "|---|---|---|---|---|"]
    names_c = {"network": "Network count alone (Poisson interval)",
               "ppi": "Prediction-powered, weight 1",
               "ppi_ratio": "Prediction-powered, ratio weight (default)"}
    for name, t in res_c.items():
        for k, label in names_c.items():
            r = t[k]
            lines.append(f"| {name} | {label} | {r['mean_per_hour']:.1f} | "
                         f"{r['coverage']:.1%} | {r['mean_ci_width_per_hour']:.1f} |")
    lines += ["", "![Experiment C](simulation_lookalike.png)", "",
              "## Planning example", "",
              "Chance that the 95% interval excludes 0 for a +20% effect on a street with 60 "
              "vehicles per 5 minutes, by hours filmed per day and days per period "
              "(`traffic-vision design --base 60 --effect 0.2`).", "",
              "| days per period | hours per day | chance to detect | interval half-width |",
              "|---|---|---|---|"]
    for r in design:
        lines.append(f"| {r['days']} | {r['hours_per_day']:g} | {r['power']:.0%} | "
                     f"{r['ci_half_width']:.1f} |")
    lines.append("")
    plot_a(res_a, out / "simulation_coverage.png")
    plot_b(res_b, out / "simulation_cameras.png")
    plot_c(res_c, out / "simulation_lookalike.png")
    (out / "simulation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
