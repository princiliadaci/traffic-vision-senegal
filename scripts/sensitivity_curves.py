"""Sensitivity curves: when do video counts break, and what repairs them?

The tables of scripts/simulate_evaluation.py give one point per scenario. These
curves move one condition at a time, with everything else as in experiment B
(closure, 6-hour recordings, autocorrelated traffic, true effect +12 vehicles
per 5 minutes):

1. detection during the measure falls from 82% to 52% (a crowd, a security
   perimeter, the night): estimated effect and coverage of the 95% interval,
   for raw counts, counts corrected on one camera, and prediction-powered
   counts with 6 hand-counted windows per recording;
2. a share of car rapides is seen as trucks during a truck ban (0 to 60%):
   estimated truck rate and coverage, network alone against prediction-powered;
3. number of hand-counted windows per recording (3 to 24): coverage and width
   of the prediction-powered interval;
4. chance to detect a +20% effect against days filmed per period.

    python scripts/sensitivity_curves.py        # writes docs/results/curves_*.png and curves.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_vision.simulate import (BanScenario, Scenario, ban_study, one_study,  # noqa: E402
                                     power_table)

TEXT, MUTED, GRID = "#191919", "#666666", "#e5e5e5"
# Poster palette: orange for counts left as they are, purples for corrected counts.
RAW, MID, OURS = "#d9702a", "#a585dc", "#7040b0"
RAMP = ("#cdbdf0", "#9a74d6", "#5a2f99")  # 2, 4, 8 hours a day (light to dark)
TRUTH = 12.0


def _collect(rng, sc, n, which):
    rows = {k: [] for k in which}
    for _ in range(n):
        for k, v in one_study(rng, sc, 199, which).items():
            rows[k].append(v)
    out = {}
    for k, r in rows.items():
        a = np.array(r)
        est, lo, hi = a[:, 0], a[:, 1], a[:, 2]
        out[k] = {"mean": float(est.mean()), "q10": float(np.quantile(est, 0.1)),
                  "q90": float(np.quantile(est, 0.9)),
                  "coverage": float(((lo <= sc.effect) & (sc.effect <= hi)).mean()),
                  "half_width": float(((hi - lo) / 2).mean())}
    return out


def detection_curve(n, seed):
    which = ("did_raw", "did_ratio_batch", "did_ppi_ratio")
    rng = np.random.default_rng(seed)
    pts = []
    for r in (0.82, 0.76, 0.70, 0.64, 0.58, 0.52):
        sc = Scenario(hours=6, n_ppi=6, recall={"tb": 0.82, "td": r, "cb": 0.75, "cd": 0.75})
        pts.append({"recall_during": r, **_collect(rng, sc, n, which)})
    return pts


def lookalike_curve(n, seed):
    rng = np.random.default_rng(seed)
    pts = []
    for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        sc = BanScenario(confusion=c)
        rows = {"network": [], "ppi_ratio": []}
        for _ in range(n):
            res = ban_study(rng, sc)
            for k in rows:
                rows[k].append(res[k])
        per_hour = 60 / sc.interval_min
        point = {"confusion": c}
        for k, r in rows.items():
            a = np.array(r) * per_hour
            est, lo, hi = a[:, 0], a[:, 1], a[:, 2]
            point[k] = {"mean": float(est.mean()), "q10": float(np.quantile(est, 0.1)),
                        "q90": float(np.quantile(est, 0.9)),
                        "coverage": float(((lo <= TRUTH) & (TRUTH <= hi)).mean()),
                        "half_width": float(((hi - lo) / 2).mean())}
        pts.append(point)
    return pts


def handcount_curve(n, seed):
    rng = np.random.default_rng(seed)
    pts = []
    for k in (3, 4, 6, 8, 12, 16, 24):
        sc = Scenario(hours=6, n_ppi=k, recall={"tb": 0.82, "td": 0.70, "cb": 0.75, "cd": 0.75})
        pts.append({"hand_counts": k, **_collect(rng, sc, n, ("did_ppi_ratio",))})
    return pts


FS = 12  # base font size; the figures are printed on a poster too


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=FS, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _legend(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=ncol,
               frameon=False, fontsize=FS, handlelength=1.8, columnspacing=1.6)


def _ref(ax, y, text, x, ha="left", va="bottom", dy=0.0):
    ax.axhline(y, color=MUTED, ls="--", lw=1)
    ax.text(x, y + dy, text, fontsize=FS - 1, color=MUTED, ha=ha, va=va)


def plot_detection(pts, path):
    import matplotlib.pyplot as plt

    x = [100 * p["recall_during"] for p in pts]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.42})
    series = (("did_raw", RAW, "raw counts"), ("did_ratio_batch", MID, "corrected on one camera"),
              ("did_ppi_ratio", OURS, "prediction-powered (6 hand counts per recording)"))
    handles = []
    for k, c, lab in series:
        a1.fill_between(x, [p[k]["q10"] for p in pts], [p[k]["q90"] for p in pts], color=c,
                        alpha=0.09, lw=0)
        (h,) = a1.plot(x, [p[k]["mean"] for p in pts], color=c, lw=2.6, marker="o", ms=5,
                       mec="white", mew=1)
        a2.plot(x, [100 * p[k]["coverage"] for p in pts], color=c, lw=2.6, marker="o", ms=5,
                mec="white", mew=1)
        handles.append(h)
    _legend(fig, handles, [lab for _, _, lab in series], 3)
    _ref(a1, TRUTH, "true effect +12", x[0] - 0.5, dy=2.2)
    a1.axhline(0, color=TEXT, lw=0.8)
    a1.annotate("the effect\nchanges sign", xy=(70, 0.8), xytext=(62.5, 14), fontsize=FS - 1,
                color=TEXT, ha="center", arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 1})
    _ref(a2, 95, "target 95%", x[-1] + 0.5, ha="right", dy=1.5)
    for ax in (a1, a2):
        ax.invert_xaxis()
        ax.set_xlabel("detection rate during the measure (%)", fontsize=FS, color=TEXT)
        _style(ax)
        ax.set_xlim(x[0] + 1.5, x[-1] - 1.5)
    a1.set_ylabel("estimated effect (vehicles per 5 min)", fontsize=FS, color=TEXT)
    a2.set_ylabel("intervals containing the truth (%)", fontsize=FS, color=TEXT)
    a2.set_ylim(0, 105)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.15)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_lookalike(pts, path):
    import matplotlib.pyplot as plt

    x = [100 * p["confusion"] for p in pts]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.42})
    series = (("network", RAW, "network alone"),
              ("ppi_ratio", OURS, "prediction-powered (6 hand counts)"))
    handles = []
    for k, c, lab in series:
        a1.fill_between(x, [p[k]["q10"] for p in pts], [p[k]["q90"] for p in pts], color=c,
                        alpha=0.12, lw=0)
        (h,) = a1.plot(x, [p[k]["mean"] for p in pts], color=c, lw=2.6, marker="o", ms=5,
                       mec="white", mew=1)
        a2.plot(x, [100 * p[k]["coverage"] for p in pts], color=c, lw=2.6, marker="o", ms=5,
                mec="white", mew=1)
        handles.append(h)
    _legend(fig, handles, [lab for _, _, lab in series], 2)
    a1.axhline(TRUTH, color=MUTED, ls="--", lw=1)
    a1.text(x[-1], 2.2, "true rate: 12 trucks per hour", fontsize=FS - 1, color=MUTED, ha="right")
    _ref(a2, 95, "target 95%", x[-1], ha="right", va="bottom", dy=1.5)
    for ax in (a1, a2):
        ax.set_xlabel("car rapides seen as trucks (%)", fontsize=FS, color=TEXT)
        _style(ax)
        ax.set_xlim(-2, x[-1] + 2)
    a1.set_ylabel("trucks per hour during the ban", fontsize=FS, color=TEXT)
    a1.set_ylim(0, 80)
    a2.set_ylabel("intervals containing the truth (%)", fontsize=FS, color=TEXT)
    a2.set_ylim(0, 105)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.15)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_design(hand, power, path):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.42})
    k = [p["hand_counts"] for p in hand]
    hw = [p["did_ppi_ratio"]["half_width"] for p in hand]
    a1.plot(k, hw, color=OURS, lw=2.6, marker="o", ms=6, mec="white", mew=1.2)
    cov = [100 * p["did_ppi_ratio"]["coverage"] for p in hand]
    a1.text(25, 2.0, f"coverage {min(cov):.0f} to {max(cov):.0f}% at every point", ha="right",
            fontsize=FS - 1, color=MUTED)
    a1.set_xlabel("hand-counted windows per recording", fontsize=FS, color=TEXT)
    a1.set_ylabel("half-width of the 95% interval", fontsize=FS, color=TEXT)
    a1.set_ylim(0, max(hw) * 1.3)
    a1.set_xlim(0, 26)
    _style(a1)
    handles = []
    for hours, c in zip((8, 4, 2), RAMP[::-1]):
        rows = [r for r in power if r["hours_per_day"] == hours]
        (h,) = a2.plot([r["days"] for r in rows], [100 * r["power"] for r in rows], color=c,
                       lw=2.6, marker="o", ms=5, mec="white", mew=1)
        handles.append(h)
    a2.axhline(80, color=MUTED, ls="--", lw=1)
    a2.text(3.0, 81.5, "80% power", fontsize=FS - 1, color=MUTED, ha="left", va="bottom")
    a2.set_xlabel("days filmed per period", fontsize=FS, color=TEXT)
    a2.set_ylabel("chance to detect a +20% effect (%)", fontsize=FS, color=TEXT)
    a2.set_ylim(0, 105)
    _style(a2)
    a2.legend(handles, ["8 h a day", "4 h a day", "2 h a day"], loc="lower right",
              frameon=False, fontsize=FS - 1, bbox_to_anchor=(1.0, 0.02))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.15)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "results"))
    ap.add_argument("--plot-only", action="store_true", help="redraw from curves.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        d = json.loads((out / "curves.json").read_text())
        plot_detection(d["detection"], out / "curves_detection.png")
        plot_lookalike(d["lookalike"], out / "curves_lookalike.png")
        plot_design(d["hand_counts"], d["power"], out / "curves_design.png")
        return

    det = detection_curve(a.studies, a.seed)
    look = lookalike_curve(a.studies, a.seed + 1)
    hand = handcount_curve(a.studies, a.seed + 2)
    power = power_table(base=60, effect_share=0.2, hours_options=(2, 4, 8),
                        days_options=(1, 2, 3, 5, 7), n_studies=300, seed=a.seed + 3)
    (out / "curves.json").write_text(json.dumps(
        {"settings": {"studies_per_point": a.studies, "seed": a.seed},
         "detection": det, "lookalike": look, "hand_counts": hand, "power": power}, indent=1))
    plot_detection(det, out / "curves_detection.png")
    plot_lookalike(look, out / "curves_lookalike.png")
    plot_design(hand, power, out / "curves_design.png")
    for p in det:
        print("recall", p["recall_during"], {k: round(v["mean"], 1) for k, v in p.items()
                                             if isinstance(v, dict)},
              {k: round(v["coverage"], 2) for k, v in p.items() if isinstance(v, dict)})
    for p in look:
        print("confusion", p["confusion"], round(p["network"]["mean"], 1),
              round(p["ppi_ratio"]["mean"], 1), round(p["ppi_ratio"]["coverage"], 2))
    for p in hand:
        print("hand", p["hand_counts"], round(p["did_ppi_ratio"]["coverage"], 3),
              round(p["did_ppi_ratio"]["half_width"], 1))
    for r in power:
        print("power", r["days"], r["hours_per_day"], r["power"])


if __name__ == "__main__":
    main()
