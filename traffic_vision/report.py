"""Charts and a short Markdown report for one run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import exposure_per_hour, flows, load_crossings, modal_shares, pce_flow  # noqa: E402
from .taxonomy import ROAD_CLASSES, load_context  # noqa: E402


def context_pce(summary: dict) -> dict[str, float]:
    """PCE weights of the run's context profile, if it sets any."""
    try:
        return dict(load_context(summary.get("config", {}).get("context")).get("pce") or {})
    except FileNotFoundError:
        return {}

CLASS_COLOURS = {
    "pedestrian": "#3f9e4d", "bicycle": "#b8a531", "motorcycle": "#e8862a",
    "car": "#2b6cb0", "bus": "#c53030", "truck": "#805ad5",
}


def make_report(run_dir: str | Path, interval_s: float = 10.0) -> Path:
    """Write flow charts and report.md in the run folder; return the report path."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    duration = summary["video"]["duration_s"]
    cx = load_crossings([run_dir])

    f = flows(cx, interval_s, duration, by=("road_class",))
    table = f.pivot_table(index="start_s", columns="road_class", values="count", fill_value=0)
    cols = [c for c in ROAD_CLASSES if c in table.columns]

    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = None
    for c in cols:
        ax.bar(table.index, table[c], width=interval_s * 0.9, align="edge", bottom=bottom,
               label=c, color=CLASS_COLOURS[c])
        bottom = table[c] if bottom is None else bottom + table[c]
    ax.set_xlabel("video time (s)")
    ax.set_ylabel(f"objects crossing the line per {interval_s:g} s")
    ax.set_title(f"Flow by class, {summary['config']['scene_id']}")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run_dir / "flow_by_class.png", dpi=150)
    plt.close(fig)

    totals = cx.groupby(["road_class", "direction"]).size().unstack(fill_value=0)
    totals = totals.reindex([c for c in ROAD_CLASSES if c in totals.index])
    shares = modal_shares(cx)
    pce = pce_flow(cx, 60.0, duration, pce=context_pce(summary))
    per_min = len(cx[cx.road_class != "pedestrian"]) / max(duration / 60, 1e-9)

    lines = [
        f"# Run report: {summary['config']['scene_id']}",
        "",
        f"- Video: `{Path(summary['config']['video']).name}`, {duration:.0f} s, "
        f"{summary['video']['width']}x{summary['video']['height']} at "
        f"{summary['video']['fps']:.0f} fps",
        f"- Model: `{summary['config']['model']}`, confidence {summary['config']['confidence']}",
        f"- Frames without any selected object: {summary['frames_without_selected_objects']}",
        f"- Vehicles crossing the line: {per_min:.1f} per minute; "
        f"{pce['pce'].sum():.1f} passenger car equivalents in total",
        "",
        "## Crossings by class and direction",
        "",
        totals.to_markdown(),
        "",
        "## Share of vehicles by class",
        "",
        (shares.map("{:.0%}".format).to_frame("share").to_markdown() if len(shares) else "No vehicle counted."),
        "",
        "## Exposure of vulnerable road users (per hour, extrapolated)",
        "",
        exposure_per_hour(cx, duration).astype(int).to_frame("per hour").to_markdown(),
        "",
        "![Flow by class](flow_by_class.png)",
        "",
    ]
    zf = run_dir / "zone_events.csv"
    if zf.exists():
        zev = pd.read_csv(zf)
        lines += ["## Use of zones (reserved lanes)", ""]
        if len(zev):
            zt = zev.groupby(["zone", "road_class"]).size().unstack(fill_value=0)
            lines += [zt.to_markdown(), "",
                      "Each object is counted once per zone, after two processed frames "
                      "inside it. Check a lane rule with `traffic-vision check`.", ""]
        else:
            lines += ["No object used the zones.", ""]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


def _flows_for(run_dir: Path, interval_s: float) -> pd.DataFrame:
    summary = json.loads((run_dir / "summary.json").read_text())
    return pce_flow(load_crossings([run_dir]), interval_s, summary["video"]["duration_s"],
                    pce=context_pce(summary))
