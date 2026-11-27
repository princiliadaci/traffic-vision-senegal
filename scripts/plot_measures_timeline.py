"""Timeline of the Games mobility measures, drawn from configs/joj_measures.yaml.

    python scripts/plot_measures_timeline.py      # writes docs/img/joj_measures_timeline.png
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traffic_vision.rules import load_registry  # noqa: E402

TEXT, MUTED, GRID, BAND = "#191919", "#666666", "#e5e5e5", "#f2f2f2"
KIND = {"reserved_lane": ("Reserved lane", "#257de0"),
        "class_ban": ("Ban on a vehicle class", "#f47230"),
        "closure": ("Road closure", "#00b894")}
SHORT = {
    "olympic_lane_a1_opening": "Olympic lane, A1 motorway (opening day)",
    "trucks_toll_motorway_opening": "Trucks banned, toll motorway (opening day)",
    "trucks_rufisque_bridge_opening": "Trucks banned, Rufisque bridge (opening day)",
    "trucks_rn1_opening": "Trucks banned, RN1 15:00-19:00 (opening day)",
    "blaise_diagne_bus_carriageway": "Bus-only carriageway, av. Blaise Diagne",
    "iba_mar_diop_trucks_two_wheelers": "Trucks and motorbikes banned, Iba Mar Diop",
    "seydou_nourou_tall_trucks": "Trucks banned, allees Seydou Nourou Tall",
    "corniche_ouest_road_events": "Corniche Ouest closures (road events)",
    "diamniadio_spheres_shuttle_lane": "Shuttle lane, Diamniadio",
    "saly_night_motorbike_ban": "Motorbikes banned at night, Saly 18:00-07:00",
    "saly_two_wheelers_trucks": "Two-wheelers and trucks restricted, Saly",
}
MILESTONES = [  # date, label; sources in docs/context_joj_dakar_2026.md
    (date(2026, 10, 17), "traffic plan\nin force"),
    (date(2026, 10, 29), "TROC control\ncentre opens"),
    (date(2026, 11, 14), "temporary\nmeasures lifted"),
]


def main() -> None:
    reg = load_registry(ROOT / "configs" / "joj_measures.yaml")
    rows = [m for m in reg.values() if m.period]
    rows.sort(key=lambda m: (date.fromisoformat(m.period[0]), m.kind))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    games = (date(2026, 10, 31), date(2026, 11, 14))
    ax.axvspan(*games, color=BAND, zorder=0)
    ax.text(games[0] + timedelta(hours=6), len(rows) - 0.35, "Games: 31 Oct - 13 Nov",
            color=MUTED, fontsize=8.5, va="bottom")
    for i, m in enumerate(rows):
        y = len(rows) - 1 - i
        start = date.fromisoformat(m.period[0])
        end = date.fromisoformat(m.period[1]) + timedelta(days=1)
        colour = KIND[m.kind][1]
        ax.barh(y, (end - start).days, left=start, height=0.56, color=colour, zorder=2)
    for d, label in MILESTONES:
        ax.axvline(d, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        right = d > date(2026, 11, 10)
        ax.text(d + timedelta(hours=-8 if right else 8), -1.05, label, fontsize=7.5,
                color=MUTED, va="top", ha="right" if right else "left")
    ax.set_ylim(-1.9, len(rows))
    ax.set_xlim(date(2026, 10, 16), date(2026, 11, 17))
    ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    ax.set_yticklabels([SHORT.get(m.id, m.name) for m in rows], fontsize=8.5, color=TEXT)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[17, 24, 31, 7, 14]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in KIND.values()]
    ax.legend(handles, [n for n, _ in KIND.values()], loc="upper left", frameon=False,
              fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.08))
    fig.suptitle("Dakar 2026 Youth Olympic Games: mobility measures in the registry, "
                 "by period", x=0.01, ha="left", fontsize=10.5, color=TEXT)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.86, bottom=0.08)
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "joj_measures_timeline.png", dpi=150, facecolor="white")
    print(out / "joj_measures_timeline.png")


if __name__ == "__main__":
    main()
