from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from traffic_vision.rules import TimeWindow, ban_check, lane_use, load_registry, poisson_ci
from traffic_vision.zones import Zone, ZoneCounter, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon_square_and_concave():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 8), l_shape)
    assert not point_in_polygon((8, 8), l_shape)


def test_zone_from_relative_scales_and_validates():
    z = Zone.from_relative("lane", [(0, 0), (0.5, 0), (0.5, 1)], 200, 100)
    assert z.polygon[2] == (100, 100)
    with pytest.raises(ValueError):
        Zone.from_relative("bad", [(0, 0), (1.2, 0), (1, 1)], 10, 10)


def test_zone_counter_needs_min_frames_and_counts_once():
    zc = ZoneCounter([Zone("lane", SQUARE)], min_frames=2)
    assert zc.update(1, "bus", (5, 5), 0, 0.0) == []        # first frame inside
    ev = zc.update(1, "bus", (6, 5), 1, 0.1)                 # second frame: counted
    assert len(ev) == 1 and ev[0].road_class == "bus"
    assert zc.update(1, "bus", (7, 5), 2, 0.2) == []         # never twice
    zc.update(2, "car", (20, 5), 0, 0.0)                     # outside: not counted
    assert zc.totals() == {"lane": {"bus": 1}}


def test_time_window_with_and_without_midnight():
    day = TimeWindow.parse("07:00-19:00")
    night = TimeWindow.parse("18:00-07:00")
    assert day.contains(time(7, 0)) and not day.contains(time(19, 0))
    assert night.contains(time(23, 30)) and night.contains(time(6, 59))
    assert not night.contains(time(12, 0))
    with pytest.raises(ValueError):
        TimeWindow.parse("7h-19h")


def test_poisson_interval_close_to_exact_values():
    # exact Garwood bounds: n=0 -> [0, 3.689]; n=10 -> [4.795, 18.390]
    lo, hi = poisson_ci(0)
    assert lo == 0 and hi == pytest.approx(3.689, rel=0.02)
    lo, hi = poisson_ci(10)
    assert lo == pytest.approx(4.795, rel=0.02) and hi == pytest.approx(18.390, rel=0.02)


def test_lane_use_share_and_others():
    ev = pd.DataFrame({
        "zone": ["lane"] * 5 + ["other"],
        "road_class": ["bus", "bus", "bus", "car", "pedestrian", "car"],
        "video_time_s": [10, 70, 130, 140, 150, 20],
    })
    res = lane_use(ev, "lane", ["bus"], duration_s=180, interval_s=60)
    assert res.n_vehicles == 4 and res.n_allowed == 3      # pedestrian and other zone ignored
    assert res.share_allowed == pytest.approx(0.75)
    assert res.others_per_hour == pytest.approx(20.0)      # 1 car in 3 minutes
    assert 0 <= res.share_ci[0] <= 0.75 <= res.share_ci[1] <= 1
    assert "not offences" in res.as_text()


def test_ban_check_splits_time_inside_and_outside_the_window():
    # recording from 06:30 to 07:30, ban from 07:00: half an hour on each side
    cx = pd.DataFrame({"road_class": ["truck", "truck", "truck", "car"],
                       "video_time_s": [600, 1200, 2400, 100]})
    res = ban_check(cx, ["truck"], TimeWindow.parse("07:00-19:00"), duration_s=3600,
                    start_clock="2026-10-31T06:30:00")
    assert res.hours_in == pytest.approx(0.5) and res.hours_out == pytest.approx(0.5)
    assert (res.n_in, res.n_out) == (1, 2)
    assert res.rate_in == pytest.approx(2.0) and res.rate_out == pytest.approx(4.0)


def test_ban_check_reads_clock_time_column():
    cx = pd.DataFrame({"road_class": ["motorcycle"], "video_time_s": [60.0],
                       "clock_time": ["2026-11-02T18:01:00"]})
    res = ban_check(cx, ["motorcycle"], TimeWindow.parse("18:00-07:00"), duration_s=600)
    assert res.n_in == 1 and res.hours_out == 0


def test_registry_of_games_measures_is_valid():
    reg = load_registry(Path(__file__).resolve().parent.parent / "configs" / "joj_measures.yaml")
    assert len(reg) >= 10
    kinds = {m.kind for m in reg.values()}
    assert kinds == {"reserved_lane", "class_ban", "closure"}
    for m in reg.values():
        assert all(s.startswith("https://") for s in m.sources)
        assert m.observability in {"high", "partial", "low"}
    assert str(reg["saly_night_motorbike_ban"].window) == "18:00-07:00"
