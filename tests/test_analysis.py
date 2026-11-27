import pandas as pd

from traffic_vision.analysis import compare_periods, flows, modal_shares, pce_flow


def crossings():
    return pd.DataFrame({
        "video_time_s": [5, 20, 70, 75, 80, 200],
        "road_class": ["car", "bus", "car", "motorcycle", "pedestrian", "car"],
        "direction": ["in", "in", "out", "in", "in", "in"],
    })


def test_flows_keep_empty_intervals():
    f = flows(crossings(), 60, duration_s=240, by=())
    assert f["count"].tolist() == [2, 3, 0, 1, 0]


def test_pce_weights_buses_more_than_cars():
    f = pce_flow(crossings(), 60, duration_s=240, pce={"bus": 2.0, "car": 1.0, "motorcycle": 0.5})
    assert f["pce"].tolist() == [3.0, 1.5, 0.0, 1.0, 0.0]


def test_modal_shares_exclude_pedestrians():
    s = modal_shares(crossings())
    assert abs(s.sum() - 1) < 1e-9 and "pedestrian" not in s.index
    assert s["car"] == 0.6


def test_compare_detects_a_clear_change():
    res = compare_periods([10, 11, 9, 10, 12, 10], [20, 21, 19, 22, 20, 21])
    assert res.difference > 9 and res.ci_low > 0


def test_compare_same_distribution_includes_zero():
    a = [10, 12, 9, 11, 10, 13, 8, 11]
    res = compare_periods(a, list(reversed(a)))
    assert res.ci_low < 0 < res.ci_high


def test_exposure_per_hour():
    from traffic_vision.analysis import exposure_per_hour

    e = exposure_per_hour(crossings(), duration_s=360)
    assert e["pedestrian"] == 10 and e["motorcycle"] == 10 and e["bicycle"] == 0
