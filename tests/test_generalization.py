"""Registries for other cities, context profiles, movements and PPI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_vision.analysis import movements
from traffic_vision.cli import main
from traffic_vision.evaluation import ppi_diff_in_diff, ppi_from_files, ppi_mean
from traffic_vision.rules import (check_measure, frequency_check, load_registry,
                                  movement_check, registry_info)
from traffic_vision.taxonomy import (ROAD_CLASSES, context_aliases, load_context,
                                     model_class_ids, to_road_class)

ROOT = Path(__file__).resolve().parent.parent
REGISTRIES = sorted(p for p in (ROOT / "measures").glob("*.yaml") if p.stem != "TEMPLATE")
CONTEXTS = sorted((ROOT / "contexts").glob("*.yaml"))


# --- registries and contexts ------------------------------------------------

@pytest.mark.parametrize("path", REGISTRIES, ids=lambda p: p.stem)
def test_every_registry_loads_and_is_sourced(path):
    reg = load_registry(path)
    assert reg, f"{path.name} has no measure"
    info = registry_info(path)
    assert info.get("city") and info.get("context")
    load_context(info["context"])                     # the profile it names exists
    for m in reg.values():
        assert m.sources, f"{m.id} has no source"
        assert all(s.startswith("http") for s in m.sources)
        for c in m.banned_classes + m.allowed_classes:
            assert c in ROAD_CLASSES


@pytest.mark.parametrize("path", CONTEXTS, ids=lambda p: p.stem)
def test_every_context_maps_to_core_classes(path):
    ctx = load_context(path)
    aliases = context_aliases(ctx)
    assert set(aliases.values()) <= set(ROAD_CLASSES)
    for vt in ctx.get("local_vehicle_types", []):
        assert vt["core_class"] in ROAD_CLASSES


def test_registry_kinds_cover_four_policy_types():
    types = {registry_info(p).get("policy_type") for p in REGISTRIES}
    assert {"event", "infrastructure", "regulation", "recurring"} <= types


def test_open_vocabulary_and_drone_labels():
    assert to_road_class("auto rickshaw") == "tricycle"
    assert to_road_class("tricycle") == "tricycle"          # VisDrone
    assert to_road_class("awning-tricycle") == "tricycle"
    assert to_road_class("minibus") == "bus"
    assert to_road_class("horse cart") == "cart"
    assert to_road_class("moped") == "motorcycle"
    assert to_road_class("tree") is None


def test_context_label_wins_over_default():
    names = {0: "person", 1: "trailer truck", 2: "keke"}
    ids = model_class_ids(names, extra={"keke": "tricycle"})
    assert ids == {0: "pedestrian", 1: "truck", 2: "tricycle"}


# --- movements and rate checks ---------------------------------------------

def _crossings(rows):
    return pd.DataFrame(rows, columns=["track_id", "road_class", "line", "video_time_s"])


def test_movements_from_two_lines():
    cx = _crossings([(1, "car", "north", 1.0), (1, "car", "east", 4.0),
                     (2, "truck", "north", 2.0), (2, "truck", "north", 2.5),
                     (3, "car", "east", 3.0)])
    mv = movements(cx).set_index("track_id")["movement"].to_dict()
    assert mv == {1: "north>east", 2: "north", 3: "east"}


def test_movement_and_frequency_checks():
    cx = _crossings([(i, "car", "north", float(i)) for i in range(6)]
                    + [(i, "car", "east", float(i) + 2) for i in range(4)]
                    + [(10 + i, "bus", "line", 60.0 * i) for i in range(6)])
    mv = movement_check(cx, "north", "east", 3600.0)
    assert mv.n == 4 and mv.rate == pytest.approx(4.0)
    fr = frequency_check(cx, ["bus"], 20.0, 3600.0)
    assert fr.n == 6
    assert "fewer than promised" in fr.as_text()
    assert "compatible" in frequency_check(cx, ["bus"], 6.0, 3600.0).as_text()


def _run_dir(tmp_path, name, times, cls="car", zones=None, zone_rows=None, duration=600.0):
    rd = tmp_path / name
    rd.mkdir()
    pd.DataFrame({"track_id": range(len(times)), "road_class": cls, "line": "line",
                  "video_time_s": times, "clock_time": np.nan}).to_csv(rd / "crossings.csv",
                                                                       index=False)
    if zone_rows is not None:
        pd.DataFrame(zone_rows, columns=["zone", "track_id", "road_class", "video_time_s"]) \
            .to_csv(rd / "zone_events.csv", index=False)
    (rd / "summary.json").write_text(json.dumps(
        {"video": {"duration_s": duration}, "config": {"zones": zones or {}}}))
    return rd


def test_check_measure_dispatches_by_kind(tmp_path):
    kigali = load_registry(ROOT / "measures" / "kigali_car_free_day.yaml")
    m = next(iter(kigali.values()))
    rows = [("closed_road", i, "pedestrian", 10.0 * i) for i in range(20)] + \
           [("closed_road", 100 + i, "car", 30.0 * i) for i in range(2)]
    rd = _run_dir(tmp_path, "kigali", [1.0, 2.0], zones={"closed_road": [[0, 0]] * 3},
                  zone_rows=rows)
    with pytest.raises(ValueError, match="--start-clock"):
        check_measure(m, rd)
    text = check_measure(m, rd, start_clock="2026-11-15T08:00:00")
    assert "During the ban (07:00-11:00): 2 motorcycle/tricycle/car/bus/truck" in text
    with pytest.raises(ValueError, match="not in this run"):
        check_measure(m, rd, zone="other_road", start_clock="2026-11-15T08:00:00")


def test_check_measure_all_day_ban_points_to_did(tmp_path):
    lagos = load_registry(ROOT / "measures" / "lagos_okada_ban.yaml")
    m = next(iter(lagos.values()))
    rd = _run_dir(tmp_path, "ikeja", [5.0, 50.0, 500.0], cls="motorcycle", duration=1800.0)
    text = check_measure(m, rd)
    assert "3 in 30 min" in text and "traffic-vision did" in text


# --- prediction-powered inference --------------------------------------------

def test_ppi_weights():
    rng = np.random.default_rng(1)
    truth = rng.poisson(50, 48).astype(float)
    pred = np.round(truth * 0.8)
    lab = rng.choice(48, 6, replace=False)
    r = ppi_mean(pred, pred[lab], truth[lab], lam="ratio")
    assert r.lam == pytest.approx(truth[lab].sum() / pred[lab].sum())
    assert abs(r.mean - truth.mean()) < 2.5
    assert r.naive_mean == pytest.approx(pred.mean())
    one = ppi_mean(pred, pred[lab], truth[lab], lam=1)
    assert one.lam == 1.0
    tuned = ppi_mean(pred, pred[lab], truth[lab], lam="tuned")
    assert 0.0 <= tuned.lam <= 3.0
    with pytest.raises(ValueError):
        ppi_mean(pred, pred[:2], truth[:2])


def test_ppi_did_recovers_effect_when_cameras_differ():
    rng = np.random.default_rng(3)
    means, recall = {"tb": 60, "td": 78, "cb": 50, "cd": 56}, \
        {"tb": 0.82, "td": 0.65, "cb": 0.75, "cd": 0.75}
    est = []
    for k in ("tb", "td", "cb", "cd"):
        y = rng.poisson(means[k], 72).astype(float)
        f = rng.binomial(y.astype(int), recall[k]).astype(float)
        lab = rng.choice(72, 8, replace=False)
        est.append(ppi_mean(f, f[lab], y[lab]))
    res = ppi_diff_in_diff(est)
    assert res.method == "ppi"
    assert res.ci[0] < 12 < res.ci[1]
    naive = (est[1].naive_mean - est[0].naive_mean) - (est[3].naive_mean - est[2].naive_mean)
    assert abs(naive - 12) > abs(res.effect - 12)


def test_ppi_from_files_and_cli(tmp_path, capsys):
    rng = np.random.default_rng(5)
    rows, dirs = [], []
    for name, rate in (("tb", 0.10), ("td", 0.13), ("cb", 0.08), ("cd", 0.09)):
        times = np.sort(rng.uniform(0, 1200, rng.poisson(rate * 1200)))
        dirs.append(_run_dir(tmp_path, name, times, duration=1200.0))
        for w in (0, 3, 6, 9):
            n = int(((times >= 60 * w) & (times < 60 * w + 60)).sum())
            rows.append((name, 60.0 * w, 60.0 * w + 60, n + 1))
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows, columns=["run", "start_s", "end_s", "manual_count"]).to_csv(labels,
                                                                                   index=False)
    est = ppi_from_files(labels, dirs)
    assert [e.n_windows for e in est] == [20] * 4
    assert all(e.n_labeled == 4 for e in est)
    main(["did", "--treated-before", str(dirs[0]), "--treated-during", str(dirs[1]),
          "--control-before", str(dirs[2]), "--control-during", str(dirs[3]),
          "--labels", str(labels)])
    out = capsys.readouterr().out
    assert "prediction-powered" in out and "treated before" in out


def test_cli_lists_registries(capsys):
    main(["measures", "--list"])
    out = capsys.readouterr().out
    for p in REGISTRIES:
        assert p.stem in out
    main(["measures", "--registry", "lagos_okada_ban"])
    assert "Lagos" in capsys.readouterr().out


# --- simulation: a look-alike class and study design ---------------------------

def test_lookalike_bias_is_removed_by_hand_counts():
    from traffic_vision.simulate import BanScenario, ban_run

    r = ban_run(BanScenario(confusion=0.5), n_studies=150, seed=3)
    assert r["network"]["mean_per_hour"] > 40 and r["network"]["coverage"] < 0.1
    assert abs(r["ppi_ratio"]["mean_per_hour"] - 12) < 2
    assert r["ppi_ratio"]["coverage"] > 0.8


def test_cli_design(capsys):
    main(["design", "--base", "60", "--effect", "0.2", "--hours", "2", "--days", "1",
          "--studies", "40"])
    out = capsys.readouterr().out
    assert "power" in out and "+20%" in out
