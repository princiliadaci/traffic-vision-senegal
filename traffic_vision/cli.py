"""Command line: `traffic-vision run | report | compare | aggregate | measures | check | did | calibrate`.

Examples
--------
    traffic-vision run --video data/corniche_before.mp4 --scene corniche_before \
        --line 0.05,0.62 0.95,0.62 --directions to_ouakam to_fann \
        --start-clock 2026-10-27T07:00:00
    traffic-vision run --config configs/scenes.example.yaml --scene blaise_diagne_during
    traffic-vision report outputs/corniche_before
    traffic-vision compare outputs/corniche_before outputs/corniche_during --interval 300
    traffic-vision aggregate            # merge every run in outputs/ into one dataset

    traffic-vision measures             # the Games measures and what video can measure
    traffic-vision check blaise_diagne_bus_carriageway outputs/blaise_diagne_during
    traffic-vision did --treated-before outputs/cheikh_anta_before \
        --treated-during outputs/cheikh_anta_closure --control-before outputs/bourguiba_before \
        --control-during outputs/bourguiba_closure --interval 300 --calibration manual.csv
    traffic-vision calibrate manual_counts.csv outputs/cheikh_anta_before
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .taxonomy import ROAD_CLASSES


def _point(s: str) -> tuple[float, float]:
    x, y = (float(v) for v in s.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("line points are fractions of the frame, between 0 and 1")
    return x, y


def _from_yaml(path: str, scene: str) -> dict:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    defaults = data.get("defaults", {})
    scenes = {s["scene_id"]: s for s in data.get("scenes", [])}
    if scene not in scenes:
        sys.exit(f"Scene '{scene}' not found in {path}. Available: {', '.join(scenes)}")
    cfg = {**defaults, **scenes[scene]}
    if "line" in cfg:
        cfg["line"] = tuple(tuple(p) for p in cfg["line"])
    if "line_directions" in cfg:
        cfg["line_directions"] = tuple(cfg["line_directions"])
    if "start_clock" in cfg and cfg["start_clock"] is not None:
        cfg["start_clock"] = str(cfg["start_clock"]).replace(" ", "T")
    cfg.pop("measure", None)
    cfg.pop("description", None)
    cfg.pop("period", None)
    return cfg


def cmd_run(a: argparse.Namespace) -> None:
    from .pipeline import RunConfig, run
    from .report import make_report

    if a.config:
        kw = _from_yaml(a.config, a.scene)
    else:
        if not a.video:
            sys.exit("Give --video, or --config with --scene.")
        kw = {"video": a.video, "scene_id": a.scene or Path(a.video).stem}
    for k in ("model", "confidence", "imgsz", "stride", "max_seconds", "output_dir", "device"):
        v = getattr(a, k)
        if v is not None:
            kw[k] = v
    if a.classes:
        kw["classes"] = a.classes
    if a.line:
        kw["line"] = tuple(a.line)
    if a.directions:
        kw["line_directions"] = tuple(a.directions)
    if a.no_video:
        kw["save_video"] = False
    if a.zone:
        kw["zones"] = {z[0]: [list(_point(v)) for v in z[1:]] for z in a.zone}
    if a.start_clock:
        kw["start_clock"] = a.start_clock
    out = run(RunConfig(**kw))
    rep = make_report(out)
    print(json.dumps(json.loads((out / "summary.json").read_text())["line_crossings"], indent=2))
    print(f"Results: {out}\nReport: {rep}")


def cmd_report(a: argparse.Namespace) -> None:
    from .report import make_report

    print(make_report(a.run_dir, a.interval))


def cmd_compare(a: argparse.Namespace) -> None:
    from .analysis import compare_periods, flows, load_crossings
    from .report import _flows_for

    ra, rb = Path(a.run_a), Path(a.run_b)
    if a.metric == "pce":
        va = _flows_for(ra, a.interval)["pce"]
        vb = _flows_for(rb, a.interval)["pce"]
        unit = f"PCE per {a.interval:g} s"
    else:
        def veh(run: Path):
            dur = json.loads((run / "summary.json").read_text())["video"]["duration_s"]
            cx = load_crossings([run])
            cx = cx[cx.road_class.isin(a.classes or [c for c in ROAD_CLASSES if c != "pedestrian"])]
            return flows(cx, a.interval, dur, by=()).set_index("interval")["count"]
        va, vb = veh(ra), veh(rb)
        unit = f"objects per {a.interval:g} s"
    res = compare_periods(va, vb, a.label_a or ra.name, a.label_b or rb.name)
    print(res.as_text(unit))


def cmd_aggregate(a: argparse.Namespace) -> None:
    from .analysis import aggregate_runs

    runs = a.run_dirs or sorted(str(p.parent) for p in Path(a.outputs).glob("*/summary.json"))
    if not runs:
        sys.exit("No run found.")
    cx, table = aggregate_runs(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cx.to_csv(out / "all_crossings.csv", index=False)
    table.to_csv(out / "scenes_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"Written: {out / 'all_crossings.csv'}, {out / 'scenes_summary.csv'}")


DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "configs" / "joj_measures.yaml"


def _registry(path: str | None):
    from .rules import load_registry

    p = Path(path) if path else DEFAULT_REGISTRY
    if not p.exists():
        sys.exit(f"Measure registry not found: {p}")
    return load_registry(p)


def cmd_measures(a: argparse.Namespace) -> None:
    reg = _registry(a.registry)
    for m in reg.values():
        when = f"{m.period[0]} to {m.period[1]}" if m.period else "dates not set"
        hours = f", {m.window}" if m.window else ""
        print(f"{m.id}  [{m.kind}, observability {m.observability}]\n"
              f"    {m.name}\n    {m.place}; {when}{hours}\n"
              f"    measure: {' '.join(m.indicator.split())}\n"
              f"    video cannot see: {' '.join(m.video_cannot_see.split()) or '-'}\n")


def cmd_check(a: argparse.Namespace) -> None:
    import pandas as pd

    from .rules import ban_check, lane_use

    reg = _registry(a.registry)
    if a.measure not in reg:
        sys.exit(f"Unknown measure '{a.measure}'. Run `traffic-vision measures` for the list.")
    m = reg[a.measure]
    run = Path(a.run_dir)
    summary = json.loads((run / "summary.json").read_text())
    duration = summary["video"]["duration_s"]
    print(f"{m.name}\n{m.place}\n")
    if m.kind == "reserved_lane":
        zf = run / "zone_events.csv"
        if not zf.exists():
            sys.exit(f"No zone_events.csv in {run}: run with a zone named '{m.zone}' "
                     f"(--zone {m.zone} X,Y X,Y X,Y ...).")
        zone = a.zone or m.zone
        print(lane_use(pd.read_csv(zf), zone, m.allowed_classes, duration, a.interval).as_text())
    elif m.kind == "class_ban":
        if m.window is None:
            sys.exit("This ban applies all day: compare a recording during the Games with one "
                     "after them, with `traffic-vision compare --classes "
                     f"{' '.join(m.banned_classes)}`.")
        cx = pd.read_csv(run / "crossings.csv")
        start = a.start_clock or summary["config"].get("start_clock")
        print(ban_check(cx, m.banned_classes, m.window, duration, start, m).as_text())
    else:
        print("A closure is measured on the diversion routes, against a control street, with "
              "`traffic-vision did` (four recordings: treated and control, before and during).")
    print(f"\nSources: {', '.join(m.sources)}")


def _series(run: Path, interval: float, classes: list[str] | None):
    from .analysis import flows, load_crossings

    dur = json.loads((run / "summary.json").read_text())["video"]["duration_s"]
    cx = load_crossings([run])
    cx = cx[cx.road_class.isin(classes or [c for c in ROAD_CLASSES if c != "pedestrian"])]
    return flows(cx, interval, dur, by=()).set_index("interval")["count"].to_numpy()


def cmd_did(a: argparse.Namespace) -> None:
    from .evaluation import calibrate, calibration_from_files, diff_in_diff

    runs = [Path(r) for r in (a.treated_before, a.treated_during, a.control_before,
                              a.control_during)]
    series = [_series(r, a.interval, a.classes) for r in runs]
    cal = None
    if a.calibration:
        cals = calibration_from_files(a.calibration, a.calibration_run or runs[0])
        classes = a.classes or list(cals)
        manual = sum(cals[c].manual for c in classes if c in cals)
        auto = sum(cals[c].auto for c in classes if c in cals)
        cal = calibrate(manual, auto, "+".join(classes))
        print("Calibration:", cal.as_text())
    res = diff_in_diff(*series, block=a.block, calibration=cal)
    print(res.as_text(f"objects per {a.interval:g} s"))


def cmd_calibrate(a: argparse.Namespace) -> None:
    from .evaluation import calibration_from_files

    for c in calibration_from_files(a.manual_csv, a.run_dir).values():
        print(c.as_text())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="traffic-vision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="detect, track and count in a video")
    r.add_argument("--video")
    r.add_argument("--scene", help="scene id (output folder name)")
    r.add_argument("--config", help="YAML file describing scenes")
    r.add_argument("--model", help="YOLO weights, default yolo11n.pt")
    r.add_argument("--classes", nargs="+", choices=ROAD_CLASSES)
    r.add_argument("--confidence", type=float)
    r.add_argument("--imgsz", type=int)
    r.add_argument("--stride", type=int, help="process 1 frame out of N")
    r.add_argument("--max-seconds", dest="max_seconds", type=float)
    r.add_argument("--line", nargs=2, type=_point, metavar="X,Y",
                   help="counting line, two points as fractions of the frame")
    r.add_argument("--directions", nargs=2, metavar=("NAME_1", "NAME_2"))
    r.add_argument("--output-dir", dest="output_dir")
    r.add_argument("--device")
    r.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    r.add_argument("--zone", nargs="+", action="append", metavar="NAME_OR_X,Y",
                   help="zone (reserved lane): a name then 3 or more points X,Y as fractions "
                        "of the frame; repeat for several zones")
    r.add_argument("--start-clock", dest="start_clock",
                   help="clock time of the first frame, e.g. 2026-11-07T07:00:00")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="charts and report.md for a run")
    rep.add_argument("run_dir")
    rep.add_argument("--interval", type=float, default=10.0)
    rep.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="compare flows between two runs (e.g. before / during)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--interval", type=float, default=60.0)
    c.add_argument("--metric", choices=["vehicles", "pce"], default="vehicles")
    c.add_argument("--classes", nargs="+", choices=ROAD_CLASSES)
    c.add_argument("--label-a", dest="label_a")
    c.add_argument("--label-b", dest="label_b")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("aggregate", help="merge several runs (cameras, sites) into one dataset")
    g.add_argument("run_dirs", nargs="*", help="run folders; all runs in --outputs when omitted")
    g.add_argument("--outputs", default="outputs")
    g.add_argument("--out", default="outputs/_aggregate")
    g.set_defaults(func=cmd_aggregate)

    ms = sub.add_parser("measures", help="list the Games measures and what video can measure")
    ms.add_argument("--registry", help="measure registry (YAML)")
    ms.set_defaults(func=cmd_measures)

    ck = sub.add_parser("check", help="check one measure against a run")
    ck.add_argument("measure")
    ck.add_argument("run_dir")
    ck.add_argument("--registry")
    ck.add_argument("--zone", help="zone name in the run, if not the registry's")
    ck.add_argument("--interval", type=float, default=60.0)
    ck.add_argument("--start-clock", dest="start_clock")
    ck.set_defaults(func=cmd_check)

    d = sub.add_parser("did", help="difference-in-differences: treated and control, before and during")
    for name in ("treated-before", "treated-during", "control-before", "control-during"):
        d.add_argument(f"--{name}", dest=name.replace("-", "_"), required=True)
    d.add_argument("--interval", type=float, default=300.0)
    d.add_argument("--classes", nargs="+", choices=ROAD_CLASSES)
    d.add_argument("--block", type=int, help="block length in intervals (default n^(1/3))")
    d.add_argument("--calibration", help="manual count CSV (start_s,end_s,road_class,manual_count)")
    d.add_argument("--calibration-run", dest="calibration_run",
                   help="run the manual count refers to (default: treated before)")
    d.set_defaults(func=cmd_did)

    cb = sub.add_parser("calibrate", help="compare a manual count with the automatic count")
    cb.add_argument("manual_csv")
    cb.add_argument("run_dir")
    cb.set_defaults(func=cmd_calibrate)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
