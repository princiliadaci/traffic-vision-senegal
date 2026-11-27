"""Command line: `traffic-vision run | report | compare | aggregate | measures | check | did | calibrate | design`.

Examples
--------
    traffic-vision run --video data/corniche_before.mp4 --scene corniche_before \
        --line 0.05,0.62 0.95,0.62 --directions to_ouakam to_fann \
        --context dakar --start-clock 2026-10-27T07:00:00
    traffic-vision run --config configs/scenes.example.yaml --scene blaise_diagne_during
    traffic-vision report outputs/corniche_before
    traffic-vision compare outputs/corniche_before outputs/corniche_during --interval 300
    traffic-vision aggregate            # merge every run in outputs/ into one dataset

    traffic-vision measures --list      # every registry in measures/ (cities, policies)
    traffic-vision measures --registry measures/lagos_okada_ban.yaml
    traffic-vision check blaise_diagne_bus_carriageway outputs/blaise_diagne_during
    traffic-vision did --treated-before outputs/cheikh_anta_before \
        --treated-during outputs/cheikh_anta_closure --control-before outputs/bourguiba_before \
        --control-during outputs/bourguiba_closure --interval 300 --labels labels.csv
    traffic-vision calibrate manual_counts.csv outputs/cheikh_anta_before
    traffic-vision design --base 60 --effect 0.2     # how many hours and days to film
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .taxonomy import ROAD_CLASSES, data_dir


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
    if a.context:
        kw["context"] = a.context
    if a.line_extra:
        kw["extra_lines"] = {e[0]: [list(_point(v)) for v in e[1:3]] for e in a.line_extra}
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


MEASURES_DIR = data_dir("measures")
DEFAULT_REGISTRY = MEASURES_DIR / "dakar_joj_2026.yaml"


def _registry_path(path: str | None) -> Path:
    if not path:
        return DEFAULT_REGISTRY
    p = Path(path)
    if not p.exists() and (MEASURES_DIR / path).exists():
        p = MEASURES_DIR / path
    if not p.exists() and (MEASURES_DIR / f"{path}.yaml").exists():
        p = MEASURES_DIR / f"{path}.yaml"
    if not p.exists():
        sys.exit(f"Measure registry not found: {path}. Run `traffic-vision measures --list`.")
    return p


def _registry(path: str | None):
    from .rules import load_registry

    return load_registry(_registry_path(path))


def cmd_measures(a: argparse.Namespace) -> None:
    from .rules import load_registry, registry_info

    if a.list:
        for f in sorted(MEASURES_DIR.glob("*.yaml")):
            if f.stem.upper() == "TEMPLATE":
                continue
            info = registry_info(f)
            n = len(load_registry(f))
            print(f"{f.stem}  [{info.get('policy_type', '-')}, context {info.get('context', '-')}]"
                  f"\n    {info.get('city', '-')}, {info.get('country', '-')}: {n} measure{"s" if n != 1 else ""}"
                  f"\n    {' '.join(str(info.get('summary', '')).split())}\n")
        return
    p = _registry_path(a.registry)
    info = registry_info(p)
    print(f"{info.get('city', p.stem)} ({info.get('country', '-')}), "
          f"{info.get('policy_type', '-')}, context profile: {info.get('context', 'generic')}\n")
    for m in load_registry(p).values():
        if m.period:
            end = m.period[1] if m.period[1] not in (None, "", "open") else "open-ended"
            when = f"{m.period[0]} to {end}"
        else:
            when = "dates not set"
        hours = f", {m.window}" if m.window else ""
        again = f", {m.recurrence}" if getattr(m, "recurrence", None) else ""
        print(f"{m.id}  [{m.kind}, observability {m.observability}]\n"
              f"    {m.name}\n    {m.place}; {when}{hours}{again}\n"
              f"    measure: {' '.join(m.indicator.split())}\n"
              f"    video cannot see: {' '.join(m.video_cannot_see.split()) or '-'}\n")


def cmd_check(a: argparse.Namespace) -> None:
    from .rules import check_measure

    reg = _registry(a.registry)
    if a.measure not in reg:
        sys.exit(f"Unknown measure '{a.measure}'. Run `traffic-vision measures` for the list.")
    m = reg[a.measure]
    print(f"{m.name}\n{m.place}\n")
    try:
        print(check_measure(m, a.run_dir, a.zone, a.interval, a.start_clock))
    except ValueError as e:
        sys.exit(str(e))
    print(f"\nSources: {', '.join(m.sources)}")


def _series(run: Path, interval: float, classes: list[str] | None):
    from .analysis import flows, load_crossings

    dur = json.loads((run / "summary.json").read_text())["video"]["duration_s"]
    cx = load_crossings([run])
    cx = cx[cx.road_class.isin(classes or [c for c in ROAD_CLASSES if c != "pedestrian"])]
    return flows(cx, interval, dur, by=()).set_index("interval")["count"].to_numpy()


def cmd_did(a: argparse.Namespace) -> None:
    from .evaluation import (calibrate, calibration_from_files, diff_in_diff, ppi_diff_in_diff,
                             ppi_from_files)

    runs = [Path(r) for r in (a.treated_before, a.treated_during, a.control_before,
                              a.control_during)]
    if a.labels:
        lam = a.ppi_weight if a.ppi_weight in ("ratio", "tuned") else float(a.ppi_weight)
        est = ppi_from_files(a.labels, runs, a.classes, lam=lam)
        for name, e in zip(("treated before", "treated during", "control before",
                            "control during"), est):
            print(f"{name}: {e.mean:.2f} per window (network alone {e.naive_mean:.2f}), "
                  f"{e.n_labeled} hand-counted windows of {e.n_windows}, weight {e.lam:.2f}")
        print(ppi_diff_in_diff(est).as_text("objects per labelled window"))
        return
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


def cmd_design(a: argparse.Namespace) -> None:
    from .simulate import power_table

    rows = power_table(a.base, a.effect, tuple(a.hours), tuple(a.days), rho=a.rho,
                       recall=a.recall, interval_min=a.interval, n_studies=a.studies)
    print(f"Flow before the measure: {a.base:g} per {a.interval:g} min; expected change "
          f"{a.effect:+.0%}; autocorrelation {a.rho}; detector recall {a.recall}.\n"
          f"Chance of a 95% interval that excludes zero ({a.studies} simulated studies each):\n")
    print(f"{'days per period':>16} {'hours per day':>14} {'power':>7} {'coverage':>9} "
          f"{'half-width':>11}")
    for r in rows:
        print(f"{r['days']:>16} {r['hours_per_day']:>14g} {r['power']:>7.0%} "
              f"{r['coverage']:>9.0%} {r['ci_half_width']:>11.1f}")
    print("\nA design with power under 80% will often miss a real effect of that size.")


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
    r.add_argument("--context", help="context profile in contexts/ (dakar, lagos, kigali, "
                                     "generic) or a YAML path: maps local vehicle names")
    r.add_argument("--line-extra", dest="line_extra", nargs=3, action="append",
                   metavar=("NAME", "X,Y", "X,Y"),
                   help="extra named counting line (for turning movements A>B); repeat")
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

    ms = sub.add_parser("measures", help="list the measures of a registry and what video can measure")
    ms.add_argument("--registry", help="registry name in measures/ or a YAML path "
                                       "(default dakar_joj_2026)")
    ms.add_argument("--list", action="store_true", help="list every registry in measures/")
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
    d.add_argument("--labels", help="hand counts for every recording (run,start_s,end_s,"
                                    "manual_count): prediction-powered estimate per recording")
    d.add_argument("--ppi-weight", dest="ppi_weight", default="ratio",
                   help="weight of the network's counts: ratio (default), tuned or a number")
    d.set_defaults(func=cmd_did)

    cb = sub.add_parser("calibrate", help="compare a manual count with the automatic count")
    cb.add_argument("manual_csv")
    cb.add_argument("run_dir")
    cb.set_defaults(func=cmd_calibrate)

    ds = sub.add_parser("design", help="how many hours and days to film to detect an effect")
    ds.add_argument("--base", type=float, required=True,
                    help="flow per interval at the treated site before the measure")
    ds.add_argument("--effect", type=float, required=True,
                    help="expected change as a share (0.2 = +20%%, -0.3 = -30%%)")
    ds.add_argument("--hours", type=float, nargs="+", default=[2, 4, 8])
    ds.add_argument("--days", type=int, nargs="+", default=[1, 3, 7])
    ds.add_argument("--interval", type=float, default=5.0, help="interval length in minutes")
    ds.add_argument("--rho", type=float, default=0.6, help="autocorrelation between intervals")
    ds.add_argument("--recall", type=float, default=0.82, help="detector recall")
    ds.add_argument("--studies", type=int, default=300, help="simulated studies per cell")
    ds.set_defaults(func=cmd_design)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
