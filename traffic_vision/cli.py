"""Command line: `traffic-vision run | report | compare | aggregate`.

Examples
--------
    traffic-vision run --video data/corniche_before.mp4 --scene corniche_before \
        --line 0.05,0.62 0.95,0.62 --directions to_ouakam to_fann
    traffic-vision run --config configs/scenes.example.yaml --scene corniche_before
    traffic-vision report outputs/corniche_before
    traffic-vision compare outputs/corniche_before outputs/corniche_during --interval 300
    traffic-vision aggregate            # merge every run in outputs/ into one dataset
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

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
