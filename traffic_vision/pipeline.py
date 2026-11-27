"""Detect, track and count road users in a video.

Detection uses an Ultralytics model (YOLO11 trained on COCO by default; a
fine-tuned model or the open-vocabulary YOLOE prompt-free model work too) and
tracking uses ByteTrack. Model labels are mapped to eight core classes, with
the local names of a context profile first (``contexts/``). Each run writes a
folder with:

    annotated.mp4    the video with boxes, track ids, the counting line and live counters
    detections.csv   one row per tracked object per processed frame
    crossings.csv    one row per object crossing the counting line or an extra line
    zone_events.csv  one row per object using a zone (a reserved lane), if zones are set
    summary.json     run settings and totals

With ``start_clock`` (the clock time of the first frame), every event also
gets a ``clock_time``, which rules with hours (a truck ban from 7:00 to 19:00)
need.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .counting import CountingLine
from .taxonomy import RIDDEN_CLASSES, ROAD_CLASSES, context_aliases, load_context, model_class_ids
from .zones import Zone, ZoneCounter

log = logging.getLogger(__name__)

COLOURS = {  # BGR
    "pedestrian": (80, 200, 80),
    "bicycle": (200, 200, 60),
    "motorcycle": (40, 140, 255),
    "car": (230, 120, 40),
    "bus": (60, 60, 230),
    "truck": (180, 80, 180),
    "tricycle": (0, 180, 180),
    "cart": (60, 100, 140),
}

DETECTION_FIELDS = [
    "scene_id", "video", "frame", "video_time_s", "track_id", "road_class", "frame_class",
    "model_label", "confidence", "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "speed_px_s",
]
CROSSING_FIELDS = ["scene_id", "video", "line", "track_id", "road_class",
                   "direction", "frame", "video_time_s", "clock_time"]
ZONE_FIELDS = ["scene_id", "video", "zone", "track_id", "road_class", "frame",
               "video_time_s", "clock_time"]
ZONE_COLOUR = (0, 200, 255)  # BGR, amber


@dataclass
class RunConfig:
    video: str
    scene_id: str = "scene"
    model: str = "yolo11n.pt"
    classes: list[str] = field(default_factory=lambda: list(ROAD_CLASSES))
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 960
    stride: int = 1  # process one frame out of `stride`
    line: tuple[tuple[float, float], tuple[float, float]] = ((0.05, 0.6), (0.95, 0.6))
    line_directions: tuple[str, str] = ("in", "out")
    line_margin: float = 0.01  # dead band around the line, fraction of frame height
    max_seconds: float | None = None
    output_dir: str = "outputs"
    save_video: bool = True
    tracker: str = "bytetrack.yaml"
    device: str | None = None
    drop_riders: bool = True  # a person on a motorbike or bicycle is not a pedestrian
    zones: dict[str, list[list[float]]] | None = None  # name -> polygon, fractions of the frame
    zone_min_frames: int = 2  # frames inside a zone before an object counts as using it
    start_clock: str | None = None  # clock time of the first frame, ISO, e.g. 2026-11-07T07:00:00
    extra_lines: dict[str, list[list[float]]] | None = None  # name -> [[x, y], [x, y]], fractions
    context: str | None = None  # context profile: name in contexts/ or path to a YAML file


def drop_riders(boxes: list[dict], min_overlap: float = 0.3) -> list[dict]:
    """Remove "pedestrian" boxes that are people riding a two-wheeler or a cart.

    COCO models detect the rider and the motorbike as two objects. In cities
    where motorbikes are everywhere, counting riders as pedestrians would
    inflate pedestrian counts. A person box is dropped when it overlaps a
    bicycle, motorcycle, tricycle or cart box by at least ``min_overlap`` of its
    own area and its feet are inside that box.
    """
    wheels = [b["xyxy"] for b in boxes if b["road_class"] in RIDDEN_CLASSES]
    if not wheels:
        return boxes
    kept = []
    for b in boxes:
        if b["road_class"] == "pedestrian":
            px1, py1, px2, py2 = b["xyxy"]
            area = max((px2 - px1) * (py2 - py1), 1)
            rider = False
            for wx1, wy1, wx2, wy2 in wheels:
                iw = max(0, min(px2, wx2) - max(px1, wx1))
                ih = max(0, min(py2, wy2) - max(py1, wy1))
                if iw * ih / area >= min_overlap and wy1 <= py2 <= wy2 + 0.1 * (wy2 - wy1):
                    rider = True
                    break
            if rider:
                continue
        kept.append(b)
    return kept


@dataclass
class FrameResult:
    frame_index: int
    video_time_s: float
    image: np.ndarray
    objects_in_view: int
    totals: dict[str, dict[str, int]]


def _draw(frame: np.ndarray, boxes: list[dict], line: CountingLine,
          totals: dict[str, dict[str, int]], directions: tuple[str, str], t: float,
          zones: ZoneCounter | None = None,
          extra_lines: list[CountingLine] | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, w / 1600)
    if zones is not None and zones.zones:
        fill = out.copy()
        zt = zones.totals()
        for z in zones.zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.fillPoly(fill, [pts], ZONE_COLOUR)
        out = cv2.addWeighted(fill, 0.28, out, 0.72, 0)
        for z in zones.zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, ZONE_COLOUR, max(2, int(3 * scale)))
    a = tuple(int(v) for v in line.a)
    b = tuple(int(v) for v in line.b)
    cv2.line(out, a, b, (0, 230, 255), max(2, int(3 * scale)))
    for ln in extra_lines or []:
        ea = tuple(int(v) for v in ln.a)
        eb = tuple(int(v) for v in ln.b)
        cv2.line(out, ea, eb, (255, 200, 0), max(2, int(3 * scale)))
        cv2.putText(out, ln.name, (ea[0] + 4, ea[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * scale, (255, 200, 0), 2, cv2.LINE_AA)

    for bx in boxes:
        c = COLOURS.get(bx.get("track_class", bx["road_class"]), (255, 255, 255))
        x1, y1, x2, y2 = bx["xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, max(1, int(2 * scale)))
        label = f'{bx.get("track_class", bx["road_class"])} #{bx["track_id"]}'
        fs = 0.5 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Live counter panel
    rows = [f"t = {t:6.1f} s", f"{directions[0]:>10} {directions[1]:>6}"]
    for cls in ROAD_CLASSES:
        d = totals.get(cls)
        if d:
            rows.append(f"{cls:<10} {d.get(directions[0], 0):>4} {d.get(directions[1], 0):>6}")
    if zones is not None:
        for zname, per_class in zones.totals().items():
            detail = ", ".join(f"{c} {n}" for c, n in sorted(per_class.items()))
            rows.append(f"{zname}: {detail}")
    if not boxes:
        rows.append("no selected object in view")
    fs = 0.55 * scale
    lh = int(26 * scale)
    pw = int(max(330, 11 * max(len(r) for r in rows)) * scale)
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + pw, 20 + lh * len(rows)), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for i, r in enumerate(rows):
        cv2.putText(out, r, (20, 10 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def iter_run(cfg: RunConfig, model=None) -> Iterator[FrameResult]:
    """Process the video and yield annotated frames as they are ready.

    Used both by :func:`run` (command line) and by the Streamlit dashboard,
    which shows frames live. Files are written while the generator runs.
    """
    from ultralytics import YOLO  # imported here so that tests do not need torch

    model = model or YOLO(cfg.model)
    ids = model_class_ids(model.names, set(cfg.classes),
                          context_aliases(load_context(cfg.context)))
    if not ids:
        raise ValueError("The model has none of the selected classes.")

    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cfg.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(cfg.output_dir) / cfg.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    line = CountingLine.from_relative(cfg.line[0], cfg.line[1], w, h, name="line_1",
                                      margin_frac=cfg.line_margin,
                                      directions=tuple(cfg.line_directions))
    extra_lines = [CountingLine.from_relative(tuple(pts[0]), tuple(pts[1]), w, h, name=name,
                                              margin_frac=cfg.line_margin)
                   for name, pts in (cfg.extra_lines or {}).items()]
    zone_counter = None
    if cfg.zones:
        zone_counter = ZoneCounter([Zone.from_relative(name, [tuple(p) for p in pts], w, h)
                                    for name, pts in cfg.zones.items()], cfg.zone_min_frames)
    t_start = datetime.fromisoformat(cfg.start_clock) if cfg.start_clock else None

    def clock(t: float) -> str:
        return (t_start + timedelta(seconds=t)).isoformat(timespec="seconds") if t_start else ""

    writer = None
    if cfg.save_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / cfg.stride, (w, h))

    det_f = open(out_dir / "detections.csv", "w", newline="")
    cross_f = open(out_dir / "crossings.csv", "w", newline="")
    det_w = csv.DictWriter(det_f, DETECTION_FIELDS)
    cross_w = csv.DictWriter(cross_f, CROSSING_FIELDS)
    det_w.writeheader()
    cross_w.writeheader()
    zone_f = zone_w = None
    if zone_counter is not None:
        zone_f = open(out_dir / "zone_events.csv", "w", newline="")
        zone_w = csv.DictWriter(zone_f, ZONE_FIELDS)
        zone_w.writeheader()

    video_name = Path(cfg.video).name
    seen_tracks: dict[str, set[int]] = {}
    votes: dict[int, dict[str, float]] = {}  # track -> class -> summed confidence
    last_anchor: dict[int, tuple[float, float, float]] = {}  # track -> (x, y, t)
    empty_frames = processed = 0
    t0 = time.time()
    frame_index = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            t = frame_index / fps
            if cfg.max_seconds is not None and t > cfg.max_seconds:
                break
            if frame_index % cfg.stride:
                continue
            processed += 1
            res = model.track(frame, persist=True, tracker=cfg.tracker, conf=cfg.confidence,
                              iou=cfg.iou, imgsz=cfg.imgsz, classes=list(ids), verbose=False,
                              device=cfg.device)[0]
            raw = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                tids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, tids, clss, confs):
                    road = ids.get(int(c))
                    if road is not None:
                        raw.append({"xyxy": (int(x1), int(y1), int(x2), int(y2)),
                                    "track_id": int(tid), "road_class": road,
                                    "model_label": model.names[int(c)], "confidence": float(cf)})
            boxes = drop_riders(raw) if cfg.drop_riders else raw
            for bx in boxes:
                # A track can flicker between classes (a bus half hidden by a pole is
                # read as a car or a truck for a few frames). Events use the class
                # with the highest summed confidence so far for that track.
                v = votes.setdefault(bx["track_id"], {})
                v[bx["road_class"]] = v.get(bx["road_class"], 0.0) + bx["confidence"]
                bx["track_class"] = max(v, key=v.get)
            for bx in boxes:
                x1, y1, x2, y2 = bx["xyxy"]
                tid, road = bx["track_id"], bx["track_class"]
                anchor = ((x1 + x2) / 2, float(y2))  # where the object touches the road
                seen_tracks.setdefault(road, set()).add(tid)
                speed = ""
                prev = last_anchor.get(tid)
                if prev is not None and t > prev[2]:
                    speed = round(((anchor[0] - prev[0]) ** 2 + (anchor[1] - prev[1]) ** 2) ** 0.5
                                  / (t - prev[2]), 1)
                last_anchor[tid] = (anchor[0], anchor[1], t)
                det_w.writerow({
                    "scene_id": cfg.scene_id, "video": video_name, "frame": frame_index,
                    "video_time_s": round(t, 3), "track_id": tid, "road_class": road,
                    "model_label": bx["model_label"], "confidence": round(bx["confidence"], 4),
                    "frame_class": bx["road_class"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "anchor_x": round(anchor[0], 1), "anchor_y": round(anchor[1], 1),
                    "speed_px_s": speed,
                })
                for ln in (line, *extra_lines):
                    ev = ln.update(tid, road, anchor, frame_index, t)
                    if ev:
                        cross_w.writerow({"scene_id": cfg.scene_id, "video": video_name,
                                          "line": ln.name, **asdict(ev),
                                          "clock_time": clock(ev.video_time_s)})
                if zone_counter is not None:
                    for zev in zone_counter.update(tid, road, anchor, frame_index, t):
                        zone_w.writerow({"scene_id": cfg.scene_id, "video": video_name,
                                         **asdict(zev), "clock_time": clock(zev.video_time_s)})
            if not boxes:
                empty_frames += 1
            totals = line.totals()
            img = _draw(frame, boxes, line, totals, line.directions, t, zone_counter, extra_lines)
            if writer is not None:
                writer.write(img)
            if processed % 100 == 0:
                log.info("frame %d/%d, %.1f frames/s", frame_index, n_frames,
                         processed / max(time.time() - t0, 1e-6))
            yield FrameResult(frame_index, t, img, len(boxes), totals)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        det_f.close()
        cross_f.close()
        if zone_f is not None:
            zone_f.close()

    duration = (frame_index + 1) / fps
    summary = {
        "config": {**asdict(cfg), "line": [list(p) for p in cfg.line]},
        "video": {"fps": fps, "width": w, "height": h, "frames_read": frame_index + 1,
                  "duration_s": round(duration, 2)},
        "frames_processed": processed,
        "frames_without_selected_objects": empty_frames,
        "unique_tracks_seen": {k: len(v) for k, v in sorted(seen_tracks.items())},
        "line_crossings": line.totals(),
        "extra_line_crossings": {ln.name: ln.totals() for ln in extra_lines},
        "zone_use": zone_counter.totals() if zone_counter is not None else {},
        "processing_seconds": round(time.time() - t0, 1),
        "author": "Traffic Vision Senegal, Princilia DACI",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Results in %s", out_dir)


def run(cfg: RunConfig, on_frame: Callable[[FrameResult], None] | None = None) -> Path:
    """Process a whole video; return the output folder."""
    for fr in iter_run(cfg):
        if on_frame:
            on_frame(fr)
    return Path(cfg.output_dir) / cfg.scene_id
