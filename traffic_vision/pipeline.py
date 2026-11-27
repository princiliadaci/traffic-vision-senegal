"""Detect, track and count road users in a video.

Detection uses a YOLO model (YOLO11 by default) and tracking uses ByteTrack,
both through Ultralytics. Each run writes a folder with:

    annotated.mp4    the video with boxes, track ids, the counting line and live counters
    detections.csv   one row per tracked object per processed frame
    crossings.csv    one row per object crossing the counting line
    summary.json     run settings and totals
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .counting import CountingLine
from .taxonomy import ROAD_CLASSES, model_class_ids

log = logging.getLogger(__name__)

COLOURS = {  # BGR
    "pedestrian": (80, 200, 80),
    "bicycle": (200, 200, 60),
    "motorcycle": (40, 140, 255),
    "car": (230, 120, 40),
    "bus": (60, 60, 230),
    "truck": (180, 80, 180),
}

DETECTION_FIELDS = [
    "scene_id", "video", "frame", "video_time_s", "track_id", "road_class",
    "model_label", "confidence", "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "speed_px_s",
]
CROSSING_FIELDS = ["scene_id", "video", "line", "track_id", "road_class",
                   "direction", "frame", "video_time_s"]


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
    max_seconds: float | None = None
    output_dir: str = "outputs"
    save_video: bool = True
    tracker: str = "bytetrack.yaml"
    device: str | None = None
    drop_riders: bool = True  # a person on a motorbike or bicycle is not a pedestrian


def drop_riders(boxes: list[dict], min_overlap: float = 0.3) -> list[dict]:
    """Remove "pedestrian" boxes that are people riding a two-wheeler.

    COCO models detect the rider and the motorbike as two objects. In cities
    like Dakar, where motorbikes are everywhere, counting riders as pedestrians
    would inflate pedestrian counts. A person box is dropped when it overlaps a
    motorcycle or bicycle box by at least ``min_overlap`` of its own area and
    its feet are inside that box.
    """
    wheels = [b["xyxy"] for b in boxes if b["road_class"] in ("motorcycle", "bicycle")]
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
          totals: dict[str, dict[str, int]], directions: tuple[str, str], t: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, w / 1600)
    a = tuple(int(v) for v in line.a)
    b = tuple(int(v) for v in line.b)
    cv2.line(out, a, b, (0, 230, 255), max(2, int(3 * scale)))

    for bx in boxes:
        c = COLOURS.get(bx["road_class"], (255, 255, 255))
        x1, y1, x2, y2 = bx["xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, max(1, int(2 * scale)))
        label = f'{bx["road_class"]} #{bx["track_id"]}'
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
    if not boxes:
        rows.append("no selected object in view")
    fs = 0.55 * scale
    lh = int(26 * scale)
    pw = int(330 * scale)
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
    ids = model_class_ids(model.names, set(cfg.classes))
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
                                      directions=tuple(cfg.line_directions))
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

    video_name = Path(cfg.video).name
    seen_tracks: dict[str, set[int]] = {}
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
                x1, y1, x2, y2 = bx["xyxy"]
                tid, road = bx["track_id"], bx["road_class"]
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
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "anchor_x": round(anchor[0], 1), "anchor_y": round(anchor[1], 1),
                    "speed_px_s": speed,
                })
                ev = line.update(tid, road, anchor, frame_index, t)
                if ev:
                    cross_w.writerow({"scene_id": cfg.scene_id, "video": video_name,
                                      "line": line.name, **asdict(ev)})
            if not boxes:
                empty_frames += 1
            totals = line.totals()
            img = _draw(frame, boxes, line, totals, line.directions, t)
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

    duration = (frame_index + 1) / fps
    summary = {
        "config": {**asdict(cfg), "line": [list(p) for p in cfg.line]},
        "video": {"fps": fps, "width": w, "height": h, "frames_read": frame_index + 1,
                  "duration_s": round(duration, 2)},
        "frames_processed": processed,
        "frames_without_selected_objects": empty_frames,
        "unique_tracks_seen": {k: len(v) for k, v in sorted(seen_tracks.items())},
        "line_crossings": line.totals(),
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
