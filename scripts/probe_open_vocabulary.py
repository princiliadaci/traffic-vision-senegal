"""Do detection models name Dakar's vehicles correctly?

A small probe, not a benchmark: four photos of Dakar (docs/img), the clearly
visible vehicles boxed by hand (docs/results/probe_annotations.yaml), and three
ways of detecting them:

1. a closed-vocabulary detector trained on COCO (YOLO11n, 80 classes);
2. an open-vocabulary detector without prompts (YOLOE-11S prompt-free, which
   names objects from a vocabulary of 4,585 classes);
3. the same open-vocabulary detector with a one-shot visual prompt: one box
   per local vehicle type drawn on one photo, applied to another photo.

For each hand-drawn box the script keeps the detection that overlaps it most
(IoU of at least 0.3, one detection per box), maps its label to the tool's
core classes (traffic_vision.taxonomy, with the Dakar context profile) and
checks the class. Writes docs/results/open_vocabulary_probe.md and
open_vocabulary_probe.png.

    python scripts/probe_open_vocabulary.py --pf yoloe-11s-seg-pf.pt --vp yoloe-11s-seg.pt

Weights: yolo11n.pt and the two YOLOE files are downloaded by Ultralytics from
its GitHub releases on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traffic_vision.taxonomy import context_aliases, load_context, to_road_class  # noqa: E402

CONF = 0.25
MIN_IOU = 0.3


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def detections(result) -> list[dict]:
    names = result.names
    out = []
    for box, c, p in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                         result.boxes.conf.cpu().numpy()):
        out.append({"box": [float(v) for v in box], "label": names[int(c)], "conf": float(p)})
    return out


def match(refs: list[dict], dets: list[dict]) -> list[dict | None]:
    """Greedy one-to-one matching by IoU, best pairs first."""
    pairs = sorted(((iou(r["box"], d["box"]), i, j) for i, r in enumerate(refs)
                    for j, d in enumerate(dets)), reverse=True)
    used_r, used_d, out = set(), set(), [None] * len(refs)
    for v, i, j in pairs:
        if v < MIN_IOU:
            break
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        out[i] = {**dets[j], "iou": v}
    return out


def verdict(ref: dict, det: dict | None, aliases: dict) -> tuple[str, str]:
    if det is None:
        return "-", "missed" if ref["core"] != "none" else "ignored (right)"
    core = to_road_class(det["label"], aliases)
    if ref["core"] == "none":
        return core or "not a road user", "counted (wrong)" if core else "ignored (right)"
    return core or "not a road user", "right" if core == ref["core"] else "wrong"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=str(ROOT / "docs/results/probe_annotations.yaml"))
    ap.add_argument("--coco", default="yolo11n.pt")
    ap.add_argument("--pf", default="yoloe-11s-seg-pf.pt", help="YOLOE prompt-free weights")
    ap.add_argument("--vp", default="yoloe-11s-seg.pt", help="YOLOE weights for visual prompts")
    ap.add_argument("--context", default="dakar")
    ap.add_argument("--out", default=str(ROOT / "docs/results"))
    a = ap.parse_args()

    from ultralytics import YOLO, YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    spec = yaml.safe_load(Path(a.annotations).read_text())
    aliases = context_aliases(load_context(a.context))
    models = {"COCO (YOLO11n)": YOLO(a.coco), "Open vocabulary, no prompt (YOLOE-11S-PF)":
              YOLOE(a.pf)}

    rows, drawn = [], {}
    for item in spec["images"]:
        img = str(ROOT / item["file"])
        refs = item["vehicles"]
        drawn[item["file"]] = [{"ref": r, "labels": {}} for r in refs]
        for mname, model in models.items():
            res = model.predict(img, conf=CONF, verbose=False)[0]
            for k, (ref, det) in enumerate(zip(refs, match(refs, detections(res)))):
                core, v = verdict(ref, det, aliases)
                label = f"{det['label']} ({det['conf']:.2f})" if det else "nothing"
                rows.append((Path(item["file"]).name, ref["local"], ref["core"], mname, label,
                             core, v))
                drawn[item["file"]][k]["labels"][mname] = (det["label"] if det else "nothing", v)

    vp = spec.get("visual_prompt")
    vp_rows = []
    if vp:
        names = [c["name"] for c in vp["classes"]]
        prompts = {"bboxes": np.array([c["box"] for c in vp["classes"]], dtype=float),
                   "cls": np.arange(len(names))}
        model = YOLOE(a.vp)
        for target in vp["targets"]:
            res = model.predict(str(ROOT / target), refer_image=str(ROOT / vp["reference"]),
                                visual_prompts=prompts, predictor=YOLOEVPSegPredictor,
                                conf=CONF, verbose=False)[0]
            res.names = dict(enumerate(names))
            refs = next(i["vehicles"] for i in spec["images"] if i["file"] == target)
            for ref, det in zip(refs, match(refs, detections(res))):
                got = det["label"] if det else "nothing"
                want = {"car rapide": "car rapide", "city bus": "city bus",
                        "taxi": "taxi"}.get(ref["local"])
                ok = "-" if want is None else ("right" if got == want else
                                               ("missed" if det is None else "wrong"))
                conf = f" ({det['conf']:.2f})" if det else ""
                vp_rows.append((Path(target).name, ref["local"], got + conf, ok))

    write_markdown(Path(a.out), rows, vp_rows, spec)
    draw(Path(a.out), drawn, list(models))


def write_markdown(out: Path, rows, vp_rows, spec) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Do detection models name Dakar's vehicles correctly?", "",
             "Generated by `python scripts/probe_open_vocabulary.py`. A probe on four "
             "photos, not a benchmark: it shows the kind of error to expect, not its rate.",
             f"Confidence threshold {CONF}, a detection matches a hand-drawn box when their "
             f"IoU is at least {MIN_IOU}.", ""]
    for mname in dict.fromkeys(r[3] for r in rows):
        sub = [r for r in rows if r[3] == mname and r[2] != "none"]
        n_ok = sum(r[6] == "right" for r in sub)
        n_miss = sum(r[6] == "missed" for r in sub)
        lines.append(f"- {mname}: {n_ok} of {len(sub)} vehicles in the right class, "
                     f"{n_miss} missed.")
    lines += ["", "## Every vehicle", "",
              "| Photo | Vehicle (local name) | Should count as | Model | Label given | "
              "Counted as | Result |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    if vp_rows:
        ref = spec["visual_prompt"]["reference"]
        lines += ["", "## One-shot visual prompt", "",
                  f"One box per type drawn on `{Path(ref).name}` "
                  f"({', '.join(c['name'] for c in spec['visual_prompt']['classes'])}), "
                  "then applied to a photo the prompt has not seen.", "",
                  "| Photo | Vehicle | Label given | Result |", "|---|---|---|---|"]
        for r in vp_rows:
            lines.append("| " + " | ".join(r) + " |")
    (out / "open_vocabulary_probe.md").write_text("\n".join(lines) + "\n")


def draw(out: Path, drawn: dict, model_names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    files = [f for f, v in drawn.items() if len(v) > 1][:2]
    fig = plt.figure(figsize=(7 * len(files), 7.4))
    grid = fig.add_gridspec(2, len(files), height_ratios=[3, 1.05], hspace=0.04, wspace=0.05)
    short = {model_names[0]: "COCO", model_names[1]: "open vocabulary"}
    red, green = "#c0392b", "#1e8449"
    for col, f in enumerate(files):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(Image.open(ROOT / f))
        ax.set_axis_off()
        ax.set_title(Path(f).name, fontsize=10, loc="left")
        notes, badges = [], []
        refs = [d for d in drawn[f] if d["ref"]["core"] != "none"]
        for k, d in enumerate(refs, 1):
            r = d["ref"]
            x1, y1, x2, y2 = r["box"]
            wrong = any(v != "right" for _, v in d["labels"].values())
            colour = red if wrong else green
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2,
                                   edgecolor=colour, linestyle="-" if wrong else "--"))
            bx, by, va = x1 + 4, y1 + 4, "top"
            if any(abs(bx - px) < 40 and abs(by - py) < 30 for px, py in badges):
                bx, by, va = x1 + 4, y2 - 4, "bottom"      # corner taken: use the bottom
            badges.append((bx, by))
            ax.text(bx, by, str(k), fontsize=9, fontweight="bold", va=va,
                    color="white", bbox={"facecolor": colour, "pad": 2.5, "lw": 0})
            given = ", ".join(f"{short[m]} \u201c{d['labels'][m][0]}\u201d" for m in model_names)
            mark = "WRONG" if wrong else "ok"
            notes.append((f"{k}. {r['local']} (counts as {r['core']}): {given}", mark, colour))
        tx = fig.add_subplot(grid[1, col])
        tx.set_axis_off()
        for i, (line, mark, colour) in enumerate(notes):
            y = 1 - i / max(len(notes), 6)
            tx.text(0.0, y, mark, fontsize=8.5, fontweight="bold", color=colour, va="top",
                    transform=tx.transAxes)
            tx.text(0.09, y, line, fontsize=8.5, va="top", transform=tx.transAxes)
    fig.suptitle("Hand-drawn boxes and the label given by each model. Solid red box: at least "
                 "one model put the vehicle in the wrong class", fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Photos: Lisa Coulaud, CC BY-SA 4.0 (left); Ji-Elle, public domain "
             "(right); Wikimedia Commons. This figure: CC BY-SA 4.0.", fontsize=7,
             color="#666666")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    path = out / "open_vocabulary_probe.png"
    fig.savefig(path, dpi=110)
    # 256-colour palette: the photos make a full-colour PNG five times larger
    Image.open(path).convert("RGB").quantize(256, dither=Image.Dither.FLOYDSTEINBERG) \
        .save(path, optimize=True)


if __name__ == "__main__":
    main()
