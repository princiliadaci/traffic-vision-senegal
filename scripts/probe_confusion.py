"""Confusion matrices of the detector probe, by local vehicle type.

Reads docs/results/open_vocabulary_probe.md (written by
scripts/probe_open_vocabulary.py) and draws, for the COCO and the
open-vocabulary detector, the class each hand-boxed vehicle was counted in.

    python scripts/probe_confusion.py      # writes docs/results/open_vocabulary_confusion.png
                                           # and open_vocabulary_boxes.jpg (the two photos, boxed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"

GROUP = {"car rapide": "car rapide", "city bus": "city or BRT bus", "BRT bus": "city or BRT bus",
         "taxi": "car or taxi", "private car": "car or taxi", "truck": "truck",
         "scooter": "motorbike"}
ROWS = ["car rapide", "city or BRT bus", "car or taxi", "truck", "motorbike"]
COLS = ["bus", "car", "truck", "motorcycle", "missed"]
RIGHT = {(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)}
MODELS = {"COCO (YOLO11n)": "COCO, closed vocabulary (YOLO11n)",
          "Open vocabulary, no prompt (YOLOE-11S-PF)": "Open vocabulary, no prompt (YOLOE)"}
TEXT, MUTED, WRONG = "#191919", "#666666", "#d62728"


def matrices(md_path: Path) -> dict[str, np.ndarray]:
    lines = md_path.read_text().splitlines()
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines
            if ln.startswith("| dakar_")]
    rows = [r for r in rows if len(r) == 7]          # the per-vehicle table only
    out = {}
    for model in MODELS:
        m = np.zeros((len(ROWS), len(COLS)), int)
        for _photo, local, core, mod, _label, counted, result in rows:
            if mod != model or core == "none":
                continue
            m[ROWS.index(GROUP[local]), COLS.index("missed" if result == "missed" else counted)] += 1
        out[model] = m
    return out


def boxed_photos(md_path: Path, out: Path) -> None:
    """The two street photos with each hand-drawn box: solid red when at least one
    detector put the vehicle in the wrong class (or missed it), dashed green otherwise,
    and the labels the two detectors gave to the wrong ones."""
    import matplotlib.pyplot as plt
    import yaml
    from matplotlib.patches import Rectangle
    from PIL import Image

    spec = yaml.safe_load((RESULTS / "probe_annotations.yaml").read_text())
    lines = md_path.read_text().splitlines()
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines
            if ln.startswith("| dakar_")]
    rows = [r for r in rows if len(r) == 7]
    files = ["docs/img/dakar_car_rapide_hlm.jpg", "docs/img/dakar_traffic.jpg"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.95), gridspec_kw={"wspace": 0.02})
    for ax, f in zip(axes, files):
        item = next(i for i in spec["images"] if i["file"] == f)
        name = Path(f).name
        per_model = {m: [r for r in rows if r[0] == name and r[3] == m] for m in MODELS}
        ax.imshow(Image.open(ROOT / f))
        ax.set_axis_off()
        for k, v in enumerate(item["vehicles"]):
            res = {m: per_model[m][k] for m in MODELS}
            wrong = any(r[6] != "right" for r in res.values())
            x1, y1, x2, y2 = v["box"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.6 if wrong else 1.8,
                                   ls="-" if wrong else "--", ec=WRONG if wrong else "#2e7d32"))
            if wrong:
                labels = " / ".join(dict.fromkeys(r[4].split(" (")[0] for r in res.values()))
                right_edge = x2 > 700
                ax.text(x2 - 4 if right_edge else x1 + 4, y2 - 6 if y2 < 560 else y1 + 6,
                        f"{v['local']}: {labels}", fontsize=9.5, color="white", fontweight="bold",
                        va="bottom" if y2 < 560 else "top", ha="right" if right_edge else "left",
                        bbox={"facecolor": WRONG, "pad": 2.5, "lw": 0})
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out, dpi=250, facecolor="white", pil_kwargs={"quality": 90})
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mats = matrices(RESULTS / "open_vocabulary_probe.md")
    cmap = LinearSegmentedColormap.from_list("purple", ["#ffffff", "#e3d6f1", "#9b59b6", "#3f1d6b"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.06})
    names = ["bus", "car", "truck", "motorbike", "missed"]
    for ax, (model, m) in zip(axes, mats.items()):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=6, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not m[i, j]:
                    continue
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=16,
                        fontweight="bold", color="white" if m[i, j] >= 3 else TEXT)
                if (i, j) not in RIGHT:
                    ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                               ec=WRONG, lw=3))
        ax.set_xticks(range(len(names)), names, fontsize=12.5, color=TEXT)
        ax.set_yticks(range(len(ROWS)), ROWS if ax is axes[0] else [""] * len(ROWS),
                      fontsize=12.5, color=TEXT)
        ax.tick_params(length=0)
        ax.xaxis.set_ticks_position("top")
        ax.set_title(MODELS[model], fontsize=13, color=TEXT, y=-0.12, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="#d9d9d9", lw=1)
        ax.tick_params(which="minor", length=0)
    fig.text(0.01, 0.955, "true vehicle  ↓      counted as →", fontsize=12, color=MUTED)
    fig.text(0.99, 0.955, "red frame: wrong class or missed", fontsize=12, color=WRONG,
             ha="right")
    fig.subplots_adjust(left=0.17, right=0.995, top=0.83, bottom=0.12)
    fig.savefig(RESULTS / "open_vocabulary_confusion.png", dpi=300, facecolor="white")
    for model, m in mats.items():
        print(model, "right:", sum(m[i, j] for i, j in RIGHT), "of", m.sum())
    boxed_photos(RESULTS / "open_vocabulary_probe.md", RESULTS / "open_vocabulary_boxes.jpg")


if __name__ == "__main__":
    main()
