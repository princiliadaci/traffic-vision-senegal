"""Fine-tune a YOLO model on road-traffic images.

The default COCO weights (yolo11n.pt) already know cars, buses, trucks,
motorcycles, bicycles and people. Fine-tuning helps when the scene looks
different from COCO: heavy two-wheeler traffic, minibuses such as Dakar's
"cars rapides" and "Ndiaga Ndiaye", night video, or high camera angles.

Datasets that fit:
  - BDD100K in YOLO format (driving scenes), for example on Kaggle;
  - VisDrone2019-DET (high viewpoint), downloadable through Ultralytics
    with `data=VisDrone.yaml`;
  - your own images from Dakar, labelled with the six classes of
    traffic_vision.taxonomy.ROAD_CLASSES (CVAT or Label Studio export to YOLO).

Usage:
    python scripts/train_yolo.py --data path/to/data.yaml --epochs 50
    python scripts/train_yolo.py --data VisDrone.yaml --model yolo11s.pt --imgsz 1024

After training, evaluate on held-out images and keep the metrics with the
weights (runs/detect/<name>/results.csv) before using them in a study.

Author: Princilia DACI
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="dataset YAML in Ultralytics format")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="dakar_traffic")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name,
                device=a.device, patience=15, plots=True)
    metrics = model.val()
    print(f"mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f} | "
          f"precision {metrics.box.mp:.3f} | recall {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
