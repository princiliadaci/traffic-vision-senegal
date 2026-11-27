"""One set of road-user classes, whatever the detection model.

Detection models name the same objects differently: COCO says "motorcycle",
BDD100K says "motor", VisDrone says "motor" or "van". The rest of the project
only ever sees the six classes below, so counts from different models and
scenes can be merged.
"""

from __future__ import annotations

ROAD_CLASSES: tuple[str, ...] = (
    "pedestrian",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
)

# Source label (lower case) -> project class. Labels not listed are ignored.
_ALIASES: dict[str, str] = {
    # COCO (yolo11n.pt, yolov8n.pt, ...)
    "person": "pedestrian",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    # BDD100K
    "pedestrian": "pedestrian",
    "rider": "motorcycle",  # in BDD100K a rider sits on a bike or motorbike
    "bike": "bicycle",
    "motor": "motorcycle",
    # VisDrone
    "people": "pedestrian",
    "van": "car",
    "tricycle": "motorcycle",
    "awning-tricycle": "motorcycle",
}


def to_road_class(label: str) -> str | None:
    """Return the project class for a model label, or None if it is not a road user."""
    return _ALIASES.get(label.strip().lower())


def model_class_ids(model_names: dict[int, str], keep: set[str] | None = None) -> dict[int, str]:
    """Map the ids of a model to project classes.

    Args:
        model_names: the ``names`` attribute of an Ultralytics model.
        keep: project classes to keep; all road classes when None.

    Returns:
        {model class id: project class} for the classes to track.
    """
    keep = set(ROAD_CLASSES) if keep is None else set(keep)
    unknown = keep - set(ROAD_CLASSES)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}. Choose from {ROAD_CLASSES}.")
    mapping = {}
    for idx, label in model_names.items():
        road = to_road_class(label)
        if road is not None and road in keep:
            mapping[int(idx)] = road
    return mapping
