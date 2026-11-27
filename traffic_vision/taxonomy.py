"""One set of road-user classes, whatever the detection model and the city.

Detection models name the same objects differently: COCO says "motorcycle",
BDD100K says "motor", VisDrone says "motor" or "van", open-vocabulary models
say "moped", "minibus" or "horse cart". The rest of the project only sees the
eight core classes below, so counts from different models, scenes and cities
can be merged.

The two last classes matter in many African cities and are absent from COCO:
three-wheelers (keke, bajaj, tuk-tuk, Saly tricycles) and carts (horse or
donkey carts, handcarts). A context profile (``contexts/*.yaml``) can add
local aliases, for example "minibus" -> bus for the cars rapides of Dakar or
the danfos of Lagos.
"""

from __future__ import annotations

from pathlib import Path

ROAD_CLASSES: tuple[str, ...] = (
    "pedestrian",
    "bicycle",
    "motorcycle",
    "tricycle",
    "car",
    "bus",
    "truck",
    "cart",
)

# Classes whose occupants should not be counted as pedestrians.
RIDDEN_CLASSES: frozenset[str] = frozenset({"bicycle", "motorcycle", "tricycle", "cart"})

# Source label (lower case) -> core class. Labels not listed are ignored.
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
    "rider": "pedestrian",  # a person on a bike or motorbike: dropped by drop_riders
                            # when the vehicle is detected too, so it is not counted twice
    "bike": "bicycle",
    "motor": "motorcycle",
    # VisDrone
    "people": "pedestrian",
    "van": "car",
    "tricycle": "tricycle",
    "awning-tricycle": "tricycle",
    # Open-vocabulary models (labels of the YOLOE prompt-free vocabulary)
    "moped": "motorcycle",
    "scooter": "motorcycle",
    "motorcycle racer": "pedestrian",  # the person, not the motorbike (see "rider")
    "rickshaw": "tricycle",
    "auto rickshaw": "tricycle",
    "taxi": "car",
    "suv": "car",
    "family car": "car",
    "police car": "car",
    "minivan": "car",
    "police van": "car",
    "sports car": "car",
    "jeep": "car",
    "minibus": "bus",
    "tour bus": "bus",
    "school bus": "bus",
    "double-decker bus": "bus",
    "trailer truck": "truck",
    "tow truck": "truck",
    "garbage truck": "truck",
    "fire truck": "truck",
    "pickup truck": "truck",
    "dump truck": "truck",
    "horse cart": "cart",
    "cart": "cart",
    "handcart": "cart",
    "wagon": "cart",
}


def data_dir(name: str) -> Path:
    """Folder of the repository (``measures``, ``contexts``), wherever the code runs.

    Looks next to the package first (a clone), then in the working directory
    (an installed package run from the repository, as in the Docker image).
    """
    here = Path(__file__).resolve().parent.parent / name
    return here if here.exists() else Path.cwd() / name


def load_context(path: str | Path | None) -> dict:
    """Read a context profile (YAML). ``None`` gives an empty profile."""
    if path is None:
        return {}
    import yaml

    p = Path(path)
    if not p.exists():
        candidate = data_dir("contexts") / f"{path}.yaml"
        if candidate.exists():
            p = candidate
        else:
            raise FileNotFoundError(f"Context profile not found: {path}")
    return yaml.safe_load(p.read_text()) or {}


def context_aliases(context: dict) -> dict[str, str]:
    """Extra label -> core class pairs declared by a context profile."""
    out = {}
    for vt in context.get("local_vehicle_types", []):
        for label in vt.get("model_labels", []):
            out[label.strip().lower()] = vt["core_class"]
    for label, core in (context.get("extra_aliases") or {}).items():
        out[label.strip().lower()] = core
    bad = {k: v for k, v in out.items() if v not in ROAD_CLASSES}
    if bad:
        raise ValueError(f"Context maps labels to unknown classes: {bad}")
    return out


def to_road_class(label: str, extra: dict[str, str] | None = None) -> str | None:
    """Return the core class for a model label, or None if it is not a road user."""
    key = label.strip().lower()
    if extra and key in extra:
        return extra[key]
    return _ALIASES.get(key)


def model_class_ids(model_names: dict[int, str], keep: set[str] | None = None,
                    extra: dict[str, str] | None = None) -> dict[int, str]:
    """Map the ids of a model to core classes.

    Args:
        model_names: the ``names`` attribute of an Ultralytics model.
        keep: core classes to keep; all of them when None.
        extra: label -> class pairs from a context profile, checked first.

    Returns:
        {model class id: core class} for the classes to track.
    """
    keep = set(ROAD_CLASSES) if keep is None else set(keep)
    unknown = keep - set(ROAD_CLASSES)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}. Choose from {ROAD_CLASSES}.")
    mapping = {}
    for idx, label in model_names.items():
        road = to_road_class(label, extra)
        if road is not None and road in keep:
            mapping[int(idx)] = road
    return mapping
