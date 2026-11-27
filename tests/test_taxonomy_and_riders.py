import pytest

from traffic_vision.taxonomy import model_class_ids, to_road_class

COCO_SUBSET = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
               9: "traffic light"}
BDD = {0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck", 5: "bike", 6: "motor",
       7: "traffic light", 8: "traffic sign", 9: "train"}


def test_coco_ids_are_mapped_by_name_not_by_position():
    ids = model_class_ids(COCO_SUBSET)
    assert ids[3] == "motorcycle" and ids[5] == "bus" and ids[7] == "truck"
    assert 9 not in ids  # traffic lights are not road users


def test_bdd_names():
    ids = model_class_ids(BDD)
    assert ids[6] == "motorcycle" and ids[5] == "bicycle" and ids[3] == "bus"
    assert ids[1] == "pedestrian"  # the rider, removed later by drop_riders
    assert 9 not in ids  # trains are ignored


def test_keep_subset_and_unknown_class():
    assert set(model_class_ids(COCO_SUBSET, {"bus"}).values()) == {"bus"}
    with pytest.raises(ValueError):
        model_class_ids(COCO_SUBSET, {"tram"})


def test_to_road_class_case():
    assert to_road_class("Person") == "pedestrian"
    assert to_road_class("kite") is None


def test_drop_riders():
    pytest.importorskip("cv2")
    from traffic_vision.pipeline import drop_riders

    moto = {"xyxy": (100, 150, 160, 260), "track_id": 1, "road_class": "motorcycle"}
    rider = {"xyxy": (110, 100, 150, 230), "track_id": 2, "road_class": "pedestrian"}
    walker = {"xyxy": (400, 100, 430, 220), "track_id": 3, "road_class": "pedestrian"}
    kept = drop_riders([moto, rider, walker])
    assert [b["track_id"] for b in kept] == [1, 3]
