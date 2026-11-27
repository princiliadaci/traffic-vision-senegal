# Context profiles

A profile describes the vehicles and the legal setting of a city, so the same
code can count a car rapide in Dakar, a keke in Lagos or a moto-taxi in
Kigali.

| File | City | Local vehicle types |
|---|---|---|
| [`dakar.yaml`](dakar.yaml) | Dakar and the Petite-Côte (Senegal) | car rapide, Ndiaga Ndiaye, Tata minibus, Dakar Dem Dikk and BRT buses, taxi, jakarta, tricycle, charrette, pousse-pousse |
| [`lagos.yaml`](lagos.yaml) | Lagos (Nigeria) | danfo, BRT buses, keke, okada, taxi |
| [`kigali.yaml`](kigali.yaml) | Kigali (Rwanda) | moto-taxi, city bus, taxi voiture |
| [`generic.yaml`](generic.yaml) | start here | |

```bash
traffic-vision run --video my_video.mp4 --context dakar ...
```

## Fields

| Field | Meaning |
|---|---|
| `local_vehicle_types` | for each type: `name`, `core_class` (one of `pedestrian`, `bicycle`, `motorcycle`, `tricycle`, `car`, `bus`, `truck`, `cart`), `model_labels` (labels of a detection model counted in that class), and optional `description` and `detection_notes` |
| `extra_aliases` | other label to class pairs |
| `pce` | passenger car equivalents of the traffic authority; empty for the illustrative defaults |
| `legal` | `data_protection` law, the `authority` that enforces it, the `transport_authority` |
| `registries` | measure registries for this city in [`measures/`](../measures) |

Labels listed in a profile are used before the default mapping of
[`traffic_vision/taxonomy.py`](../traffic_vision/taxonomy.py). A label can map
to only one class: if a model gives the same label to two local vehicles that
belong to different classes (the car rapide and real trucks are both "trailer
truck" for YOLOE), renaming cannot separate them. Write it in
`detection_notes` and use hand-counted windows (`traffic-vision did --labels`)
or a fine-tuned model.

The Lagos and Kigali profiles are first drafts written from public sources;
check names and detection notes with local partners before a study.
