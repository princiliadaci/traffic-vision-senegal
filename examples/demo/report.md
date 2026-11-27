# Run report: demo_pexels

- Video: `12159587-hd_1920_1080_50fps.mp4`, 57 s, 1920x1080 at 50 fps
- Model: `yolo11n.pt`, confidence 0.35
- Frames without any selected object: 0
- Vehicles crossing the line: 47.2 per minute; 54.0 passenger car equivalents in total

## Crossings by class and direction

| road_class   |   away_from_camera |   towards_camera |
|:-------------|-------------------:|-----------------:|
| car          |                 27 |               12 |
| bus          |                  0 |                2 |
| truck        |                  2 |                2 |

## Share of vehicles by class

| road_class   | share   |
|:-------------|:--------|
| bus          | 4%      |
| car          | 87%     |
| truck        | 9%      |

## Exposure of vulnerable road users (per hour, extrapolated)

| road_class   |   per hour |
|:-------------|-----------:|
| pedestrian   |          0 |
| motorcycle   |          0 |
| bicycle      |          0 |

![Flow by class](flow_by_class.png)

## Use of zones (reserved lanes)

| zone      |   bus |   car |   truck |
|:----------|------:|------:|--------:|
| kerb_lane |     7 |    11 |       9 |

Each object is counted once per zone, after two processed frames inside it. Check a lane rule with `traffic-vision check`.
