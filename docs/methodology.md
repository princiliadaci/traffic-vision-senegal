# Methodology

## 1. Detection

Each frame goes through a YOLO detector (YOLO11 nano by default, trained on
COCO). Model labels are mapped to six road-user classes (`pedestrian`,
`bicycle`, `motorcycle`, `car`, `bus`, `truck`) **by name**, through
`traffic_vision/taxonomy.py`. Mapping by name rather than by class number
matters: in COCO, id 3 is "motorcycle" while in BDD100K id 3 is "bus". The same
code therefore works with COCO weights, BDD100K weights, VisDrone weights or a
model fine-tuned on Dakar images.

People riding a motorbike or a bicycle are detected as "person" by COCO
models. Where two-wheelers are a large share of traffic, this would inflate
pedestrian counts, so a person box whose feet sit inside a two-wheeler box and
that overlaps it by at least 30% is treated as a rider and not counted
(`drop_riders`, can be turned off).

## 2. Tracking

ByteTrack (through Ultralytics) links detections from frame to frame and gives
each object an id. ByteTrack also uses low-confidence boxes to keep tracks
alive through short occlusions, which is common in dense traffic.

## 3. Counting

An object is counted once, when the bottom-centre of its box (its contact point
with the road) crosses a counting line drawn across the lanes. The side of the
line is given by the sign of a cross product; the direction of the crossing
gives the direction of travel. A box that jitters on the line is counted at
most once per direction. Counting crossings, not detections per frame, avoids
counting a slow vehicle many times.

Lines are given as fractions of the frame, so the same scene file works at any
resolution.

## 4. Indicators

From the crossings (`crossings.csv`):

- **flow**: crossings per interval, by class and direction, with empty
  intervals kept as zeros;
- **passenger car equivalents (PCE)**: flow weighted by class (illustrative
  defaults: car 1, bus and truck 2.5, motorcycle 0.5, bicycle 0.3, pedestrian 0);
  replace them with the weights used by the traffic authority;
- **vehicle shares**: share of each class among vehicles.

## 5. Comparing periods

`compare_periods` compares the mean flow per interval of two recordings (for
example before and during a measure) with a percentile bootstrap on the
difference of means (10,000 resamples). It reports the difference, the
relative change and a 95% confidence interval. When the interval includes 0,
the data do not show a clear change at that sample size.

Two cautions:

- intervals are resampled as if independent. Traffic is autocorrelated, so
  use intervals of 5 minutes or more and long recordings when possible;
- the comparison is only as good as the recordings: same place, same camera
  position, same weekday and time of day.

## 6. Data schema

`detections.csv`, one row per tracked object per processed frame:

| column | meaning |
|---|---|
| scene_id | scene identifier from the run |
| video | video file name |
| frame | frame index in the video |
| video_time_s | time in the video, in seconds |
| track_id | anonymous id given by the tracker |
| road_class | one of the six classes |
| model_label | label given by the model before mapping |
| confidence | detection confidence |
| x1, y1, x2, y2 | box corners in pixels |
| anchor_x, anchor_y | bottom-centre of the box |
| speed_px_s | speed of the anchor since the previous processed frame, in pixels per second (empty on the first frame of a track); converting to km/h needs a calibrated view |

`crossings.csv`, one row per counted crossing: `scene_id, video, line,
track_id, road_class, direction, frame, video_time_s`.

`summary.json`: run settings, video properties, frames without any selected
object, unique tracks seen per class and line totals.

Several scenes share the same schema, so their files can be concatenated into
one dataset for a city-wide dashboard.

## 7. Validating counts

Before using counts in a study, count one or two minutes of each video by hand
and compare. Report the relative error per class with the results.
