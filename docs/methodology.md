# Methodology

## 1. Detection

Each frame goes through a detector (YOLO11 nano by default, trained on COCO).
Model labels are mapped to eight core classes (`pedestrian`, `bicycle`,
`motorcycle`, `tricycle`, `car`, `bus`, `truck`, `cart`) **by name**, through
`traffic_vision/taxonomy.py`. Mapping by name rather than by class number
matters: in COCO, id 3 is "motorcycle" while in BDD100K id 3 is "bus". The same
code therefore works with COCO weights, BDD100K weights, VisDrone weights, a
model fine-tuned on local images, or an open-vocabulary model.

**Context profiles.** A file in `contexts/` lists the local vehicle types of a
city, the core class each one is counted in and the labels a model may give
it (for Dakar: car rapide, Ndiaga Ndiaye and Tata minibuses as `bus`, jakarta
as `motorcycle`, charrette as `cart`). `--context dakar` puts these labels
before the default mapping. A profile also records the data protection law
and the authority to ask before filming.

**Open-vocabulary models.** YOLOE prompt-free (`yoloe-11s-seg-pf.pt`) names
objects from a vocabulary of 4,585 classes; its road labels ("minibus", "tour
bus", "auto rickshaw", "horse cart", "suv", "trailer truck"...) are mapped to
the core classes. It is not a fix for local vehicles: on photos of Dakar it
called the car rapide a "trailer truck" (`docs/results/open_vocabulary_probe.md`).
Relabelling "trailer truck" as `bus` would break real trucks, so such
confusions are corrected statistically (section 7) or by fine-tuning.

People riding a motorbike or a bicycle are detected as "person" by COCO
models. Where two-wheelers are a large share of traffic, this would inflate
pedestrian counts, so a person box whose feet sit inside a two-wheeler box and
that overlaps it by at least 30% is treated as a rider and not counted
(`drop_riders`, can be turned off).

## 2. Tracking and the class of a track

ByteTrack (through Ultralytics) links detections from frame to frame and gives
each object an id. It also uses low-confidence boxes to keep tracks alive
through short occlusions, which are common in dense traffic.

The class given by the detector can change from frame to frame: a police van
half hidden by a pole was read as car, truck and bus on the demo clip. Every
counted event therefore uses the **class of the track**: the class with the
highest confidence summed over the frames seen so far. `detections.csv` keeps
both the track class (`road_class`) and the raw class of the frame
(`frame_class`).

## 3. Counting on a line

An object is counted when the bottom-centre of its box (its contact point with
the road) crosses a counting line drawn across the lanes. The side of the line
is given by the sign of a cross product; the direction of the crossing gives
the direction of travel. Counting crossings, not detections per frame, avoids
counting a slow vehicle many times.

A **dead band** of 1% of the frame height on each side of the line
(`line_margin`) absorbs the jitter of boxes of vehicles waiting on the line. A
side change only counts once the contact point is clearly past the line. On
the demo clip this removed five spurious crossings by four cars waiting at a
red light.

Lines are given as fractions of the frame, so the same scene file works at any
resolution. Extra named lines (`--line-extra NAME X,Y X,Y`, one per arm of a
junction) give **movements**: a track counted first on line A and last on line
B made the movement `A>B` (`analysis.movements`), as in the vehicle counting
track of the AI City Challenge. A banned turn is checked this way. Place them where traffic flows, away from stop lines, queues and
street furniture that hides vehicles.

## 4. Counting in a zone

A zone is a polygon drawn on the image, typically a reserved lane. An object
uses the zone when its contact point stays inside for at least two processed
frames (`zone_min_frames`); each object is counted once per zone. Events go
to `zone_events.csv`.

Zones are more sensitive than lines to identity switches: a bus lost behind a
pole and found again gets a new id and is counted twice. Check zone counts
against a manual count before using them.

## 5. Indicators

From `crossings.csv`:

- **flow**: crossings per interval, by class and direction, with empty
  intervals kept as zeros;
- **passenger car equivalents (PCE)**: flow weighted by class (illustrative
  defaults: car 1, bus and truck 2.5, cart 1, tricycle 0.75, motorcycle 0.5,
  bicycle 0.3, pedestrian 0); replace them with the weights used by the
  traffic authority;
- **vehicle shares** and **exposure** of pedestrians and two-wheelers per hour.

## 6. Checking a rule

Rules are described in measure registries (`measures/*.yaml`, field reference
in `measures/README.md`); `traffic-vision measures --list` lists them and
`traffic-vision check <measure> <run> --registry <name>` applies the test that
fits the kind of rule:

- **reserved lane**: share of the allowed classes among vehicles using the
  zone, with a bootstrap interval that resamples time intervals (the Wilson
  interval when the recording has fewer than five intervals), and the rate of
  other vehicles per hour with a Poisson interval;
- **ban on a class during some hours**: the recording time is split into time
  inside and outside the ban window (clock time comes from `--start-clock`);
  the rate of the banned class is computed on each side with a Poisson
  interval (Garwood's exact interval, chi-square quantiles by the
  Wilson-Hilferty approximation);
  a ban that applies inside a zone (a road closed on Sunday mornings) uses
  the zone events instead of the line; an all-day ban gives a rate to compare
  with a recording from before the ban or from a control area;
- **banned movement**: vehicles per hour going from one named line to
  another, with a Poisson interval;
- **service frequency**: vehicles of the service per hour against the
  promised rate (below, compatible with or above the promise, judged by the
  Poisson interval);
- **closure**: measured on diversion routes with a difference-in-differences
  (section 7).

Video sees classes, not permits or weights. Results describe the use of the
road, not offences.

## 7. Estimating the effect of a measure

`traffic-vision did` compares four recordings: a treated site and a control
site, before and during the measure. The effect is

    (treated during - treated before) - (control during - control before)

on the mean flow per interval. The control site absorbs what changed
everywhere in the city at the same time. The design assumes that, without the
measure, both sites would have changed alike (parallel trends): choose a
control street of the same kind, outside the reach of the measure.

**Calibration.** A short manual count of the same video gives, per class, a
correction factor: manual total divided by automatic total. The manual count
file has one row per window and class:

    start_s,end_s,road_class,manual_count
    0,60,car,41
    0,60,motorcycle,17
    60,120,car,38

`traffic-vision calibrate manual.csv outputs/<run>` prints the error and the
factor per class; `traffic-vision did ... --calibration manual.csv` applies it.
This correction assumes the detector makes the same errors in all four
recordings, which fails when a camera sees worse than another or a crowd
hides vehicles during the measure.

**Prediction-powered estimate (recommended).** With `--labels`, each
recording gets its own correction. A person counts a few windows of each
recording, chosen at random, all of the same length:

    run,start_s,end_s,manual_count
    cheikh_anta_diop_before,600,900,63
    cheikh_anta_diop_before,4200,4500,58
    cheikh_anta_diop_closure,1500,1800,81

The network's counts are computed on consecutive windows of that length over
the whole recording, and the mean true count per window is estimated as

    mean(hand counts) + w * (mean(network, all windows) - mean(network, counted windows))

This is prediction-powered inference (Angelopoulos et al., 2023). The weight
`w` is by default the ratio of hand to network counts on the counted windows,
which fits a detector that misses or adds a share of vehicles (it is the ratio
estimator of survey sampling written in this form); `--ppi-weight 1` gives
plain PPI and `tuned` the weight of PPI++. The variance adds the batch-means
variance of the network's mean and the variance of the hand-count correction;
the degrees of freedom follow Welch-Satterthwaite. The four estimates are
combined into the difference-in-differences. Because the hand counts measure
the true class, the same estimate also removes the bias of a look-alike class
(car rapides seen as trucks under a truck ban).

**Interval.** Traffic is autocorrelated. The default interval splits each
recording into four batches of consecutive intervals, uses the variance of the
batch means, and a Student t value with Welch-Satterthwaite degrees of
freedom; the uncertainty of the calibration factor is added by the delta
method. A moving block bootstrap is available (`--block`). The simulation
study in `docs/results/simulation.md` compares these choices: with two-hour
recordings and autocorrelated traffic, batch means covered the true effect in
92% of simulated studies, the block bootstrap in 79%. When detection fell
from 82% to 70% during the measure, the singly calibrated estimate lost the
effect (0.8 for a true 12) while the prediction-powered estimate recovered it
(11.8, coverage 94%).

**How much to film.** `traffic-vision design --base 60 --effect 0.2` simulates
the study before anyone films and prints, for several numbers of hours per
day and days per period, the chance that the interval excludes zero.

`traffic-vision compare` remains available for a simple comparison of two
recordings; it does not separate the measure from city-wide changes.

## 8. Data schema

`detections.csv`, one row per tracked object per processed frame:

| column | meaning |
|---|---|
| scene_id | scene identifier from the run |
| video | video file name |
| frame | frame index in the video |
| video_time_s | time in the video, in seconds |
| track_id | anonymous id given by the tracker |
| road_class | class of the track (highest summed confidence so far) |
| frame_class | class given by the detector in this frame |
| model_label | label given by the model before mapping |
| confidence | detection confidence |
| x1, y1, x2, y2 | box corners in pixels |
| anchor_x, anchor_y | bottom-centre of the box |
| speed_px_s | speed of the anchor since the previous processed frame, in pixels per second (empty on the first frame of a track); converting to km/h needs a calibrated view |

`crossings.csv`, one row per counted crossing: `scene_id, video, line,
track_id, road_class, direction, frame, video_time_s, clock_time`. `line` is
`line_1` for the main line and the name of each extra line.

`zone_events.csv`, one row per object using a zone: `scene_id, video, zone,
track_id, road_class, frame, video_time_s, clock_time`.

`clock_time` is filled when the run is given the clock time of the first frame
(`--start-clock 2026-11-07T07:00:00`).

`summary.json`: run settings (including the context profile), video
properties, frames without any selected object, unique tracks seen per class,
line totals, extra line totals and zone totals.

Several scenes share the same schema, so their files can be concatenated into
one dataset (`traffic-vision aggregate`).

## 9. Validating counts

Before using counts in a study, count about ten windows of each video by
hand, chosen at random, in the format of section 7, and report the error per
class with the results. The same file corrects the counts. Windows chosen by
convenience (the quiet minutes, the clear frames) can bias the correction.
