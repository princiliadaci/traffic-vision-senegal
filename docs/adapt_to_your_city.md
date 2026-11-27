# Apply the method to your city or your policy

The Dakar 2026 Youth Olympic Games are the worked example of this project,
not its limit. Any rule that changes who may use a road, when and how can be
studied the same way. This page goes from a policy to an estimate in seven
steps. Nothing below needs a change to the code.

## 1. Describe the rules: a measure registry

Copy [`measures/TEMPLATE.yaml`](../measures/TEMPLATE.yaml) to
`measures/<city>_<policy>.yaml` and write one entry per rule. Pick the kind
that matches what the rule does to traffic:

| The rule... | Kind | What the video gives |
|---|---|---|
| reserves a lane or a road for some vehicles (bus lane, BRT corridor, Olympic lane) | `reserved_lane` | share of allowed vehicles among lane users; other vehicles per hour |
| bans a class, everywhere or at some hours (trucks by day, motorbikes at night, okadas) | `class_ban` | rate of the banned class during and outside the ban, on a line or in a zone |
| bans a turn or a crossing | `movement_ban` | vehicles per hour going from one named line to another |
| promises a frequency (a bus every 6 minutes) | `service_frequency` | observed vehicles per hour against the promise |
| closes a road (an event, a car-free day) | `closure`, or `class_ban` with a zone | effect on diversion routes against a control street |

Then fill in, for each rule, what the camera **cannot** see (permits, weight,
commercial or private use) and an `observability` of high, partial or low.
This step is research in itself: it tells the authority, before any filming,
which rules video can evaluate. Every entry needs at least one source.

Check the file: `traffic-vision measures --registry measures/<your_file>.yaml`.

## 2. Describe the vehicles: a context profile

Copy [`contexts/generic.yaml`](../contexts/generic.yaml) to
`contexts/<city>.yaml`. List the local vehicle types that matter for your
rules, the core class each one is counted in (`pedestrian`, `bicycle`,
`motorcycle`, `tricycle`, `car`, `bus`, `truck`, `cart`), and the labels a
model may give it. Add the data protection law and the authority to ask before
filming, and, if your traffic authority uses them, passenger car equivalents.

Point the registry to it with `context: <city>`.

## 3. Check what the detector sees

Before filming hours of video, take a few photos or frames of your city,
draw boxes around the vehicles your rules target, and run the probe:

```bash
python scripts/probe_open_vocabulary.py --annotations my_boxes.yaml --context <city>
```

The annotation format is in
[`docs/results/probe_annotations.yaml`](results/probe_annotations.yaml). Look
for **confusions between classes that your rule treats differently**: in Dakar,
car rapides seen as trucks matter under a truck ban, not under a bus lane
rule. Such a confusion means you need hand counts in every recording (step
6), or a model fine-tuned on local images
([`scripts/train_yolo.py`](../scripts/train_yolo.py)).

## 4. Plan the study

Choose a treated site (where the rule applies) and a control site (similar,
out of its reach), filmed before and during the rule. Then ask how long to
film:

```bash
traffic-vision design --base <vehicles per 5 min> --effect <expected change, e.g. 0.2>
```

Aim for a power of at least 80%. On a street with 60 vehicles per 5 minutes, a
+20% effect needs about three days of eight hours per period.

## 5. Film or collect video, and run it

```bash
traffic-vision run --video data/site_before.mp4 --scene site_before --context <city> \
    --line 0.05,0.7 0.95,0.7 --zone bus_lane 0.1,0.45 0.5,0.45 0.5,0.95 0.05,0.95 \
    --start-clock 2026-11-04T07:00:00
```

Use `--line-extra NAME X,Y X,Y` for the arms of a junction (movements),
`--model yoloe-11s-seg-pf.pt` for the open-vocabulary model, and a scene file
([`configs/scenes.example.yaml`](../configs/scenes.example.yaml)) when there
are many recordings.

## 6. Count a few windows by hand

In each recording, pick about ten windows **at random** (for example five
minutes each) and count the vehicles by hand. Write them in one file:

```
run,start_s,end_s,manual_count
site_before,600,900,63
site_before,4200,4500,58
```

This is what makes the estimate honest: the network's errors, whatever they
are, are measured in each recording and removed.

## 7. Check the rule and estimate the effect

```bash
traffic-vision check <measure_id> outputs/site_during --registry measures/<your_file>.yaml
traffic-vision did --treated-before outputs/site_before --treated-during outputs/site_during \
    --control-before outputs/control_before --control-during outputs/control_during \
    --labels hand_counts.csv
```

Report the effect with its interval, the hand-count errors by class, and what
the camera could not see.

## Before you film

- Filming public space is processing personal data in most countries. Ask
  the data protection authority named in your context profile, keep only
  classes and anonymous tracks, and delete raw video once it is processed.
- Prefer existing cameras (traffic control centres, bus operators) and
  short extracts.
- Agree on the questions with the authority that made the rule: the registry
  is a good basis for that conversation.
