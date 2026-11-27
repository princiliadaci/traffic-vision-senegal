# Measure registries

One file per city and policy. Each file describes the rules, what a camera can
measure about each one, what it cannot see, and the sources.

| File | City | Policy type | Measures |
|---|---|---|---|
| [`dakar_joj_2026.yaml`](dakar_joj_2026.yaml) | Dakar, Diamniadio, Saly (Senegal) | event: Youth Olympic Games, 31 October to 13 November 2026 | 11 |
| [`dakar_brt_corridor.yaml`](dakar_brt_corridor.yaml) | Dakar (Senegal) | infrastructure: BRT corridor and its zone of influence | 4 |
| [`lagos_okada_ban.yaml`](lagos_okada_ban.yaml) | Lagos (Nigeria) | regulation: commercial motorcycles banned since 1 June 2022 | 1 |
| [`kigali_car_free_day.yaml`](kigali_car_free_day.yaml) | Kigali (Rwanda) | recurring: car-free Sunday mornings since 2016 | 1 |
| [`TEMPLATE.yaml`](TEMPLATE.yaml) | yours | | |

```bash
traffic-vision measures --list
traffic-vision measures --registry lagos_okada_ban
traffic-vision check <measure_id> <run_folder> --registry lagos_okada_ban
```

## Header

| Field | Meaning |
|---|---|
| `city`, `country` | where the rules apply |
| `context` | a profile in [`contexts/`](../contexts): local vehicles, law |
| `policy_type` | `event`, `infrastructure`, `regulation` or `recurring` |
| `summary` | one sentence on the plan or the policy |

## Fields of a measure

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | short name, used on the command line |
| `name` | yes | the rule in one line |
| `kind` | yes | `reserved_lane`, `class_ban`, `movement_ban`, `service_frequency` or `closure` |
| `place` | yes | road, section, side |
| `sources` | yes | at least one link: official text, press release or newspaper |
| `period` | no | `[first day, last day]` in ISO format; `"open"` for no end |
| `window` | no | daily hours, `"HH:MM-HH:MM"`, may cross midnight; none means all day |
| `recurrence` | no | text, e.g. "first and third Sunday of each month" |
| `allowed_classes` | reserved lane, service frequency | core classes allowed in the lane, or the vehicles of the service |
| `banned_classes` | class ban | core classes banned |
| `zone` | reserved lane; class ban inside an area | polygon name used when running the video |
| `lines` | movement ban | `[from_line, to_line]`, names of the lines given with `--line-extra` |
| `expected_per_hour` | service frequency | promised vehicles per hour |
| `indicator` | no | what the video gives, in words |
| `observability` | no | `high`, `partial` (default) or `low` |
| `video_cannot_see` | no | permits, weights, purpose of the trip |
| `notes` | no | context, uncertainties, things to check |

Core classes: `pedestrian`, `bicycle`, `motorcycle`, `tricycle`, `car`, `bus`,
`truck`, `cart`.

Facts in these files come from the sources listed with each measure. Rules
change: check the official texts in force before using a registry for a
study, and flag what you could not confirm in `notes`.
