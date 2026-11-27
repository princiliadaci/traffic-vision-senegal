from traffic_vision.counting import CountingLine


def line():
    # horizontal line y = 100 from x = 0 to x = 200; moving down = "in"
    return CountingLine((0, 100), (200, 100), directions=("in", "out"))


def test_counts_once_when_crossing_down():
    ln = line()
    assert ln.update(1, "car", (50, 80), 0, 0.0) is None
    ev = ln.update(1, "car", (52, 110), 1, 0.04)
    assert ev is not None and ev.direction == "in"
    assert ln.totals() == {"car": {"in": 1}}


def test_opposite_direction():
    ln = line()
    ln.update(7, "bus", (100, 150), 0, 0.0)
    ev = ln.update(7, "bus", (100, 90), 1, 0.04)
    assert ev.direction == "out"


def test_jitter_on_the_line_counts_once_per_direction():
    ln = line()
    for i, y in enumerate([95, 105, 95, 105, 95]):
        ln.update(3, "motorcycle", (60, y), i, i / 25)
    assert ln.totals() == {"motorcycle": {"in": 1, "out": 1}}


def test_crossing_outside_the_segment_is_ignored():
    ln = line()
    ln.update(4, "car", (300, 80), 0, 0.0)
    assert ln.update(4, "car", (300, 120), 1, 0.04) is None


def test_from_relative():
    ln = CountingLine.from_relative((0.1, 0.5), (0.9, 0.5), 1000, 400)
    assert ln.a == (100.0, 200.0) and ln.b == (900.0, 200.0)


def test_dead_band_ignores_a_car_waiting_on_the_line():
    ln = CountingLine((0, 100), (200, 100), directions=("in", "out"), margin=10)
    ln.update(5, "car", (60, 70), 0, 0.0)                    # clearly before the line
    for i, y in enumerate([96, 104, 97, 105, 99], start=1):  # waiting at a red light
        assert ln.update(5, "car", (60, y), i, i / 25) is None
    ev = ln.update(5, "car", (60, 130), 6, 0.3)               # drives on
    assert ev is not None and ev.direction == "in"
    assert ln.totals() == {"car": {"in": 1}}


def test_from_relative_margin():
    ln = CountingLine.from_relative((0.1, 0.5), (0.9, 0.5), 1000, 400, margin_frac=0.01)
    assert ln.margin == 4.0
