"""Streamlit dashboard: upload a video, watch detection live, explore results, check the Games measures.

Run with:  streamlit run app/dashboard.py
Author: Princilia DACI
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_vision.analysis import compare_periods, flows, load_crossings, modal_shares  # noqa: E402
from traffic_vision.pipeline import RunConfig, iter_run  # noqa: E402
from traffic_vision.rules import ban_check, lane_use, load_registry  # noqa: E402
from traffic_vision.taxonomy import ROAD_CLASSES  # noqa: E402

OUTPUT_DIR = Path("outputs")
st.set_page_config(page_title="Traffic Vision Senegal", layout="wide")
st.title("Traffic Vision Senegal")
st.caption("Count road users from video and check what the Dakar 2026 mobility measures "
           "changed. Built by Princilia DACI.")
REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "joj_measures.yaml"

tab_run, tab_results, tab_compare, tab_measures = st.tabs(
    ["Run a video", "Results", "Compare two runs", "Games measures"])

with tab_run:
    left, right = st.columns([1, 2])
    with left:
        up = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        scene = st.text_input("Scene id", "my_scene")
        classes = st.multiselect("Classes to track", ROAD_CLASSES, default=list(ROAD_CLASSES))
        model = st.text_input("Model weights", "yolo11n.pt")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        y = st.slider("Counting line height (fraction of the frame)", 0.05, 0.95, 0.6, 0.01)
        x1, x2 = st.slider("Counting line extent", 0.0, 1.0, (0.05, 0.95), 0.01)
        d1 = st.text_input("Name of the downward direction", "towards_camera")
        d2 = st.text_input("Name of the upward direction", "away_from_camera")
        stride = st.number_input("Process one frame out of", 1, 10, 2)
        clock = st.text_input("Clock time of the first frame (optional)", "",
                              placeholder="2026-11-07T07:00:00")
        go = st.button("Start", type="primary", disabled=up is None or not classes)
    with right:
        frame_box = st.empty()
        status = st.empty()
        if go and up is not None:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getbuffer())
            cfg = RunConfig(video=str(tmp), scene_id=scene, model=model, classes=classes,
                            confidence=conf, line=((x1, y), (x2, y)), line_directions=(d1, d2),
                            stride=int(stride), output_dir=str(OUTPUT_DIR),
                            start_clock=clock.strip() or None)
            for fr in iter_run(cfg):
                frame_box.image(fr.image[:, :, ::-1], use_container_width=True)
                total = sum(sum(d.values()) for d in fr.totals.values())
                msg = (f"t = {fr.video_time_s:.1f} s · {fr.objects_in_view} objects in view · "
                       f"{total} crossings")
                if fr.objects_in_view == 0:
                    msg += " · no selected object in view"
                status.write(msg)
            st.success(f"Done. Results saved in {OUTPUT_DIR / scene}")

runs = sorted(p.parent for p in OUTPUT_DIR.glob("*/summary.json"))

with tab_results:
    if not runs:
        st.info("No finished run yet.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda p: p.name)
        summary = json.loads((run / "summary.json").read_text())
        cx = load_crossings([run])
        c1, c2, c3 = st.columns(3)
        dur = summary["video"]["duration_s"]
        veh = cx[cx.road_class != "pedestrian"]
        c1.metric("Video length", f"{dur:.0f} s")
        c2.metric("Vehicles crossing", len(veh))
        c3.metric("Vehicles per minute", f"{len(veh) / max(dur / 60, 1e-9):.1f}")
        interval = st.select_slider("Interval (s)", [5, 10, 15, 30, 60, 300], value=10)
        f = flows(cx, interval, dur, by=("road_class",))
        st.bar_chart(f.pivot_table(index="start_s", columns="road_class", values="count",
                                   fill_value=0))
        st.subheader("Crossings by class and direction")
        st.dataframe(cx.groupby(["road_class", "direction"]).size().unstack(fill_value=0))
        st.subheader("Share of vehicles")
        st.dataframe(modal_shares(cx).rename("share").map("{:.0%}".format))
        video = run / "annotated.mp4"
        if video.exists():
            st.caption("Annotated video (mp4v; some browsers need it converted to H.264 to play)")
            st.video(str(video))

with tab_compare:
    if len(runs) < 2:
        st.info("Run at least two videos of the same place (for example before and during a measure).")
    else:
        a = st.selectbox("Period A (reference)", runs, format_func=lambda p: p.name, key="a")
        b = st.selectbox("Period B", runs, index=1, format_func=lambda p: p.name, key="b")
        interval = st.select_slider("Interval (s) ", [10, 30, 60, 300], value=60)

        def per_interval(run: Path) -> pd.Series:
            dur = json.loads((run / "summary.json").read_text())["video"]["duration_s"]
            cx = load_crossings([run])
            cx = cx[cx.road_class != "pedestrian"]
            return flows(cx, interval, dur, by=()).set_index("interval")["count"]

        try:
            res = compare_periods(per_interval(a), per_interval(b), a.name, b.name)
            st.write(res.as_text(f"vehicles per {interval} s"))
        except ValueError as e:
            st.warning(f"{e} Use a shorter interval or longer videos.")

with tab_measures:
    reg = load_registry(REGISTRY)
    st.write("Measures of the Dakar 2026 Youth Olympic Games, with what a camera can measure "
             "and what it cannot see. Sources are listed in `configs/joj_measures.yaml`.")
    table = pd.DataFrame([{
        "measure": m.name, "kind": m.kind.replace("_", " "), "place": m.place,
        "dates": f"{m.period[0]} to {m.period[1]}" if m.period else "",
        "hours": str(m.window) if m.window else "all day",
        "observability": m.observability, "video cannot see": " ".join(m.video_cannot_see.split()),
    } for m in reg.values()])
    st.dataframe(table, use_container_width=True, hide_index=True)
    if runs:
        st.subheader("Check a measure against a run")
        mid = st.selectbox("Measure", list(reg), format_func=lambda k: reg[k].name)
        run = st.selectbox("Run ", runs, format_func=lambda p: p.name, key="check_run")
        m = reg[mid]
        summary = json.loads((run / "summary.json").read_text())
        dur = summary["video"]["duration_s"]
        try:
            if m.kind == "reserved_lane":
                zf = run / "zone_events.csv"
                zones = sorted(pd.read_csv(zf)["zone"].unique()) if zf.exists() else []
                if not zones:
                    st.warning("This run has no zone. Run the video with a zone on the lane.")
                else:
                    zone = st.selectbox("Zone", zones)
                    st.write(lane_use(pd.read_csv(zf), zone, m.allowed_classes, dur, 10).as_text())
            elif m.kind == "class_ban" and m.window is not None:
                start = summary["config"].get("start_clock")
                st.write(ban_check(load_crossings([run]), m.banned_classes, m.window, dur,
                                   start, m).as_text())
            else:
                st.info("Compare a recording during the measure with one after it, or use "
                        "`traffic-vision did` with a control site.")
        except ValueError as e:
            st.warning(str(e))
