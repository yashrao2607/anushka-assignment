"""Streamlit upload-and-analyse UI (PRD US-5.4).

    streamlit run app.py

Upload a video, pick a tracker, and get the annotated output plus the
per-frame CSV and the unique-object analytics — the same pipeline the CLI
runs, so the browser and the terminal cannot disagree about a number.

Everything heavy is imported lazily inside the run handler. Streamlit re-executes
this file top to bottom on every widget interaction, and importing torch at
module scope would make every checkbox click wait on it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="cv-detection-reid", page_icon="🎥", layout="wide")


@st.cache_resource
def get_config():
    from src.config import load_config

    return load_config()


cfg = get_config()

st.title("Object Detection, Tracking & Re-Identification")
st.caption(
    "Detection answers *what is in this frame*. Tracking answers *is this the same thing "
    "as before*. **Re-identification** answers *is this the same thing as ten seconds ago, "
    "or on the other camera* — which is the question that turns pixels into countable events."
)

with st.sidebar:
    st.header("Pipeline")
    tracker = st.selectbox(
        "Tracker", ["botsort", "bytetrack", "iou"],
        help="iou is the deliberately naive baseline; botsort adds camera-motion "
             "compensation and appearance.",
    )
    use_reid = st.checkbox("Appearance (ReID) association", value=True,
                           disabled=tracker != "botsort")
    use_gallery = st.checkbox("ReID gallery — long-occlusion recovery", value=True,
                              disabled=not (use_reid and tracker == "botsort"))
    blur = st.checkbox("Privacy blur (faces + plates)", value=True,
                       help="PRD §17: enabled by default for shared output.")
    max_frames = st.slider("Max frames", 30, 900, 300, step=30)

    st.divider()
    st.header("Environment")
    from src.utils.device import environment_report

    env = environment_report()
    st.write(f"**Device:** `{env['resolved_device']}`")
    st.write(f"**Torch:** {env['torch']} · **OpenCV:** {env['opencv']}")
    if not env["cuda_available"]:
        st.warning(
            "No GPU detected. FPS below is the **CPU worst case** (PRD risk R9), "
            "not the ≥ 30 FPS GPU target.",
            icon="⚠️",
        )

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
sample_dir = cfg.path("raw_videos_dir")
samples = sorted(p.name for p in sample_dir.glob("*.mp4")) if sample_dir.exists() else []

source_path: Path | None = None
if uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    tmp.write_bytes(uploaded.read())
    source_path = tmp
elif samples:
    chosen = st.selectbox("…or pick a bundled scene", ["—"] + samples)
    if chosen != "—":
        source_path = sample_dir / chosen

if source_path and st.button("Run", type="primary"):
    from src.pipeline.demo import run_demo

    with st.spinner(f"Processing {source_path.name} on {env['resolved_device']}…"):
        result = run_demo(
            cfg, str(source_path), tracker_type=tracker, with_reid=use_reid,
            with_gallery=use_gallery, save=True, show=False,
            max_frames=max_frames, blur_faces=blur,
        )

    cols = st.columns(4)
    cols[0].metric("Unique objects", result["unique_objects"])
    cols[1].metric("FPS (CPU)", result["wall_fps"])
    cols[2].metric("p95 latency", f"{result['latency_p95_ms']} ms")
    cols[3].metric("Frames", result["frames"])

    left, right = st.columns([3, 2])
    with left:
        video_path = cfg.root / result.get("annotated_video", "")
        if video_path.exists():
            st.video(str(video_path))
            st.download_button("Download annotated video", video_path.read_bytes(),
                               file_name=video_path.name, mime="video/mp4")
    with right:
        st.subheader("Unique objects by class")
        st.json(result["unique_by_class"] or {"(none detected)": 0})
        st.subheader("Mean dwell time (s)")
        st.json(result["mean_dwell_s_by_class"] or {"(none)": 0})
        csv_path = cfg.root / result.get("results_csv", "")
        if csv_path.exists():
            st.download_button("Download per-frame CSV", csv_path.read_bytes(),
                               file_name=csv_path.name, mime="text/csv")

    with st.expander("Full run record"):
        st.json(result)

st.divider()
reports = cfg.path("reports_dir")
available = sorted(p.name for p in reports.glob("*.md")) if reports.exists() else []
if available:
    st.subheader("Generated reports")
    picked = st.selectbox("Report", available)
    st.markdown((reports / picked).read_text(encoding="utf-8"))
else:
    st.info("No reports yet — run `python -m src.cli eval-det` to generate them.")
