# app.py — Crowd Guardian (Video + Image) — stable outputs + H.264 preview
# v2: writes to ./outputs/, transcodes to MP4 (H.264), never deletes the only copy

import os
import cv2
import time
import tempfile
import subprocess
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import streamlit as st

# =========================
# Streamlit UI config
# =========================
st.set_page_config(page_title="Crowd Guardian – Stampede Detection", layout="wide")
st.title("🎥 Crowd Guardian — Stampede Detection (Grayscale CNN + Head Count Drop)")

# =========================
# Helpers
# =========================
def sec_to_tc(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def build_blob_detector(frame_w, frame_h, min_frac=0.00005, max_frac=0.0020,
                        min_circ=0.2, min_inertia=0.1, thresh_step=10):
    frame_area = float(frame_w * frame_h)
    minArea = max(5.0, min_frac * frame_area)
    maxArea = max(minArea + 1.0, max_frac * frame_area)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = thresh_step

    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea

    params.filterByCircularity = True
    params.minCircularity = min_circ

    params.filterByInertia = True
    params.minInertiaRatio = min_inertia

    params.filterByConvexity = False
    params.filterByColor = False

    ver = cv2.__version__.split('.')[0]
    return (cv2.SimpleBlobDetector(params) if int(ver) < 3
            else cv2.SimpleBlobDetector_create(params))

def detect_heads_gray(gray, detector):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    kps = detector.detect(blur)
    return [(float(k.pt[0]), float(k.pt[1])) for k in kps]

def assign_matches(prev_pts, curr_pts, max_dist):
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return [], list(range(len(prev_pts))), list(range(len(curr_pts)))
    cost = np.zeros((len(prev_pts), len(curr_pts)), dtype=np.float32)
    for i, p in enumerate(prev_pts):
        for j, c in enumerate(curr_pts):
            cost[i, j] = np.linalg.norm(np.array(p) - np.array(c))
    r_idx, c_idx = linear_sum_assignment(cost)
    matches, un_prev, un_curr = [], set(range(len(prev_pts))), set(range(len(curr_pts)))
    for i, j in zip(r_idx, c_idx):
        if cost[i, j] <= max_dist:
            matches.append((i, j))
            un_prev.discard(i)
            un_curr.discard(j)
    return matches, sorted(list(un_prev)), sorted(list(un_curr))

def preprocess_for_cnn(gray):
    resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))  # (1,100,100,1)
    return x

def combine_labels(y_heads, y_cnn, rule="and"):
    rule = (rule or "and").lower()
    if rule == "and": return 1 if (y_heads == 1 and y_cnn == 1) else 0
    if rule == "or": return 1 if (y_heads == 1 or y_cnn == 1) else 0
    if rule == "cnn_only": return int(y_cnn)
    if rule == "heads_only": return int(y_heads)
    return int(y_heads)

def transcode_to_h264(src_path: str, dst_path: str, fps: float):
    """
    Convert src video to H.264 MP4 (yuv420p + faststart) for browser playback.
    Returns (final_path, ok, log). Never deletes src here.
    """
    ff = shutil.which("ffmpeg")
    if ff is None:
        return src_path, False, "ffmpeg not found on PATH"

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-i", src_path,
        "-r", str(int(round(fps if fps and fps > 0 else 25))),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = (proc.returncode == 0) and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    log = (proc.stderr or proc.stdout or "")
    return (dst_path if ok else src_path), ok, log

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.selectbox("Input Type", ["Video", "Image"], index=0)
    model_path = st.text_input("Model (.h5) path", value="deep_cnn_stampede.h5")
    cnn_threshold = st.slider("CNN threshold (τ)", 0.0, 1.0, 0.50, 0.01)

    combine_rule = st.selectbox("Combine rule (video)", ["and", "or", "cnn_only", "heads_only"], index=0)
    st.markdown("**Head-drop rule (video)**")
    abs_drop = st.number_input("Absolute drop ≥", min_value=0, value=2, step=1)
    rel_drop = st.slider("Relative drop ≥", 0.0, 1.0, 0.20, 0.01)
    min_event_sec = st.number_input("Min event duration (s)", min_value=0.0, value=1.0, step=0.5)

    st.markdown("**Sampling (video)**")
    target_fps = st.number_input("Target FPS (0 = all frames)", min_value=0.0, value=0.0, step=1.0)

    st.markdown("**Blob detector (visual only)**")
    min_frac = st.number_input("Min frac area", value=0.00005, format="%.6f")
    max_frac = st.number_input("Max frac area", value=0.0020, format="%.6f")
    min_circ = st.slider("Min circularity", 0.0, 1.0, 0.20, 0.05)
    min_iner = st.slider("Min inertia", 0.0, 1.0, 0.10, 0.05)
    draw_links = st.checkbox("Draw matches between frames (video)", value=True)
    draw_blobs_on_image = st.checkbox("Draw head blobs on image (visual only)", value=True)

# =========================
# Model load
# =========================
@st.cache_resource(show_spinner=False)
def load_cnn(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

cnn_model, load_err = None, None
if model_path:
    try:
        cnn_model = load_cnn(model_path)
    except Exception as e:
        load_err = str(e)

# =========================
# Analysis functions
# =========================
def analyze_video(
    video_path,
    model,
    target_fps=None,
    cnn_threshold=0.5,
    abs_drop=2,
    rel_drop=0.20,
    min_event_sec=1.0,
    combine_rule="and",
    min_frac=0.00005,
    max_frac=0.0020,
    min_circ=0.2,
    min_iner=0.1,
    draw_links=True
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = 1 if not target_fps or target_fps <= 0 else max(1, int(round(fps / float(target_fps))))
    max_match_dist = max(15, int(0.03 * max(W, H)))

    detector = build_blob_detector(W, H, min_frac, max_frac, min_circ, min_iner)

    # outputs
    frames_rows = [(
        "frame_index","time_sec","timecode",
        "head_count","delta_vs_prev",
        "prob_cnn","cnn_label","heads_label","final_label"
    )]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    # Persistent output files in ./outputs
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = os.path.join(out_dir, f"{base}_{stamp}_raw.avi")
    mp4_path = os.path.join(out_dir, f"{base}_{stamp}_labeled.mp4")

    # Write robust RAW first (MJPG AVI)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(raw_path, fourcc, fps if fps and fps > 0 else 25.0, (W, H))
    if not out.isOpened():
        st.warning("VideoWriter failed to open. Preview may not render; downloads may still work.")

    prev_pts = []
    prev_count = None
    in_event = False
    start_f = None
    start_t = None
    min_event_frames = max(1, int(round(min_event_sec * (fps/step))))

    prog = st.progress(0.0)
    status = st.empty()
    processed = 0
    total_steps = (N // step + 1) if N > 0 else 0

    f = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if f % step == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # 1) Head-like blobs
            head_pts = detect_heads_gray(gray, detector)
            curr_count = len(head_pts)

            # 2) Match heads to previous
            matches, _, _ = assign_matches(prev_pts, head_pts, max_match_dist)

            # 3) Head-drop rule
            if prev_count is None:
                delta = 0
                y_heads = 0
            else:
                delta = prev_count - curr_count
                r = (prev_count - curr_count) / max(1, prev_count)
                y_heads = 1 if (delta >= abs_drop or r >= rel_drop) else 0

            # 
