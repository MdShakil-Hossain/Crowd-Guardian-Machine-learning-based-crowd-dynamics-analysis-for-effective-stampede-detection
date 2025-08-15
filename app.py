# app.py — Crowd Guardian (Video + Image) — session_state + stable H.264 preview

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

# ---------- URL / path helpers for model loading ----------
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import urllib.request, hashlib

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "models"
CACHE_DIR.mkdir(exist_ok=True)

def _normalize_dropbox(url: str) -> str:
    """Ensure Dropbox URL is direct-download."""
    if "dropbox.com" not in url:
        return url
    parts = urlparse(url)
    q = parse_qs(parts.query)
    q["dl"] = ["1"]
    new_q = urlencode({k: v[0] for k, v in q.items()})
    return urlunparse(parts._replace(query=new_q))

def _download_to_cache(url: str) -> str:
    url = _normalize_dropbox(url)
    name = "model_" + hashlib.sha1(url.encode()).hexdigest() + ".h5"
    dst = CACHE_DIR / name
    if not dst.exists():
        with st.spinner("Downloading model…"):
            with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
    return str(dst)

def _resolve_model_path(path_or_url: str) -> str:
    p = (path_or_url or "").strip()
    if p.startswith("http://") or p.startswith("https://"):
        return _download_to_cache(p)
    # local: try relative to app dir, then absolute
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = APP_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Model not found at: {candidate}")
    return str(candidate.resolve())

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
    params.filterByArea = True; params.minArea = minArea; params.maxArea = maxArea
    params.filterByCircularity = True; params.minCircularity = min_circ
    params.filterByInertia = True; params.minInertiaRatio = min_inertia
    params.filterByConvexity = False; params.filterByColor = False

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
            matches.append((i, j)); un_prev.discard(i); un_curr.discard(j)
    return matches, sorted(list(un_prev)), sorted(list(un_curr))

def preprocess_for_cnn(gray):
    # grayscale -> 100x100 -> [0,1] -> (1,100,100,1)
    resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    return x

def combine_labels(y_heads, y_cnn, rule="and"):
    rule = (rule or "and").lower()
    if rule == "and": return 1 if (y_heads == 1 and y_cnn == 1) else 0
    if rule == "or": return 1 if (y_heads == 1 or y_cnn == 1) else 0
    if rule == "cnn_only": return int(y_cnn)
    if rule == "heads_only": return int(y_heads)
    return int(y_heads)

def _get_ffmpeg_exe():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None

def transcode_to_h264(src_path: str, dst_path: str, fps: float):
    """
    Convert src video to H.264 MP4 (yuv420p + faststart) for browser playback.
    Returns (final_path, ok, log). Never deletes src.
    """
    ff = _get_ffmpeg_exe()
    if ff is None:
        return src_path, False, "ffmpeg not found on PATH and imageio-ffmpeg not available"

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

    # You can paste a local path (repo file) or a URL (e.g., Dropbox)
    model_path = st.text_input(
        "Model (.h5) path or URL",
        value="https://www.dropbox.com/scl/fi/zswpw1ucbj7bkkkykc8oc/deep_cnn_stampede.h5"
              "?rlkey=e863b9skyvpyn0dn4gbwxd71s&st=uvgqjq7q&dl=1"
    )

    # Optional: upload a .h5 directly at runtime
    up = st.file_uploader("…or upload a .h5 file", type=["h5"], key="h5_upl")
    if up is not None:
        up_path = CACHE_DIR / "uploaded_model.h5"
        up_path.write_bytes(up.read())
        model_path = str(up_path)

    cnn_threshold = st.slider("CNN threshold (τ)", 0.0, 1.0, 0.50, 0.01)

    # Video-only
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
# Model load (cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_cnn(path_or_url: str):
    import tensorflow as tf
    resolved = _resolve_model_path(path_or_url)
    return tf.keras.models.load_model(resolved, compile=False)

cnn_model, load_err = None, None
if model_path:
    try:
        cnn_model = load_cnn(model_path)
    except Exception as e:
        load_err = str(e)

# =========================
# Video analysis
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

    # CSV outputs
    frames_rows = [("frame_index","time_sec","timecode","head_count","delta_vs_prev",
                    "prob_cnn","cnn_label","heads_label","final_label")]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    # Persistent files under ./outputs
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

    prev_pts, prev_count = [], None
    in_event, start_f, start_t = False, None, None
    min_event_frames = max(1, int(round(min_event_sec * (fps/step))))

    prog = st.progress(0.0); status = st.empty()
    processed = 0; total_steps = (N // step + 1) if N > 0 else 0

    f = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break

        if f % step == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            head_pts = detect_heads_gray(gray, detector)
            curr_count = len(head_pts)

            matches, _, _ = assign_matches(prev_pts, head_pts, max_match_dist)

            if prev_count is None:
                delta, y_heads = 0, 0
            else:
                delta = prev_count - curr_count
                r = (prev_count - curr_count) / max(1, prev_count)
                y_heads = 1 if (delta >= abs_drop or r >= rel_drop) else 0

            x = preprocess_for_cnn(gray)
            p_cnn = float(model.predict(x, verbose=0)[0][0])
            y_cnn = 1 if p_cnn >= cnn_threshold else 0

            final_label = combine_labels(y_heads, y_cnn, combine_rule)

            t = f / fps; tc = sec_to_tc(t)
            frames_rows.append((f, f"{t:.3f}", tc, curr_count, delta,
                                f"{p_cnn:.6f}", y_cnn, y_heads, final_label))

            if final_label == 1 and not in_event:
                in_event, start_f, start_t = True, f, t
            elif final_label == 0 and in_event:
                dur_frames = (f - start_f) // step
                if dur_frames >= min_event_frames:
                    end_t = (f-1) / fps
                    events_rows.append((start_f, f-1, f"{start_t:.3f}", f"{end_t:.3f}",
                                        sec_to_tc(start_t), sec_to_tc(end_t), f"{(end_t-start_t):.3f}"))
                in_event, start_f, start_t = False, None, None

            # Overlay
            vis = frame_bgr.copy()
            for (cx, cy) in head_pts:
                cv2.circle(vis, (int(cx), int(cy)), 4, (255,255,0), -1)
            if draw_links:
                for (i_prev, j_curr) in matches:
                    x1, y1 = prev_pts[i_prev]; x2, y2 = head_pts[j_curr]
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            banner_h = max(40, H//14)
            color = (0,0,255) if final_label==1 else (0,180,0)
            cv2.rectangle(vis, (0,0), (W, banner_h), color, -1)
            txt = (f"heads={curr_count}  Δ={delta:+d}  "
                   f"p_cnn={p_cnn:.2f} (τ={cnn_threshold:.2f})  "
                   f"rule={combine_rule}  label={'Stampede' if final_label==1 else 'No Stampede'}  t={tc}")
            cv2.putText(vis, txt, (12, banner_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

            if out.isOpened():
                out.write(vis)

            prev_pts, prev_count = head_pts, curr_count
            processed += 1
            if total_steps:
                prog.progress(min(1.0, processed/total_steps))
                status.write(f"Processed {processed}/{total_steps} sampled frames…")
        f += 1

    if in_event and start_f is not None:
        end_t = (f-1) / fps
        events_rows.append((start_f, f-1, f"{start_t:.3f}", f"{end_t:.3f}",
                            sec_to_tc(start_t), sec_to_tc(end_t), f"{(end_t-start_t):.3f}"))

    cap.release()
    if out.isOpened(): out.release()

    # Transcode to browser-playable H.264 MP4
    playable_path, ok, log = transcode_to_h264(raw_path, mp4_path, fps)
    if not ok:
        st.warning("Transcode failed; preview may not play. Details: " +
                   (log[:250] + ("..." if len(log) > 250 else "")))

    prog.progress(1.0); status.write("Done.")

    df_frames = pd.DataFrame(frames_rows[1:], columns=frames_rows[0])
    df_events = pd.DataFrame(events_rows[1:], columns=events_rows[0])
    return df_frames, df_events, playable_path

def analyze_image_cnn_only(image_bgr, model, cnn_threshold=0.5):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x = preprocess_for_cnn(gray)
    p = float(model.predict(x, verbose=0)[0][0])
    y = 1 if p >= cnn_threshold else 0
    return p, y

def overlay_heads_on_image(image_bgr, min_frac, max_frac, min_circ, min_iner):
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = build_blob_detector(W, H, min_frac, max_frac, min_circ, min_iner)
    head_pts = detect_heads_gray(gray, detector)
    vis = image_bgr.copy()
    for (cx, cy) in head_pts:
        cv2.circle(vis, (int(cx), int(cy)), 4, (255,255,0), -1)
    return vis, len(head_pts)

# =========================
# Results renderer (persists across reruns)
# =========================
def render_results(res):
    df_frames = res["df_frames"]
    df_events = res["df_events"]
    labeled_path = res["labeled_path"]

    st.subheader("Results — Video")
    if df_events.empty:
        st.success("No stampede detected.")
    else:
        total = df_events["duration_sec"].astype(float).sum()
        longest = df_events["duration_sec"].astype(float).max()
        st.error(f"Stampede DETECTED — events={len(df_events)}, total={total:.2f}s, longest={longest:.2f}s")

    with st.expander("Detected intervals (events)", expanded=not df_events.empty):
        st.dataframe(df_events, use_container_width=True)
    with st.expander("Per-frame predictions"):
        st.dataframe(df_frames.head(1000), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download events.csv",
            data=df_events.to_csv(index=False).encode("utf-8"),
            file_name="events.csv", mime="text/csv", key="dl_events")
    with c2:
        st.download_button("⬇️ Download frame_preds.csv",
            data=df_frames.to_csv(index=False).encode("utf-8"),
            file_name="frame_preds.csv", mime="text/csv", key="dl_frames")
    with c3:
        if os.path.exists(labeled_path) and os.path.getsize(labeled_path) > 0:
            with open(labeled_path, "rb") as fh:
                vid_bytes = fh.read()
            st.download_button("⬇️ Download labeled.mp4",
                data=vid_bytes, file_name=os.path.basename(labeled_path),
                mime="video/mp4", key="dl_video")
        else:
            st.warning("Labeled video not found or empty.")

    st.subheader("Labeled Video Preview")
    if os.path.exists(labeled_path):
        st.video(labeled_path)
    else:
        st.info("Preview unavailable.")

    if st.button("Clear results", key="clear_results"):
        st.session_state.pop("video_results", None)
        st.rerun()

# =========================
# UI actions
# =========================
uploader_key = "vid_upl" if input_mode == "Video" else "img_upl"
uploaded = st.file_uploader(
    "Upload a crowd video" if input_mode == "Video" else "Upload a crowd image",
    type=(["mp4","mov","mkv","avi"] if input_mode == "Video" else ["jpg","jpeg","png","bmp","tif","tiff"]),
    key=uploader_key
)
go = st.button("Analyze", key="analyze_btn")

# Re-render previous results so downloads don't "reset" the page
if input_mode == "Video" and "video_results" in st.session_state:
    render_results(st.session_state["video_results"])

if go:
    if load_err:
        st.error(f"Failed to load model: {load_err}")
    elif not cnn_model:
        st.error("Provide a valid model path or URL to your .h5.")
    elif not uploaded:
        st.warning("Please upload a file.")
    else:
        if input_mode == "Video":
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp_in.write(uploaded.read()); tmp_in.close()

            with st.spinner("Analyzing video…"):
                df_frames, df_events, labeled_path = analyze_video(
                    tmp_in.name, cnn_model,
                    target_fps=target_fps if target_fps and target_fps > 0 else None,
                    cnn_threshold=cnn_threshold,
                    abs_drop=abs_drop, rel_drop=rel_drop, min_event_sec=min_event_sec,
                    combine_rule=combine_rule, min_frac=min_frac, max_frac=max_frac,
                    min_circ=min_circ, min_iner=min_iner, draw_links=draw_links
                )

            st.session_state["video_results"] = {
                "df_frames": df_frames,
                "df_events": df_events,
                "labeled_path": labeled_path,
            }
            render_results(st.session_state["video_results"])

        else:  # Image
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.error("Could not read the uploaded image.")
            else:
                with st.spinner("Analyzing image…"):
                    prob, label = analyze_image_cnn_only(image_bgr, cnn_model, cnn_threshold=cnn_threshold)
                lbl_text = "Stampede" if label == 1 else "No Stampede"
                st.subheader("Results — Image")
                st.write(f"**Prediction:** {lbl_text}  |  **p_cnn:** {prob:.3f}")

                if draw_blobs_on_image:
                    vis_bgr, head_count = overlay_heads_on_image(image_bgr, min_frac, max_frac, min_circ, min_iner)
                    banner_h = max(40, image_bgr.shape[0]//14)
                    color = (0,0,255) if label==1 else (0,180,0)
                    cv2.rectangle(vis_bgr, (0,0), (image_bgr.shape[1], banner_h), color, -1)
                    txt = f"heads~{head_count}  p_cnn={prob:.2f} (τ={cnn_threshold:.2f})  label={lbl_text}"
                    cv2.putText(vis_bgr, txt, (12, banner_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
                    st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), caption=lbl_text, use_column_width=True)
                    ok, buf = cv2.imencode(".png", vis_bgr)
                    if ok:
                        st.download_button("⬇️ Download labeled.png", data=buf.tobytes(),
                                           file_name="labeled.png", mime="image/png", key="dl_image")
                else:
                    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=lbl_text, use_column_width=True)
