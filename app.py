# app.py — Crowd Guardian (Video) — silent model load + polished UI + logo

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

# ---------- URL / path helpers for model & logo ----------
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import urllib.request, hashlib, sys
from typing import Optional

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "models"
CACHE_DIR.mkdir(exist_ok=True)

# Your Dropbox model (direct download). You can override with a secret MODEL_URL.
DEFAULT_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/zswpw1ucbj7bkkkykc8oc/deep_cnn_stampede.h5"
    "?rlkey=e863b9skyvpyn0dn4gbwxd71s&st=uvgqjq7q&dl=1"
)
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

# ---- fixed defaults (no UI knobs) ----
CNN_THRESHOLD = 0.50
ABS_DROP      = 2
REL_DROP      = 0.20
MIN_EVENT_SEC = 1.0
COMBINE_RULE  = "and"
MIN_FRAC      = 0.00005
MAX_FRAC      = 0.0020
MIN_CIRC      = 0.20
MIN_INER      = 0.10
DRAW_LINKS    = True
TARGET_FPS    = None  # None = all frames

def _normalize_dropbox(url: str) -> str:
    if "dropbox.com" not in url:
        return url
    parts = urlparse(url)
    q = parse_qs(parts.query)
    q["dl"] = ["1"]  # force direct download
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
    if p.startswith(("http://", "https://")):
        return _download_to_cache(p)
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = APP_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Model not found at: {candidate}")
    return str(candidate.resolve())

# ---------- Logo ----------
# Looks for assets/Crowd_Guardian_EWU_logo.png (preferred), then ./Crowd_Guardian_EWU_logo.png,
# or you can set LOGO_URL in secrets to host it elsewhere.
DEFAULT_LOGO_PATHS = [
    APP_DIR / "assets" / "Crowd_Guardian_EWU_logo.png",
    APP_DIR / "Crowd_Guardian_EWU_logo.png",
]
LOGO_SRC = st.secrets.get("LOGO_URL", None)

def _load_logo_bytes(src: Optional[str]) -> Optional[bytes]:
    try:
        if src and src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src) as r:
                return r.read()
        if src:  # local explicit path
            with open(src, "rb") as f:
                return f.read()
        # try default paths
        for p in DEFAULT_LOGO_PATHS:
            if p.exists():
                with open(p, "rb") as f:
                    return f.read()
    except Exception:
        pass
    return None

_logo_bytes = _load_logo_bytes(LOGO_SRC)

# =========================
# Page config & custom style
# =========================
st.set_page_config(
    page_title="Crowd Guardian: Machine learning based crowd dynamics analysis for effective stampede detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Minimal, modern feel */
    .main .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1100px;}
    h1, h2, h3 {letter-spacing: .2px;}
    .hero {
        padding: 16px 20px;
        border-radius: 18px;
        background: radial-gradient(1200px 400px at 10% -10%, rgba(59,130,246,.18), transparent),
                    linear-gradient(180deg, rgba(30,41,59,.45), rgba(2,6,23,.35));
        border: 1px solid rgba(148,163,184,.15);
    }
    .hero .title {font-size: 1.35rem; margin: 0 0 .25rem 0;}
    .hero .subtitle {margin: 0; opacity: .85}
    .upload-card {
        border: 1px dashed rgba(148,163,184,.35); border-radius: 16px; padding: 14px;
        background: rgba(30,41,59,.25);
    }
    .badge-ok {display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;
        background: rgba(16,185,129,0.18); color:#a7f3d0; border:1px solid rgba(16,185,129,0.35);}
    .badge-err {display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;
        background: rgba(239,68,68,0.18); color:#fecaca; border:1px solid rgba(239,68,68,0.35);}
    /* Sidebar branding only (no inputs) */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f172a 0%, #111827 60%, #1f2937 100%);
        color: #e5e7eb;
    }
    .sidebar-card {border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04); border-radius: 14px; padding: 12px 14px; margin: 12px;}
    .small {font-size:12px; opacity:0.8;}
    .footer {opacity:.65; font-size:12px; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: branding + environment
with st.sidebar:
    if _logo_bytes:
        st.image(_logo_bytes, width=120)
    st.markdown('<div class="sidebar-card"><b>🛡️ Crowd Guardian</b><br/>'
                'Machine-learning aided detection of crowd stampedes.</div>', unsafe_allow_html=True)
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
    except Exception:
        tf_ver = "not loaded"
    st.markdown(
        f'<div class="sidebar-card small">Py {sys.version.split()[0]} • TF {tf_ver} '
        f'• NumPy {np.__version__} • OpenCV {cv2.__version__}</div>',
        unsafe_allow_html=True,
    )

# ======= Hero (with logo) =======
st.markdown('<div class="hero">', unsafe_allow_html=True)
col_logo, col_text = st.columns([1, 9])
with col_logo:
    if _logo_bytes:
        st.image(_logo_bytes, width=72)
with col_text:
    st.markdown('<div class="title">Crowd Guardian</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine learning based crowd dynamics analysis for effective stampede detection</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def sec_to_tc(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def build_blob_detector(frame_w, frame_h, min_frac=MIN_FRAC, max_frac=MAX_FRAC,
                        min_circ=MIN_CIRC, min_inertia=MIN_INER, thresh_step=10):
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
    resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    return x

def combine_labels(y_heads, y_cnn, rule=COMBINE_RULE):
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
# Model load (silent & cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_cnn(path_or_url: str):
    import tensorflow as tf
    resolved = _resolve_model_path(path_or_url)
    return tf.keras.models.load_model(resolved, compile=False)

cnn_model, load_err = None, None
try:
    cnn_model = load_cnn(MODEL_URL)
except Exception as e:
    load_err = str(e)

# Status chip below hero
if load_err:
    st.markdown(f'<span class="badge-err">Model error</span> — {load_err}', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-ok">Model loaded</span>', unsafe_allow_html=True)

# =========================
# Video analysis (fixed params)
# =========================
def analyze_video(
    video_path,
    model,
    target_fps=TARGET_FPS,
    cnn_threshold=CNN_THRESHOLD,
    abs_drop=ABS_DROP,
    rel_drop=REL_DROP,
    min_event_sec=MIN_EVENT_SEC,
    combine_rule=COMBINE_RULE,
    min_frac=MIN_FRAC,
    max_frac=MAX_FRAC,
    min_circ=MIN_CIRC,
    min_iner=MIN_INER,
    draw_links=DRAW_LINKS
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

    frames_rows = [("frame_index","time_sec","timecode","head_count","delta_vs_prev",
                    "prob_cnn","cnn_label","heads_label","final_label")]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = os.path.join(out_dir, f"{base}_{stamp}_raw.avi")
    mp4_path = os.path.join(out_dir, f"{base}_{stamp}_labeled.mp4")

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

    playable_path, ok, log = transcode_to_h264(raw_path, mp4_path, fps)
    if not ok:
        st.warning("Transcode failed; preview may not play. Details: " +
                   (log[:250] + ("..." if len(log) > 250 else "")))

    prog.progress(1.0); status.write("Done.")

    df_frames = pd.DataFrame(frames_rows[1:], columns=frames_rows[0])
    df_events = pd.DataFrame(events_rows[1:], columns=events_rows[0])
    return df_frames, df_events, playable_path

# =========================
# UI (video only)
# =========================
st.subheader("Upload a crowd video")
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop a video or browse",
    type=["mp4","mov","mkv","avi","mpeg4"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)
go = st.button("Analyze", type="primary")

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

    st.markdown('<div class="footer">© Crowd Guardian</div>', unsafe_allow_html=True)

if "video_results" in st.session_state:
    render_results(st.session_state["video_results"])

if go:
    if load_err:
        st.error(f"Failed to load model: {load_err}")
    elif not cnn_model:
        st.error("Model not loaded.")
    elif not uploaded:
        st.warning("Please upload a video.")
    else:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp_in.write(uploaded.read()); tmp_in.close()

        with st.spinner("Analyzing video…"):
            df_frames, df_events, labeled_path = analyze_video(tmp_in.name, cnn_model)

        st.session_state["video_results"] = {
            "df_frames": df_frames,
            "df_events": df_events,
            "labeled_path": labeled_path,
        }
        render_results(st.session_state["video_results"])
