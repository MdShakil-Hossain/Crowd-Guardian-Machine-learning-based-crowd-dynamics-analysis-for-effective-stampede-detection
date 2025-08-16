# app.py — Crowd Guardian (Video only)
# Professional UI: custom HTML/CSS, decorated sidebar, timeline, Altair charts, JS confetti

import os
import cv2
import time
import tempfile
import subprocess
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.optimize import linear_sum_assignment

from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import urllib.request, hashlib
import streamlit.components.v1 as components

# =============================================================================
# App config (sidebar visible)
# =============================================================================
st.set_page_config(
    page_title="Crowd Guardian: Machine learning based crowd dynamics analysis for effective stampede detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Inference defaults (no UI knobs) ----------
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
TARGET_FPS    = None  # None = use every frame

# ---------- Model location ----------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "models"; CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/zswpw1ucbj7bkkkykc8oc/deep_cnn_stampede.h5"
    "?rlkey=e863b9skyvpyn0dn4gbwxd71s&st=uvgqjq7q&dl=1"
)
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

# =============================================================================
# Global styles (HTML/CSS) — theme, hero, cards, sidebar, timeline
# =============================================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      :root{
        --bg:#0b1220; --panel:#0f172a; --panel2:#111827; --line:#1f2937;
        --muted:rgba(148,163,184,.35); --muted2:rgba(148,163,184,.18);
        --text:#e5e7eb; --accent:#ef4444; --accent2:#f97316; --good:#10b981; --bad:#ef4444;
      }
      html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
        color: var(--text);
      }
      .main .block-container {max-width: 1024px; padding-top: 0.8rem; padding-bottom: 2.0rem;}
      body {background: radial-gradient(1200px 420px at -10% -15%, rgba(59,130,246,.14), transparent), var(--bg) !important;}

      /* Hide Streamlit chrome for a cleaner look */
      header{visibility:hidden;}
      [data-testid="stToolbar"]{display:none;}
      #MainMenu{visibility:hidden;}
      footer{visibility:hidden;}

      /* HERO */
      .cg-hero {
        margin-top: 8px;
        padding: 26px 30px;
        border-radius: 18px;
        border: 1px solid var(--muted2);
        background:
          radial-gradient(1200px 420px at 0% -10%, rgba(59,130,246,.16), transparent),
          linear-gradient(180deg, rgba(17,24,39,.55), rgba(2,6,23,.45));
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
      }
      .cg-title  {font-size: 2.45rem; line-height: 1.1; margin: 0 0 .35rem 0; letter-spacing:.2px;}
      .cg-subtle {opacity:.92; margin:0; font-size: 1.06rem;}

      /* Section H2 (centered) */
      .cg-h2 {text-align:center; margin: 1.1rem 0 .7rem 0; font-size:1.35rem;}

      /* Cards */
      .cg-card {
        border: 1px solid var(--muted2);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(15,23,42,.45);
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
      }
      .cg-upload {border: 1px dashed var(--muted); border-radius: 12px; padding: 12px 12px;}

      /* Pill */
      .cg-center {display:flex; justify-content:center; margin-top:.6rem;}
      .pill {display:inline-block; padding:3px 12px; border-radius:999px; font-size:12px;
             border:1px solid var(--muted); background: rgba(30,41,59,.55)}
      .ok   {background: rgba(16,185,129,.18); color:#a7f3d0; border-color: rgba(16,185,129,.35)}
      .err  {background: rgba(239,68,68,.18); color:#fecaca; border-color: rgba(239,68,68,.35)}

      /* Primary button */
      .stButton>button {
        width: 100%;
        padding: 12px 14px;
        border-radius: 12px;
        font-weight: 700;
        border: 0;
        color: #fff;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        box-shadow: 0 8px 24px rgba(249,115,22,.35);
      }
      .stButton>button:hover {filter: brightness(1.07)}

      /* -------------------- SIDEBAR DECOR -------------------- */
      [data-testid="stSidebar"] {
        min-width: 320px; max-width: 360px;
        border-right: 1px solid var(--muted2);
      }
      [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 60%, #1f2937 100%);
        color: var(--text);
      }
      .sb-brand { font-weight: 700; font-size: 1.08rem; letter-spacing:.2px; margin: 10px 12px 8px 12px; }
      .sb-card {
        border:1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.04);
        border-radius: 14px;
        padding: 12px 14px;
        margin: 12px;
      }
      .sb-small {font-size:12px; opacity:.85;}
      .sb-card ul, .sb-card ol {padding-left: 1.05rem; margin: .25rem 0 0 0;}
      .sb-card li {margin-bottom: 6px;}

      /* Timeline */
      .timeline-wrap {margin-top: .75rem; margin-bottom: .75rem;}
      .timeline {
        --event: var(--accent);
        --track: rgba(148,163,184,.25);
        height: 14px;
        border-radius: 999px;
        border: 1px solid var(--muted);
        background: var(--track);
      }
      .timeline-legend {display:flex; gap:10px; align-items:center; font-size:12px; opacity:.9;}
      .legend-swatch {width:14px; height:10px; border-radius:3px; display:inline-block; border:1px solid var(--muted);}
      .swatch-event {background:var(--accent);}
      .swatch-track {background:rgba(148,163,184,.25);}

      /* Dataframe polish */
      .stDataFrame {border-radius: 10px; overflow:hidden; border:1px solid var(--muted2);}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar — PROJECT DETAILS (no settings)
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sb-brand">🛡️ Crowd Guardian</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sb-card">
          <b>Overview</b><br/>
          Detect potential stampede intervals in crowd videos by combining a CNN on grayscale frames
          with a head-count drop heuristic and event aggregation.
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="sb-card">
          <b>Method</b>
          <ul>
            <li>Grayscale frames → head-like blobs (OpenCV SimpleBlobDetector).</li>
            <li>CNN on 100×100 grayscale inputs (τ={CNN_THRESHOLD:.2f}).</li>
            <li>Head-drop: Δ≥{ABS_DROP} or {int(REL_DROP*100)}% vs. previous frame.</li>
            <li>Decision: <code>{COMBINE_RULE}</code> of CNN &amp; head-drop.</li>
            <li>Merge consecutive positives into events ≥ {MIN_EVENT_SEC:.1f}s.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="sb-card">
          <b>Outputs</b>
          <ul>
            <li><code>events.csv</code> — start/end/duration of intervals</li>
            <li><code>frame_preds.csv</code> — per-frame metrics & labels</li>
            <li><code>labeled.mp4</code> — overlay video with live stats</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )
    # Environment chip
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
    except Exception:
        tf_ver = "not loaded"
    st.markdown(
        f'<div class="sb-card sb-small">Environment: TF {tf_ver} • NumPy {np.__version__} • OpenCV {cv2.__version__}</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# Hero
# =============================================================================
st.markdown(
    '<div class="cg-hero">'
    '<div class="cg-title">Crowd Guardian</div>'
    '<div class="cg-subtle">Machine learning based crowd dynamics analysis for effective stampede detection</div>'
    '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# Helpers (ML + I/O)
# =============================================================================
def _normalize_dropbox(url: str) -> str:
    if "dropbox.com" not in url: return url
    parts = urlparse(url); q = parse_qs(parts.query); q["dl"] = ["1"]
    return urlunparse(parts._replace(query=urlencode({k: v[0] for k, v in q.items()})))

def _download_to_cache(url: str) -> str:
    url = _normalize_dropbox(url)
    dst = CACHE_DIR / ("model_" + hashlib.sha1(url.encode()).hexdigest() + ".h5")
    if not dst.exists():
        with st.spinner("Preparing model…"):
            with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
    return str(dst)

@st.cache_resource(show_spinner=False)
def load_cnn(path_or_url: str):
    import tensorflow as tf
    if path_or_url.startswith(("http://", "https://")):
        local = _download_to_cache(path_or_url)
    else:
        p = Path(path_or_url)
        local = str(p if p.is_absolute() else (APP_DIR / p))
    return tf.keras.models.load_model(local, compile=False)  # Keras 3 safe

def sec_to_tc(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def build_blob_detector(frame_w, frame_h, min_frac=MIN_FRAC, max_frac=MAX_FRAC,
                        min_circ=MIN_CIRC, min_inertia=MIN_INER, thresh_step=10):
    area = float(frame_w * frame_h)
    minArea = max(5.0, min_frac * area); maxArea = max(minArea + 1.0, max_frac * area)
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold, p.maxThreshold, p.thresholdStep = 10, 220, thresh_step
    p.filterByArea, p.minArea, p.maxArea = True, minArea, maxArea
    p.filterByCircularity, p.minCircularity = True, min_circ
    p.filterByInertia, p.minInertiaRatio = True, min_inertia
    p.filterByConvexity = p.filterByColor = False
    return (cv2.SimpleBlobDetector(p) if int(cv2.__version__.split('.')[0]) < 3
            else cv2.SimpleBlobDetector_create(p))

def detect_heads_gray(gray, detector):
    kps = detector.detect(cv2.GaussianBlur(gray, (5,5), 0))
    return [(float(k.pt[0]), float(k.pt[1])) for k in kps]

def assign_matches(prev_pts, curr_pts, max_dist):
    if not prev_pts or not curr_pts:
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
    return matches, sorted(un_prev), sorted(un_curr)

def preprocess_for_cnn(gray):
    x = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    return np.expand_dims(x, axis=(0, -1))

def combine_labels(y_heads, y_cnn, rule=COMBINE_RULE):
    rule = (rule or "and").lower()
    if rule == "and": return 1 if (y_heads and y_cnn) else 0
    if rule == "or":  return 1 if (y_heads or  y_cnn) else 0
    if rule == "cnn_only":   return int(y_cnn)
    if rule == "heads_only": return int(y_heads)
    return int(y_heads)

def _get_ffmpeg_exe():
    ff = shutil.which("ffmpeg")
    if ff: return ff
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None

def transcode_to_h264(src_path: str, dst_path: str, fps: float):
    ff = _get_ffmpeg_exe()
    if ff is None: return src_path, False, "ffmpeg not found"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = [ff, "-y", "-loglevel", "error", "-i", src_path,
           "-r", str(int(round(fps if fps and fps > 0 else 25))),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = (proc.returncode == 0) and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    return (dst_path if ok else src_path), ok, (proc.stderr or "")

# =============================================================================
# Load model (silent) + status
# =============================================================================
model, load_err = None, None
try:
    model = load_cnn(MODEL_URL)
except Exception as e:
    load_err = str(e)

st.markdown(
    f'<div class="cg-center"><span class="pill {"ok" if model and not load_err else "err"}">'
    f'{"Model loaded" if model and not load_err else "Model error"}</span></div>',
    unsafe_allow_html=True,
)
if load_err:
    st.caption(load_err)

# =============================================================================
# Upload
# =============================================================================
st.markdown('<h2 class="cg-h2">Upload a crowd video</h2>', unsafe_allow_html=True)
st.markdown('<div class="cg-card cg-upload">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop a video (MP4, MOV, MKV, AVI, MPEG4) or browse",
    type=["mp4","mov","mkv","avi","mpeg4"],
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)
go = st.button("Analyze")

# =============================================================================
# Core analysis
# =============================================================================
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
    draw_links=DRAW_LINKS,
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

    # Use numeric types for plotting
    frames_rows = [("frame_index","time_sec","timecode","head_count","delta_vs_prev",
                    "prob_cnn","cnn_label","heads_label","final_label")]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base  = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = os.path.join(out_dir, f"{base}_{stamp}_raw.avi")
    mp4_path = os.path.join(out_dir, f"{base}_{stamp}_labeled.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(raw_path, fourcc, fps if fps and fps > 0 else 25.0, (W, H))

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
            frames_rows.append((f, t, tc, curr_count, int(delta),
                                p_cnn, y_cnn, y_heads, final_label))

            if final_label == 1 and not in_event:
                in_event, start_f, start_t = True, f, t
            elif final_label == 0 and in_event:
                dur_frames = (f - start_f) // step
                if dur_frames >= min_event_frames:
                    end_t = (f-1) / fps
                    events_rows.append((start_f, f-1, start_t, end_t,
                                        sec_to_tc(start_t), sec_to_tc(end_t), end_t-start_t))
                in_event, start_f, start_t = False, None, None

            # overlay
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
            cv2.putText(
                vis,
                f"heads={curr_count}  Δ={delta:+d}  p_cnn={p_cnn:.2f} (τ={cnn_threshold:.2f})  "
                f"rule={combine_rule}  label={'Stampede' if final_label==1 else 'No Stampede'}  t={tc}",
                (12, banner_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA
            )

            if out.isOpened(): out.write(vis)
            prev_pts, prev_count = head_pts, curr_count

            processed += 1
            if total_steps:
                prog.progress(min(1.0, processed/total_steps))
                status.write(f"Processed {processed}/{total_steps} sampled frames…")
        f += 1

    if in_event and start_f is not None:
        end_t = (f-1) / fps
        events_rows.append((start_f, f-1, start_t, end_t,
                            sec_to_tc(start_t), sec_to_tc(end_t), end_t-start_t))

    cap.release()
    if out.isOpened(): out.release()

    playable_path, ok, _ = transcode_to_h264(raw_path, mp4_path, fps)
    if not ok: st.warning("Transcode failed; preview may not play.")

    prog.progress(1.0); status.write("Done.")
    df_frames = pd.DataFrame(frames_rows[1:], columns=frames_rows[0])
    df_events = pd.DataFrame(events_rows[1:], columns=events_rows[0])
    return df_frames, df_events, playable_path

# ---- Build a CSS gradient timeline bar from events ----
def make_timeline_gradient(total_sec: float, events_df: pd.DataFrame) -> str:
    if total_sec <= 0: return ""
    stops = []
    last_pct = 0.0
    if not events_df.empty:
        for _, r in events_df.iterrows():
            s = float(r["start_time_sec"]); e = float(r["end_time_sec"])
            s_pct = max(0.0, min(100.0, (s / total_sec) * 100.0))
            e_pct = max(0.0, min(100.0, (e / total_sec) * 100.0))
            if s_pct > last_pct:
                stops.append(f"var(--track) {last_pct:.3f}%, var(--track) {s_pct:.3f}%")
            stops.append(f"var(--event) {s_pct:.3f}%, var(--event) {e_pct:.3f}%")
            last_pct = e_pct
    if last_pct < 100.0:
        stops.append(f"var(--track) {last_pct:.3f}%, var(--track) 100%")
    return "linear-gradient(90deg, " + ", ".join(stops) + ")"

# ---- Render results with visuals ----
def render_results(df_frames, df_events, labeled_path):
    st.markdown('<h2 class="cg-h2">Results</h2>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3 = st.columns(3)
    total_events = int(len(df_events))
    total_dur = float(df_events["duration_sec"].sum()) if not df_events.empty else 0.0
    longest = float(df_events["duration_sec"].max()) if not df_events.empty else 0.0
    c1.metric("Events", total_events)
    c2.metric("Total Duration (s)", f"{total_dur:.2f}")
    c3.metric("Longest Event (s)", f"{longest:.2f}")

    # Confetti if we detected anything (JS in an iframe)
    if total_events > 0:
        components.html("""
        <div id="confetti"></div>
        <canvas id="c"></canvas>
        <style>
          #c{position:relative;width:100%;height:140px;display:block;border-radius:12px;margin:6px 0 4px 0;
             background:linear-gradient(90deg, rgba(16,185,129,.18), rgba(239,68,68,.18));}
        </style>
        <script>
          const canvas = document.getElementById('c');
          const ctx = canvas.getContext('2d');
          function resize(){canvas.width=canvas.clientWidth; canvas.height=140;}
          window.addEventListener('resize', resize); resize();
          const confetti = [];
          function spawn(){for(let i=0;i<50;i++){confetti.push({
            x: Math.random()*canvas.width, y: -10 - Math.random()*30,
            vx: (Math.random()-0.5)*1.5, vy: 1+Math.random()*2.5,
            s: 2+Math.random()*3, a: Math.random()*Math.PI
          });}}
          spawn();
          function tick(){
            ctx.clearRect(0,0,canvas.width,canvas.height);
            confetti.forEach(p=>{
              p.x+=p.vx; p.y+=p.vy; p.a+=0.1;
              ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.a);
              ctx.fillStyle = ['#ef4444','#f97316','#10b981','#60a5fa'][Math.floor(Math.random()*4)];
              ctx.fillRect(-p.s/2, -p.s/2, p.s, p.s*2);
              ctx.restore();
            });
            requestAnimationFrame(tick);
          }
          tick();
          setTimeout(()=>{spawn()}, 600);
        </script>
        """, height=160)

    # Timeline bar
    st.markdown('<div class="cg-card">', unsafe_allow_html=True)
    total_sec = float(df_frames["time_sec"].max()) if not df_frames.empty else 0.0
    grad = make_timeline_gradient(total_sec, df_events)
    st.markdown('<div class="timeline-wrap">', unsafe_allow_html=True)
    if grad:
        st.markdown(f'<div class="timeline" style="background:{grad}"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="timeline"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="timeline-legend">'
        '<span class="legend-swatch swatch-track"></span> Track'
        '<span class="legend-swatch swatch-event"></span> Detected Event'
        f'<span style="margin-left:auto; font-size:12px; opacity:.8;">0s — {total_sec:.2f}s</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts: Altair (probability & head count)
    if not df_frames.empty:
        base = alt.Chart(df_frames).properties(height=240)
        left, right = st.columns(2)

        with left:
            st.subheader("CNN Probability")
            line_prob = base.mark_line().encode(
                x=alt.X('frame_index:Q', title='Frame'),
                y=alt.Y('prob_cnn:Q', title='Probability', scale=alt.Scale(domain=[0,1])),
                tooltip=['frame_index','timecode','prob_cnn']
            )
            thresh = base.mark_rule(strokeDash=[4,4]).encode(y=alt.datum(CNN_THRESHOLD))
            st.altair_chart((line_prob + thresh).interactive(), use_container_width=True)

        with right:
            st.subheader("Estimated Head Count")
            line_head = base.mark_line().encode(
                x=alt.X('frame_index:Q', title='Frame'),
                y=alt.Y('head_count:Q', title='Head Count'),
                tooltip=['frame_index','timecode','head_count','delta_vs_prev']
            )
            st.altair_chart(line_head.interactive(), use_container_width=True)

    # Tables + downloads
    if not df_events.empty:
        st.subheader("Detected intervals")
        st.dataframe(df_events, use_container_width=True)
    else:
        st.info("No stampede detected.")

    st.subheader("Per-frame predictions")
    st.dataframe(df_frames.head(1000), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ events.csv", df_events.to_csv(index=False).encode("utf-8"),
                           file_name="events.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ frame_preds.csv", df_frames.to_csv(index=False).encode("utf-8"),
                           file_name="frame_preds.csv", mime="text/csv", use_container_width=True)
    with c3:
        if os.path.exists(labeled_path) and os.path.getsize(labeled_path) > 0:
            with open(labeled_path, "rb") as fh:
                st.download_button("⬇️ labeled.mp4", fh.read(), file_name=os.path.basename(labeled_path),
                                   mime="video/mp4", use_container_width=True)
        else:
            st.button("Video unavailable", disabled=True, use_container_width=True)

    st.markdown('<h2 class="cg-h2">Labeled Video Preview</h2>', unsafe_allow_html=True)
    if os.path.exists(labeled_path): st.video(labeled_path)
    else: st.info("Preview unavailable.")

# =============================================================================
# Run
# =============================================================================
if go:
    if not model:
        st.error("Model not loaded.")
    elif not uploaded:
        st.warning("Please upload a video.")
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.close()
        with st.spinner("Analyzing video…"):
            df_frames, df_events, labeled_path = analyze_video(tmp.name, model)
        render_results(df_frames, df_events, labeled_path)
