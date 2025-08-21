# app.py — Crowd Guardian (Video only)
# Premium UI: custom HTML/CSS, decorated sidebar, status banner (no timeline bar),
# Altair charts, JS confetti, animated particles background, and
# **session_state persistence** so downloads don't "refresh away" your results.

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
# App config
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

# ---------- XAI / Snapshot settings ----------
XAI_ENABLED   = True        # Grad-CAM + zone risk
GRID_ROWS     = 4
GRID_COLS     = 4
W_CAM, W_DROP, W_FLOW = 0.5, 0.4, 0.1
FLOW_ENABLED  = False       # set True to include optical flow
SNAPSHOT_ONLY = True        # ✅ Save one red-marked frame per event; no full overlay video

# ---------- Model location ----------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "models"; CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/zswpw1ucbj7bkkkykc8oc/deep_cnn_stampede.h5"
    "?rlkey=e863b9skyvpyn0dn4gbwxd71s&st=uvgqjq7q&dl=1"
)
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

# =============================================================================
# Global styles (HTML/CSS)
# =============================================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      :root{
        --bg:#0a0f1a; --panel:#0e1526; --panel2:#101826;
        --muted:rgba(148,163,184,.35); --muted2:rgba(148,163,184,.18);
        --text:#e6e9ef; --accent:#ef4444; --accent2:#f97316;
      }
      html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
        color: var(--text);
      }
      .main .block-container {max-width: 1080px; padding-top: .6rem; padding-bottom: 2rem;}
      body {
        background:
          radial-gradient(1000px 380px at -10% -15%, rgba(99,102,241,.16), transparent),
          radial-gradient(1000px 420px at 110% -20%, rgba(236,72,153,.14), transparent),
          var(--bg) !important;
      }
      header{visibility:hidden;} [data-testid="stToolbar"]{display:none;} #MainMenu{visibility:hidden;} footer{visibility:hidden;}

      .cg-hero {
        margin-top: 8px; padding: 30px 32px; border-radius: 20px;
        border: 1px solid var(--muted2);
        background:
          radial-gradient(1200px 420px at 0% -10%, rgba(59,130,246,.16), transparent),
          linear-gradient(180deg, rgba(17,24,39,.55), rgba(2,6,23,.45));
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,.25);
      }
      .cg-title  {font-size: 2.75rem; line-height: 1.08; margin: 0 0 .35rem 0; letter-spacing:.1px;}
      .cg-subtle {opacity:.95; margin:0; font-size: 1.08rem;}

      .cg-h2 {text-align:center; margin: 1.1rem 0 .7rem 0; font-size:1.35rem;}
      .cg-card {border: 1px solid var(--muted2); border-radius: 16px; padding: 14px 16px;
                background: rgba(15,23,42,.45); box-shadow: 0 10px 30px rgba(0,0,0,.25);}

      .cg-center {display:flex; justify-content:center; margin-top:.6rem;}
      .pill {display:inline-block; padding:3px 12px; border-radius:999px; font-size:12px;
             border:1px solid var(--muted); background: rgba(30,41,59,.55)}
      .ok   {background: rgba(16,185,129,.18); color:#a7f3d0; border-color: rgba(16,185,129,.35)}
      .err  {background: rgba(239,68,68,.18); color:#fecaca; border-color: rgba(239,68,68,.35)}

      .stButton>button {
        width: 100%; padding: 12px 14px; border-radius: 12px; font-weight: 700; border: 0; color: #fff;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        box-shadow: 0 8px 24px rgba(249,115,22,.35);
      }
      .stButton>button:hover {filter: brightness(1.07)}

      [data-testid="stSidebar"] {min-width: 330px; max-width: 360px; border-right: 1px solid var(--muted2);}
      [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 60%, #182234 100%); color: var(--text);
      }
      .sb-brand { font-weight: 700; font-size: 1.10rem; letter-spacing:.2px; margin: 12px 14px 10px 14px; }

      .sb-card { border:1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.04);
                 border-radius: 14px; padding: 12px 14px; margin: 12px; }
      .sb-small {font-size:12px; opacity:.85;}
      .sb-card ul, .sb-card ol {padding-left: 1.05rem; margin: .25rem 0 0 0;}
      .sb-card li {margin-bottom: 6px;}

      .capstone-badge {
        display:inline-flex; align-items:center; gap:8px; margin-bottom:10px;
        background: linear-gradient(90deg, rgba(239,68,68,.18), rgba(249,115,22,.18));
        border:1px solid rgba(249,115,22,.35); color:#ffd7c2; padding:6px 10px; border-radius:999px; font-size:12px;
      }
      .capstone-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .cap-card { border:1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.03);
                  border-radius: 12px; padding: 10px 12px; }
      .cap-title {font-weight:700; margin-bottom:6px;}
      .cap-kv {font-size: 13px; line-height: 1.25rem;}
      .cap-name {font-weight:600;}
      .cap-id {opacity:.85;}
      .divider {height:1px; background:rgba(255,255,255,.06); margin:10px 0;}

      .status-banner{
        display:flex; align-items:center; justify-content:center;
        gap:10px; height:52px; border-radius:12px; margin:12px 0;
        font-weight:800; letter-spacing:.3px; text-transform:uppercase;
        border:1px solid rgba(255,255,255,.10);
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }
      .status-dot{width:10px; height:10px; border-radius:50%; display:inline-block; box-shadow:0 0 0 3px rgba(255,255,255,.06) inset;}
      .status-ok{  background: linear-gradient(90deg, rgba(239,68,68,.22), rgba(249,115,22,.22)); color:#ffe5d5; }
      .status-safe{background: linear-gradient(90deg, rgba(16,185,129,.22), rgba(59,130,246,.22)); color:#dcfce7; }
      .status-ok .status-dot{background:#ef4444;} .status-safe .status-dot{background:#10b981;}

      .stDataFrame {border-radius: 10px; overflow:hidden; border:1px solid var(--muted2);}
      [data-testid="stFileUploadDropzone"]{ margin-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Background Particles (JS canvas behind page)
# =============================================================================
components.html("""
<canvas id="cg-bg"></canvas>
<style>
  #cg-bg{position:fixed; inset:0; z-index:-2; background:transparent;}
</style>
<script>
  const c = document.getElementById('cg-bg'), ctx = c.getContext('2d');
  function resize(){ c.width = innerWidth; c.height = innerHeight; }
  addEventListener('resize', resize); resize();

  const N = 120;
  const P = Array.from({length:N}, () => ({
    x: Math.random()*c.width,
    y: Math.random()*c.height,
    vx: -0.25 + Math.random()*0.5,
    vy: -0.25 + Math.random()*0.5,
    s: 0.6 + Math.random()*1.6
  }));

  function tick(){
    ctx.clearRect(0,0,c.width,c.height);
    P.forEach(p=>{
      p.x += p.vx; p.y += p.vy;
      if(p.x<0) p.x=c.width; if(p.x>c.width) p.x=0;
      if(p.y<0) p.y=c.height; if(p.y>c.height) p.y=0;

      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 6+p.s*2);
      g.addColorStop(0, 'rgba(255,255,255,0.8)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(p.x,p.y, 1.2+p.s, 0, Math.PI*2); ctx.fill();
    });
    requestAnimationFrame(tick);
  }
  tick();
</script>
""", height=0)

# =============================================================================
# Sidebar — PROJECT DETAILS
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
        """
        <div class="sb-card">
          <span class="capstone-badge">🎓 Capstone-C Project • East West University</span>
          <div class="capstone-grid">
            <div class="cap-card">
              <div class="cap-title">Submitted By</div>
              <div class="cap-kv">
                <div><span class="cap-name">Tasfia Tahsin Annita</span> <span class="cap-id">• 2021-3-60-031</span></div>
                <div><span class="cap-name">Md. Shakil Hossain</span> <span class="cap-id">• 2021-3-60-088</span></div>
                <div><span class="cap-name">Kanij Fatema</span> <span class="cap-id">• 2021-3-60-095</span></div>
                <div><span class="cap-name">Samura Rahman</span> <span class="cap-id">• 2021-3-60-064</span></div>
              </div>
            </div>
            <div class="cap-card">
              <div class="cap-title">Supervised By</div>
              <div class="cap-kv">
                <div class="cap-name">Dr. Anisur Rahman</div>
                <div>Proctor</div>
                <div>Associate Professor</div>
                <div>Department of Computer Science and Engineering</div>
                <div>East West University</div>
              </div>
            </div>
          </div>
          <div class="divider"></div>
          <div class="sb-small">This application demonstrates ML-assisted analysis of crowd dynamics with visual evidence and interval summaries.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
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

# =============== XAI: Robust Grad-CAM + zone grid helpers ====================
def gradcam_heatmap(model, x_100x100x1, conv_layer_name=None):
    """Grad-CAM robust to Keras 3/TF2.20 model shapes and outputs."""
    import tensorflow as tf

    def _select_score_vector(preds_any):
        t = preds_any
        if isinstance(t, dict):
            t = t[sorted(t.keys())[0]]
        if isinstance(t, (list, tuple)):
            t = t[0]
        t = tf.convert_to_tensor(t)
        rank = t.shape.rank
        if rank is None:
            t2 = tf.reshape(t, (tf.shape(t)[0], -1)); return t2[:, -1]
        if rank == 1:
            return t
        if rank == 2:
            c = t.shape[-1]
            return tf.squeeze(t, axis=-1) if c == 1 else t[:, -1]
        t2 = tf.reshape(t, (tf.shape(t)[0], -1))
        return t2[:, -1]

    target_layer = None
    if conv_layer_name is not None:
        try:
            L = model.get_layer(conv_layer_name)
            if len(L.output.shape) == 4:
                target_layer = L
        except Exception:
            target_layer = None
    if target_layer is None:
        for L in reversed(model.layers):
            try:
                if len(L.output.shape) == 4:
                    target_layer = L; break
            except Exception:
                continue
    if target_layer is None:
        return None

    try:
        grad_model = __import__("tensorflow").keras.Model(
            inputs=model.input, outputs=[target_layer.output, model.output]
        )
    except Exception:
        grad_model = __import__("tensorflow").keras.Model(
            inputs=model.inputs, outputs=[target_layer.output, model.outputs]
        )

    with __import__("tensorflow").GradientTape() as tape:
        conv_out, preds = grad_model(x_100x100x1, training=False)
        score_vec = _select_score_vector(preds)

    grads = tape.gradient(score_vec, conv_out)
    if grads is None:
        return None
    weights = __import__("tensorflow").reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = __import__("tensorflow").nn.relu(__import__("tensorflow").reduce_sum(weights * conv_out, axis=-1))
    cam = cam[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def upscale_cam(cam_small, W, H):
    if cam_small is None:
        return None
    return cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_CUBIC)

def make_grid(W, H, rows=4, cols=4):
    cell_w, cell_h = W // cols, H // rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c*cell_w, r*cell_h
            x1 = W if c == cols-1 else (c+1)*cell_w
            y1 = H if r == rows-1 else (r+1)*cell_h
            boxes.append(((x0, y0, x1, y1), f"r{r}c{c}"))
    return boxes, cell_w, cell_h

def heads_per_cell(head_pts, rows, cols, cell_w, cell_h):
    counts = np.zeros((rows, cols), dtype=int)
    for (x, y) in head_pts:
        c = min(cols-1, max(0, int(x // cell_w)))
        r = min(rows-1, max(0, int(y // cell_h)))
        counts[r, c] += 1
    return counts

# ---- Robust image loader for Streamlit (avoid st.image TypeError) ----
def _load_image_bytes_png(path: str):
    """
    Return PNG-encoded bytes for the image at `path` using OpenCV.
    This bypasses PIL and avoids the 'channels' argument entirely.
    """
    try:
        # Read in a way that tolerates weird paths
        data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if data is None:
            data = cv2.imread(path, cv2.IMREAD_COLOR)
        if data is None:
            return None
        # We can directly encode BGR -> PNG; no need to convert to RGB for PNG encoding
        ok, buf = cv2.imencode(".png", data)
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None

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
# (NEW) Re-render persisted results so downloads don't clear the page
# =============================================================================
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

    # Confetti if detected
    if total_events > 0:
        components.html("""
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

    # Status banner (only)
    st.markdown('<div class="cg-card">', unsafe_allow_html=True)
    status_cls = "status-ok" if total_events > 0 else "status-safe"
    status_text = "Stampede detected" if total_events > 0 else "No stampede detected"
    st.markdown(
        f'<div class="status-banner {status_cls}"><span class="status-dot"></span>{status_text}</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts
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
        st.info("No stampede intervals found.")

    st.subheader("Per-frame predictions")
    st.dataframe(df_frames.head(1000), use_container_width=True)

    # Stable keys so reruns don't duplicate widgets
    uid = os.path.splitext(os.path.basename(labeled_path or "na.mp4"))[0] if labeled_path else "na"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ events.csv", df_events.to_csv(index=False).encode("utf-8"),
                           file_name="events.csv", mime="text/csv",
                           use_container_width=True, key=f"dl_events_{uid}")
    with c2:
        st.download_button("⬇️ frame_preds.csv", df_frames.to_csv(index=False).encode("utf-8"),
                           file_name="frame_preds.csv", mime="text/csv",
                           use_container_width=True, key=f"dl_frames_{uid}")
    with c3:
        if labeled_path and os.path.exists(labeled_path) and os.path.getsize(labeled_path) > 0:
            with open(labeled_path, "rb") as fh:
                st.download_button("⬇️ labeled.mp4", fh.read(), file_name=os.path.basename(labeled_path),
                                   mime="video/mp4", use_container_width=True, key=f"dl_video_{uid}")
        else:
            st.button("Video unavailable", disabled=True, use_container_width=True, key=f"dl_na_{uid}")

    # Extra download if XAI ran
    extra = st.session_state.get("video_xai", {})
    df_events_zones = extra.get("events_zones")
    if isinstance(df_events_zones, pd.DataFrame) and not df_events_zones.empty:
        st.download_button("⬇️ events_zones.csv",
                           df_events_zones.to_csv(index=False).encode("utf-8"),
                           file_name="events_zones.csv", mime="text/csv",
                           use_container_width=True, key=f"dl_events_z_{uid}")

    # --- Event Snapshots (red-marked zone) ---
    snapshots = extra.get("snapshots", [])
    if snapshots:
        st.markdown('<h2 class="cg-h2">Event Snapshots (red-marked zone)</h2>', unsafe_allow_html=True)
        snap_rows = []
        for s in snapshots:
            path = s.get("path") or ""
            risk = float(s.get("risk_score", 0.0)) if s.get("risk_score", None) is not None else 0.0
            caption = f"Event {s.get('event_id','?')} • frame {s.get('frame_index','?')} • {s.get('timecode','?')} • {s.get('zone_id','?')} (risk {risk:.2f})"

            if isinstance(path, str) and os.path.exists(path) and os.path.getsize(path) > 0:
                img_bytes = _load_image_bytes_png(path)
                col1, col2 = st.columns([2,1])
                with col1:
                    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
                        # No 'channels' arg — safest across Streamlit versions
                        st.image(img_bytes, caption=caption, use_container_width=True)
                    else:
                        st.warning(f"Snapshot could not be displayed (event {s.get('event_id','?')}).")
                with col2:
                    with open(path, "rb") as fh:
                        st.download_button("⬇️ Download snapshot", fh.read(),
                                           file_name=os.path.basename(path),
                                           mime="image/jpeg", use_container_width=True)
            else:
                st.warning(f"Snapshot file missing for event {s.get('event_id','?')} (path: {path})")

            snap_rows.append({
                "event_id": s.get("event_id"),
                "frame_index": s.get("frame_index"),
                "timecode": s.get("timecode"),
                "zone_id": s.get("zone_id"),
                "x0": s.get("x0"), "y0": s.get("y0"), "x1": s.get("x1"), "y1": s.get("y1"),
                "risk_score": risk,
                "path": path,
            })
        df_snaps = pd.DataFrame(snap_rows)
        st.download_button("⬇️ event_snapshots.csv",
                           df_snaps.to_csv(index=False).encode("utf-8"),
                           file_name="event_snapshots.csv", mime="text/csv",
                           use_container_width=True)

    # Keep your original "Labeled Video Preview" section
    st.markdown('<h2 class="cg-h2">Labeled Video Preview</h2>', unsafe_allow_html=True)
    if labeled_path and os.path.exists(labeled_path): st.video(labeled_path)
    else: st.info("Preview unavailable.")

# ---- Re-render results from session on EVERY run (prevents ‘refresh’ loss)
if "video_results" in st.session_state:
    _res = st.session_state["video_results"]
    render_results(_res["df_frames"], _res["df_events"], _res.get("labeled_path"))

# =============================================================================
# Upload
# =============================================================================
st.markdown('<h2 class="cg-h2">Upload a crowd video</h2>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop a video (MP4, MOV, MKV, AVI, MPEG4) or browse",
    type=["mp4","mov","mkv","avi","mpeg4"],
    label_visibility="collapsed",
)
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

    frames_rows = [("frame_index","time_sec","timecode","head_count","delta_vs_prev",
                    "prob_cnn","cnn_label","heads_label","final_label")]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base  = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = os.path.join(out_dir, f"{base}_{stamp}_raw.avi")
    mp4_path = os.path.join(out_dir, f"{base}_{stamp}_labeled.mp4")

    # Snapshot mode: no video writer
    out = None

    prev_pts, prev_count = [], None
    in_event, start_f, start_t = False, None, None
    min_event_frames = max(1, int(round(min_event_sec * (fps/step))))
    event_id = 0  # XAI: running event index

    # XAI state
    grid_boxes, cell_w, cell_h = make_grid(W, H, rows=GRID_ROWS, cols=GRID_COLS)
    prev_counts_zone = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    prev_gray_for_flow = None
    events_z_rows = []  # buffer rows for events_zones.csv
    snapshots = []
    current_best = None  # best frame within an event

    prog = st.progress(0.0); status = st.empty()
    processed = 0; total_steps = (N // step + 1) if N > 0 else 0

    def save_current_best():
        nonlocal current_best, snapshots
        if not current_best:
            return
        frame = current_best.pop("frame", None)
        if frame is None:
            current_best = None
            return

        x0,y0,x1,y1 = current_best["x0"], current_best["y0"], current_best["x1"], current_best["y1"]
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,0,255), 3)
        label = (
            f"Event {current_best['event_id']} • {current_best['timecode']} • "
            f"{current_best['zone_id']} • risk {float(current_best['risk_score']):.2f}"
        )
        cv2.putText(frame, label, (max(6,x0), max(24,y0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        snap_path = os.path.join(out_dir, f"{base}_{stamp}_event{current_best['event_id']}_snapshot.jpg")
        # Atomic JPEG write
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            tmp_path = snap_path + ".tmp"
            with open(tmp_path, "wb") as fh:
                fh.write(buf.tobytes())
            os.replace(tmp_path, snap_path)
            snapshots.append({
                "event_id": current_best["event_id"],
                "frame_index": current_best["frame_index"],
                "timecode": current_best["timecode"],
                "zone_id": current_best["zone_id"],
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "risk_score": float(current_best["risk_score"]),
                "path": snap_path
            })
        else:
            snapshots.append({
                "event_id": current_best["event_id"],
                "frame_index": current_best["frame_index"],
                "timecode": current_best["timecode"],
                "zone_id": current_best["zone_id"],
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "risk_score": float(current_best["risk_score"]),
                "path": ""
            })
        current_best = None

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

            # Event boundary bookkeeping
            if final_label == 1 and not in_event:
                in_event, start_f, start_t = True, f, t
                event_id += 1
                current_best = None
            elif final_label == 0 and in_event:
                dur_frames = (f - start_f) // step
                if dur_frames >= min_event_frames:
                    end_t = (f-1) / fps
                    events_rows.append((start_f, f-1, start_t, end_t,
                                        sec_to_tc(start_t), sec_to_tc(end_t), end_t-start_t))
                    save_current_best()
                in_event, start_f, start_t = False, None, None

            # -------------------- XAI computations (positive frames) --------------------
            if XAI_ENABLED and final_label == 1:
                cam_small = gradcam_heatmap(model, x)
                cam_up = upscale_cam(cam_small, W, H) if cam_small is not None else None
                counts_now = heads_per_cell(head_pts, GRID_ROWS, GRID_COLS, cell_w, cell_h)
                delta_zone = prev_counts_zone - counts_now
                prev_counts_zone = counts_now

                if FLOW_ENABLED:
                    if prev_gray_for_flow is None:
                        flow_mag = np.zeros((H, W), dtype=np.float32)
                    else:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray_for_flow, gray,
                                                            pyr_scale=0.5, levels=3, winsize=15,
                                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                        flow_mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).astype(np.float32)
                    prev_gray_for_flow = gray.copy()
                else:
                    flow_mag = np.zeros((H, W), dtype=np.float32)

                max_heads_now = max(1.0, float(counts_now.max()))
                mean_flow = float(flow_mag.mean()) if FLOW_ENABLED else 1.0

                zone_scores = []
                scores_flat = []
                for ((x0,y0,x1,y1), cell_id) in grid_boxes:
                    cnn_cell = float(cam_up[y0:y1, x0:x1].mean()) if cam_up is not None else 0.0
                    r_idx = min(GRID_ROWS-1, max(0, y0 // max(1, cell_h)))
                    c_idx = min(GRID_COLS-1, max(0, x0 // max(1, cell_w)))
                    drop_cell = max(0.0, float(delta_zone[r_idx, c_idx]))
                    fmag_cell = float(flow_mag[y0:y1, x0:x1].mean()) if FLOW_ENABLED else 0.0
                    risk = (W_CAM * cnn_cell
                            + W_DROP * (drop_cell / max_heads_now)
                            + W_FLOW * (fmag_cell / (mean_flow + 1e-6)))
                    zone_scores.append(risk)
                    scores_flat.append((cell_id, x0, y0, x1, y1, float(risk)))

                # Track best snapshot frame
                if zone_scores:
                    best_i = int(np.argmax(zone_scores))
                    (bx0,by0,bx1,by1), bcell = grid_boxes[best_i]
                    best_risk = float(zone_scores[best_i])
                    if (current_best is None) or (best_risk > float(current_best["risk_score"])):
                        current_best = {
                            "event_id": event_id,
                            "frame": frame_bgr.copy(),
                            "frame_index": f,
                            "timecode": tc,
                            "zone_id": bcell,
                            "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
                            "risk_score": best_risk
                        }

                # Buffer rows for events_zones.csv
                for (cell_id, x0, y0, x1, y1, risk) in scores_flat:
                    events_z_rows.append({
                        "event_id": event_id, "frame": f, "timecode": tc, "zone_id": cell_id,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1, "risk_score": risk
                    })

            prev_pts, prev_count = head_pts, curr_count

            processed += 1
            if total_steps:
                prog.progress(min(1.0, processed/total_steps))
                status.write(f"Processed {processed}/{total_steps} sampled frames…")
        f += 1

    # close open event at EOF
    if in_event and start_f is not None:
        end_t = (f-1) / fps
        events_rows.append((start_f, f-1, start_t, end_t,
                            sec_to_tc(start_t), sec_to_tc(end_t), end_t-start_t))
        save_current_best()

    cap.release()

    prog.progress(1.0); status.write("Done.")
    df_frames = pd.DataFrame(frames_rows[1:], columns=frames_rows[0])
    df_events = pd.DataFrame(events_rows[1:], columns=events_rows[0])

    # Build events_zones DataFrame for download (if any)
    df_events_zones = pd.DataFrame(events_z_rows) if events_z_rows else pd.DataFrame(
        columns=["event_id","frame","timecode","zone_id","x0","y0","x1","y1","risk_score"]
    )
    st.session_state["video_xai"] = {"events_zones": df_events_zones, "snapshots": snapshots}
    return df_frames, df_events, ""  # snapshot mode => no video

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

        # Persist results so reruns keep the view
        st.session_state["video_results"] = {
            "df_frames": df_frames,
            "df_events": df_events,
            "labeled_path": labeled_path,
        }

        # Render now
        render_results(df_frames, df_events, labeled_path)
