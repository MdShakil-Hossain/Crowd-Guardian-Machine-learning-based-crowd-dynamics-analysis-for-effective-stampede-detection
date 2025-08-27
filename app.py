import os
import cv2
import time
import tempfile
import subprocess
import shutil
import base64
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.optimize import linear_sum_assignment
from collections import deque

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

# ---- one-time session init for stable keys / components ----
if "render_nonce" not in st.session_state:
    import time as _t
    st.session_state["render_nonce"] = str(int(_t.time() * 1e6))
st.session_state.setdefault("video_xai", {"events_zones": pd.DataFrame(), "snapshots": []})

# ---------- Inference defaults (no UI knobs) ----------
CNN_THRESHOLD = 0.75   # stricter
ABS_DROP      = 2
REL_DROP      = 0.20
MIN_EVENT_SEC = 1.5
COMBINE_RULE  = "and"
MIN_FRAC      = 0.00005
MAX_FRAC      = 0.0020
MIN_CIRC      = 0.20
MIN_INER      = 0.10
DRAW_LINKS    = True
TARGET_FPS    = None  # None = use every frame

# ---------- Stampede (running-panic) thresholds ----------
STAMP_BASELINE_SEC   = 5.0
FLOW_MEAN_Z          = 1.0
FLOW_P95_MIN         = 3.0
FLOW_FAST_FRAC_MIN   = 0.25
FLOW_COH_MIN         = 0.55
FLOW_DIV_MIN         = 0.04

# ---------- XAI / Snapshot settings ----------
XAI_ENABLED   = True
GRID_ROWS     = 6
GRID_COLS     = 6

# Head+Torso collapse detector (primary signal)
HEAD_DOWN_WINDOW_SEC      = 0.8
HEAD_DOWN_MIN_DY_FRAC     = 0.06
HEAD_DOWN_MIN_DY_RAD      = 1.20
HEAD_DOWN_MIN_STREAK_SEC  = 0.80
NEIGH_RADIUS_MULT         = 3.0
NEIGH_REL_MIN_RAD         = 1.20
MASS_DROP_PENALTY_START   = 0.88
MASS_DROP_PENALTY_STRENGTH= 0.25

# Torso motion requirements
TORSO_RATIO_MIN = 1.50
TORSO_SCENE_MIN = 1.40

# Require at least N people showing head+torso down together
HT_MIN_CAND        = 4
FLOW_MIN_FAST_FRAC = 0.18
FLOW_MIN_COH       = 0.60

# Quiet-scene suppression
QUIET_SCENE_SUPPRESS = True
QUIET_P95_MAX        = 1.20
QUIET_FAST_FRAC_MAX  = 0.08
QUIET_COH_MAX        = 0.45

# risk weights
W_HEADDOWN, W_FLOW, W_CAM = 0.60, 0.22, 0.18

FLOW_ENABLED  = True
SNAPSHOT_ONLY = False

# ---------- Snapshot overlay control ----------
SHOW_ZONE_BOX = False
STRICT_REQUIRE_HEAD_AND_TORSO = True

# ---------- Model location ----------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "models"; CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/zswpw1ucbj7bkkkykc8oc/deep_cnn_stampede.h5"
    "?rlkey=e863b9skyvpyn0dn4gbwxd71s&st=uvgqjq7q&dl=1"
)
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

# ---------- Load logo (base64) ----------
def _load_logo_b64():
    try:
        logo_path = APP_DIR / "assets" / "Crowd_Guardian_EWU_logo2.png"
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""
LOGO_B64 = _load_logo_b64()

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
        --logo-size: 120px;           /* <— change this to resize the hero logo */
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

      /* ---------------- Hero ---------------- */
      .cg-hero {
        position: relative;
        margin-top: 8px; padding: 30px 32px 32px 128px;   /* left padding reserves space for logo */
        border-radius: 20px; border: 1px solid var(--muted2);
        background:
          radial-gradient(1200px 420px at 0% -10%, rgba(59,130,246,.16), transparent),
          linear-gradient(180deg, rgba(17,24,39,.55), rgba(2,6,23,.45));
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,.25);
      }
      .cg-logo{
        position:absolute; left:26px; top:50%; transform: translateY(-50%);
        width: var(--logo-size); height: var(--logo-size);
        border-radius: 14px; box-shadow: 0 6px 24px rgba(0,0,0,.35), 0 0 0 3px rgba(255,255,255,.05) inset;
        object-fit: cover; background: rgba(255,255,255,.06);
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

      /* --------- Custom progress bar (single gradient bar) ---------- */
      .cg-prog-label{margin:6px 4px 6px; font-weight:700; letter-spacing:.2px;}
      .cg-prog-track{height:12px; border-radius:999px; background:rgba(255,255,255,.08);
                      box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);}
      .cg-prog-fill{height:12px; border-radius:999px; background:linear-gradient(90deg, var(--accent), var(--accent2));}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- safe wrapper for components.html ----------
def _safe_html(html: str, *, height: int, key: str, scrolling: bool=False, width: int=0):
    try:
        components.html(html, height=height, key=key, scrolling=scrolling, width=width)
    except TypeError:
        pass
    except Exception:
        pass

# =============================================================================
# Background Particles
# =============================================================================
_safe_html("""
<canvas id="cg-bg"></canvas>
<style>#cg-bg{position:fixed; inset:0; z-index:-2; background:transparent;}</style>
<script>
  const c = document.getElementById('cg-bg'), ctx = c.getContext('2d');
  function resize(){ c.width = innerWidth; c.height = innerHeight; }
  addEventListener('resize', resize); resize();
  const N = 120;
  const P = Array.from({length:N}, () => ({
     x: Math.random()*c.width, y: Math.random()*c.height,
     vx: -0.25 + Math.random()*0.5, vy: -0.25 + Math.random()*0.5,
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
""", height=48, key="bg_particles_iframe", scrolling=False, width=0)

# =============================================================================
# Sidebar — PROJECT DETAILS + Detection Mode (UNCHANGED UI)
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

    detection_mode = st.selectbox(
        "Detection Mode",
        ["Hybrid (Default)", "Stampede (Running Panic)", "Crush/Surge (Compression)"],
        index=0,
        help="Hybrid triggers on either running-panic (flow) or crush (head-drop) cues."
    )

# =============================================================================
# Hero (logo left, text centered)
# =============================================================================
if LOGO_B64:
    logo_tag = f'<img class="cg-logo" alt="Crowd Guardian Logo" src="data:image/png;base64,{LOGO_B64}"/>'
else:
    logo_tag = ''

st.markdown(
    f'''
    <div class="cg-hero">
      {logo_tag}
      <div class="cg-hero-center">
        <div class="cg-title">Crowd Guardian</div>
        <div class="cg-subtle">Machine learning based crowd dynamics analysis for effective stampede detection</div>
      </div>
    </div>
    ''',
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
    return tf.keras.models.load_model(local, compile=False)

def sec_to_tc(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def build_blob_detector(frame_w, frame_h, min_frac=MIN_FRAC, max_frac=MAX_FRAC,
                        min_circ=MIN_CIRC, min_iner=MIN_INER, thresh_step=10):
    """Consistent param name: min_iner (fixes NameError)."""
    area = float(frame_w * frame_h)
    minArea = max(5.0, min_frac * area); maxArea = max(minArea + 1.0, max_frac * area)
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold, p.maxThreshold, p.thresholdStep = 10, 220, thresh_step
    p.filterByArea, p.minArea, p.maxArea = True, minArea, maxArea
    p.filterByCircularity, p.minCircularity = True, min_circ
    p.filterByInertia, p.minInertiaRatio = True, min_iner
    p.filterByConvexity = p.filterByColor = False
    return (cv2.SimpleBlobDetector(p) if int(cv2.__version__.split('.')[0]) < 3
            else cv2.SimpleBlobDetector_create(p))

def detect_heads_gray(gray, detector):
    kps = detector.detect(cv2.GaussianBlur(gray, (5,5), 0))
    pts = [(float(k.pt[0]), float(k.pt[1])) for k in kps]
    radii = [max(2.0, 0.5*float(k.size)) for k in kps]
    return pts, radii

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

# =============== XAI: Grad-CAM + zone grid helpers ===========================
def gradcam_heatmap(model, x_100x100x1, conv_layer_name=None):
    import tensorflow as tf

    def _select_score_vector(preds_any):
        t = preds_any
        if isinstance(t, dict): t = t[sorted(t.keys())[0]]
        if isinstance(t, (list, tuple)): t = t[0]
        t = tf.convert_to_tensor(t)
        r = t.shape.rank
        if r is None: return tf.reshape(t, (tf.shape(t)[0], -1))[:, -1]
        if r == 1:  return t
        if r == 2:
            c = t.shape[-1]
            return tf.squeeze(t, axis=-1) if c == 1 else t[:, -1]
        return tf.reshape(t, (tf.shape(t)[0], -1))[:, -1]

    target_layer = None
    if conv_layer_name:
        try:
            L = model.get_layer(conv_layer_name)
            if len(L.output.shape) == 4: target_layer = L
        except Exception:
            pass
    if target_layer is None:
        for L in reversed(model.layers):
            try:
                if len(L.output.shape) == 4:
                    target_layer = L; break
            except Exception:
                continue
    if target_layer is None: return None

    try:
        grad_model = tf.keras.Model(inputs=model.input, outputs=[target_layer.output, model.output])
    except Exception:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.outputs])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_100x100x1, training=False)
        score_vec = _select_score_vector(preds)

    grads = tape.gradient(score_vec, conv_out)
    if grads is None: return None
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))
    cam = cam[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def upscale_cam(cam_small, W, H):
    if cam_small is None: return None
    return cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_CUBIC)

def make_grid(W, H, rows=6, cols=6):
    aspect = W / H
    if aspect < 1:  # Portrait, adjust columns
        cols = int(rows * aspect)
        cols = max(2, cols)  # Ensure minimum grid resolution
    elif aspect > 1.5:  # Wide landscape, adjust rows
        rows = int(cols / aspect)
        rows = max(2, rows)  # Ensure minimum grid resolution
    cell_w, cell_h = W // max(1, cols), H // max(1, rows)
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1 = W if c == cols - 1 else (c + 1) * cell_w
            y1 = H if r == rows - 1 else (r + 1) * cell_h
            boxes.append(((x0, y0, x1, y1), f"r{r}c{c}"))
    return boxes, cell_w, cell_h

def _show_image_resilient(path: str, caption: str) -> bool:
    try:
        data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if data is None: data = cv2.imread(path, cv2.IMREAD_COLOR)
        if data is None: return False
        rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        try:
            st.image(rgb, caption=caption, use_container_width=True)
            return True
        except Exception:
            ok, buf = cv2.imencode(".png", data)
            if not ok: return False
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            st.markdown(
                f'<figure style="margin:6px 0 18px 0">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'
                f'<figcaption style="text-align:center;opacity:.8">{caption}</figcaption>'
                f'</figure>',
                unsafe_allow_html=True
            )
            return True
    except Exception:
        return False

# -------- Flow baseline helper (stampede metrics) --------
class _FlowBaseline:
    def __init__(self, eff_fps, base_sec=STAMP_BASELINE_SEC):
        n = max(1, int(round(base_sec * max(1.0, eff_fps))))
        self.vals = deque(maxlen=n)
    def update(self, v: float):
        self.vals.append(float(v))
    @property
    def ready(self) -> bool:
        return len(self.vals) >= max(5, self.vals.maxlen // 2)
    @property
    def mean(self) -> float:
        return float(np.mean(self.vals)) if self.vals else 0.0
    @property
    def std(self) -> float:
        s = float(np.std(self.vals)) if self.vals else 0.0
        return max(s, 1e-6)

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
# Re-render persisted results
# =============================================================================
def render_results(df_frames, df_events, labeled_path, key_seed=None):
    key_seed = key_seed or st.session_state.get("render_nonce", "0")

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
        _safe_html("""
        <canvas id="c"></canvas>
        <style>
          #c{position:relative;width:100%;height:140px;display:block;border-radius:12px;margin:6px 0 4px 0;
             background:linear-gradient(90deg, rgba(16,185,129,.18), rgba(239,68,68,.18));}
        </style>
        <script>
          const canvas = document.getElementById('c');
          const ctx = canvas.getContext('2d');
          function resize(){canvas.width=canvas.clientWidth; canvas.height=140;}
