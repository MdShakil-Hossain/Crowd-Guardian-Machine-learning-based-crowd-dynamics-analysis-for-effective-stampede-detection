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
    cell_w, cell_h = W // cols, H // rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1 = W if c == cols - 1 else (c + 1) * cell_w
            y1 = H if r == rows - 1 else (r + 1) * cell_h
            # Store normalized coordinates for consistency
            boxes.append((
                (float(x0) / W, float(y0) / H, float(x1) / W, float(y1) / H),  # Normalized
                f"r{r}c{c}"
            ))
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
        """, height=160, key=f"confetti_{key_seed}")

    # Status banner
    st.markdown('<div class="cg-card">', unsafe_allow_html=True)
    mode_short = st.session_state.get("detection_mode_label", "Hybrid")
    status_cls = "status-ok" if total_events > 0 else "status-safe"
    status_text = f"{mode_short}: {'Stampede detected' if total_events > 0 else 'No stampede detected'}"
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

        left2, right2 = st.columns(2)
        with left2:
            st.subheader("Flow Speed (mean & p95)")
            line_fmean = base.mark_line().encode(
                x=alt.X('frame_index:Q', title='Frame'),
                y=alt.Y('flow_mean:Q', title='px/frame'),
                tooltip=['frame_index','timecode','flow_mean','flow_p95']
            )
            line_fp95  = base.mark_line(strokeDash=[4,4]).encode(
                x='frame_index:Q', y=alt.Y('flow_p95:Q', title=None)
            )
            st.altair_chart((line_fmean + line_fp95).interactive(), use_container_width=True)
        with right2:
            st.subheader("Flow Coherence")
            line_coh = base.mark_line().encode(
                x=alt.X('frame_index:Q', title='Frame'),
                y=alt.Y('flow_coh:Q', title='coherence (0–1)', scale=alt.Scale(domain=[0,1])),
                tooltip=['frame_index','timecode','flow_coh','flow_div_out','flow_fast_frac']
            )
            st.altair_chart(line_coh.interactive(), use_container_width=True)

    # Tables + downloads
    if not df_events.empty:
        st.subheader("Detected intervals")
        st.dataframe(df_events, use_container_width=True)
    else:
        st.info("No stampede intervals found.")

    st.subheader("Per-frame predictions")
    st.dataframe(df_frames.head(1000), use_container_width=True)

    uid_base = os.path.splitext(os.path.basename(labeled_path or "na.mp4"))[0] if labeled_path else "na"
    uid = f"{uid_base}_{key_seed}"
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
        if labeled_path and os.path.exists(labeled_path):
            with open(labeled_path, "rb") as fh:
                st.download_button("⬇️ labeled.mp4", fh.read(),
                                   file_name=os.path.basename(labeled_path),
                                   mime="video/mp4", use_container_width=True, key=f"dl_video_{uid}")
        else:
            st.button("Video unavailable", disabled=True, use_container_width=True, key=f"dl_na_{uid}")

    extra = st.session_state.get("video_xai", {})
    df_events_zones = extra.get("events_zones")
    if isinstance(df_events_zones, pd.DataFrame) and not df_events_zones.empty:
        st.download_button("⬇️ events_zones.csv",
                           df_events_zones.to_csv(index=False).encode("utf-8"),
                           file_name="events_zones.csv", mime="text/csv",
                           use_container_width=True, key=f"dl_events_z_{uid}")

    snapshots = extra.get("snapshots", [])
    if snapshots:
        st.markdown('<h2 class="cg-h2">Event Snapshots</h2>', unsafe_allow_html=True)
        snap_rows = []
        for s in snapshots:
            path = s.get("path") or ""
            # --------- NOTE: NO RISK IN CAPTION (as requested) ----------
            caption = f"Event {s.get('event_id','?')} • frame {s.get('frame_index','?')} • {s.get('timecode','?')} • {s.get('zone_id','?')}"
            if isinstance(path, str) and os.path.exists(path) and os.path.getsize(path) > 0:
                col1, col2 = st.columns([2,1])
                with col1:
                    ok = _show_image_resilient(path, caption)
                    if not ok:
                        st.warning(f"Snapshot could not be displayed (event {s.get('event_id','?')}).")
                with col2:
                    with open(path, "rb") as fh:
                        st.download_button("⬇️ Download snapshot", fh.read(),
                                           file_name=os.path.basename(path),
                                           mime="image/jpeg", use_container_width=True,
                                           key=f"dl_snap_{uid}_{s.get('event_id','x')}_{s.get('frame_index','y')}")
            else:
                st.warning(f"Snapshot file missing for event {s.get('event_id','?')} (path: {path})")
            # keep risk values in CSVs if you still need them downstream; not shown in UI
            snap_rows.append({
                "event_id": s.get("event_id"),
                "frame_index": s.get("frame_index"),
                "timecode": s.get("timecode"),
                "zone_id": s.get("zone_id"),
                "x0": s.get("x0"), "y0": s.get("y0"), "x1": s.get("x1"), "y1": s.get("y1"),
                "risk_score": float(s.get("risk_score", 0.0)),
                "path": path,
            })
        df_snaps = pd.DataFrame(snap_rows)
        st.download_button("⬇️ event_snapshots.csv",
                           df_snaps.to_csv(index=False).encode("utf-8"),
                           file_name="event_snapshots.csv", mime="text/csv",
                           use_container_width=True, key=f"dl_snaps_csv_{uid}")

    st.markdown('<h2 class="cg-h2">Labeled Video Preview</h2>', unsafe_allow_html=True)
    if labeled_path and os.path.exists(labeled_path):
        st.video(labeled_path)
    else:
        st.info("Preview unavailable.")

# Persisted rerender
if "video_results" in st.session_state:
    _res = st.session_state["video_results"]
    render_results(_res["df_frames"], _res["df_events"], _res.get("labeled_path"),
                   key_seed=st.session_state.get("render_nonce"))

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
# Tracking utilities (for head-down)
# =============================================================================
def update_tracks(tracks, detections, radii, max_dist, window_cap):
    prev_pts = [t["pos"] for t in tracks]
    matches, un_prev, un_curr = assign_matches(prev_pts, detections, max_dist)

    for i_prev, j_curr in matches:
        t = tracks[i_prev]
        p = detections[j_curr]; r = float(radii[j_curr])
        t["pos"], t["r"], t["miss"] = p, r, 0
        t["hist"].append((p[0], p[1], r))
        if len(t["hist"]) > window_cap + 2:
            while len(t["hist"]) > window_cap + 2:
                t["hist"].popleft()

    survivors = []
    matched_idx = {i for i,_ in matches}
    for k in range(len(tracks)):
        if k in matched_idx:
            survivors.append(tracks[k]); continue
        t = tracks[k]; t["miss"] += 1
        if t["miss"] <= 2:
            survivors.append(t)

    for j in un_curr:
        p = detections[j]; r = float(radii[j])
        survivors.append({
            "pos": p, "r": r, "miss": 0, "down_streak": 0,
            "hist": deque([(p[0], p[1], r)], maxlen=window_cap+2)
        })
    return survivors

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
    detection_mode="Hybrid (Default)",
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

    # stampede flow baseline state
    eff_fps = (fps / step) if step > 0 else fps
    flow_baseline = _FlowBaseline(eff_fps, base_sec=STAMP_BASELINE_SEC)
    prev_gray_for_metrics = None

    frames_rows = [(
        "frame_index","time_sec","timecode",
        "head_count","delta_vs_prev",
        "prob_cnn","cnn_label","heads_label","heads_torso_label",
        "flow_mean","flow_p95","flow_coh","flow_div_out","flow_fast_frac",
        "stampede_label",
        "final_label"
    )]
    events_rows = [("start_frame","end_frame","start_time_sec","end_time_sec","start_tc","end_tc","duration_sec")]

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base  = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    labeled_path = os.path.join(out_dir, f"{base}_{stamp}_labeled.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(labeled_path, fourcc, fps, (W, H))

    prev_pts, prev_count = [], None
    in_event, start_f, start_t = False, None, None
    min_event_frames = max(1, int(round(min_event_sec * (fps/step))))
    event_id = 0

    # XAI state
    grid_boxes, cell_w, cell_h = make_grid(W, H, rows=GRID_ROWS, cols=GRID_COLS)
    prev_gray_for_flow = None
    events_z_rows = []
    snapshots = []
    current_best = None

    # NEW: track heads over time for head-down
    tracks = []
    window_frames = max(1, int(round(HEAD_DOWN_WINDOW_SEC * (fps/step))))
    streak_frames = max(2, int(round(HEAD_DOWN_MIN_STREAK_SEC * (fps/step))))

    # --------- Custom progress (single gradient bar) ----------
    prog_box = st.empty()
    processed = 0
    total_steps = (N // step + 1) if N > 0 else 0

    def render_prog(frac, processed, total):
        pct = int(round(100.0 * max(0.0, min(1.0, frac))))
        prog_box.markdown(
            f'''
            <div class="cg-prog">
              <div class="cg-prog-label">Processing frames… <b>{pct}%</b> ({processed}/{total})</div>
            <div class="cg-prog-track"><div class="cg-prog-fill" style="width:{pct}%"></div></div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    render_prog(0.0, 0, total_steps)

    # helper to save the best snapshot per event — WITH bounding box
    def save_current_best():
        nonlocal current_best, snapshots
        if not current_best: return
        frame = current_best.pop("frame", None)
        if frame is None:
            current_best = None; return

        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]

        # Scale normalized coordinates back to pixel values
        candidates = current_best.get("candidates", [])
        if candidates:
            # Scale candidate coordinates
            scaled_candidates = []
            for c in candidates:
                scaled_c = {
                    'x0': int(c['x0'] * frame_w),
                    'y0': int(c['y0'] * frame_h),
                    'x1': int(c['x1'] * frame_w),
                    'y1': int(c['y1'] * frame_h)
                }
                scaled_candidates.append(scaled_c)
            # Compute bounding box around scaled candidates
            min_x = min(c['x0'] for c in scaled_candidates)
            min_y = min(c['y0'] for c in scaled_candidates)
            max_x = max(c['x1'] for c in scaled_candidates)
            max_y = max(c['y1'] for c in scaled_candidates)
            # Ensure coordinates are within frame bounds
            min_x = max(0, min(min_x, frame_w - 1))
            min_y = max(0, min(min_y, frame_h - 1))
            max_x = max(0, min(max_x, frame_w - 1))
            max_y = max(0, min(max_y, frame_h - 1))
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        else:
            # Scale zone box coordinates
            x0 = int(current_best["x0"] * frame_w)
            y0 = int(current_best["y0"] * frame_h)
            x1 = int(current_best["x1"] * frame_w)
            y1 = int(current_best["y1"] * frame_h)
            # Ensure coordinates are within frame bounds
            x0 = max(0, min(x0, frame_w - 1))
            y0 = max(0, min(y0, frame_h - 1))
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)  # fallback to zone box

        snap_path = os.path.join(out_dir, f"{base}_{stamp}_event{current_best['event_id']}_snapshot.jpg")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            tmp_path = snap_path + ".tmp"
            with open(tmp_path, "wb") as fh: fh.write(buf.tobytes())
            os.replace(tmp_path, snap_path)
            snapshots.append({
                "event_id": current_best["event_id"],
                "frame_index": current_best["frame_index"],
                "timecode": current_best["timecode"],
                "zone_id": current_best["zone_id"],
                "x0": current_best["x0"],  # Store normalized
                "y0": current_best["y0"],
                "x1": current_best["x1"],
                "y1": current_best["y1"],
                "risk_score": float(current_best["risk_score"]),
                "path": snap_path
            })
        current_best = None

    f = 0
    last_final_label = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if f % step == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # --- STAMPEDE metrics (full-frame Farneback) ---
            flow_mean = flow_p95 = flow_coh = flow_div_out = flow_fast_frac = 0.0
            stampede_label = 0
            mag = np.zeros((H, W), dtype=np.float32)
            if prev_gray_for_metrics is not None and prev_gray_for_metrics.shape == gray.shape:
                try:
                    flow_full = cv2.calcOpticalFlowFarneback(
                        prev_gray_for_metrics, gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    fx = flow_full[..., 0].astype(np.float32)
                    fy = flow_full[..., 1].astype(np.float32)

                    # ---- camera-shake compensation (remove global drift) ----
                    gx, gy = np.median(fx), np.median(fy)
                    fx = fx - gx
                    fy = fy - gy

                    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

                    flow_mean = float(np.mean(mag))
                    flow_p95  = float(np.percentile(mag, 95))

                    w = mag + 1e-6
                    c = float(np.average(np.cos(ang), weights=w))
                    s = float(np.average(np.sin(ang), weights=w))
                    flow_coh = float(np.sqrt(c*c + s*s))

                    du_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
                    dv_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
                    div   = du_dx + dv_dy
                    move_mask = (mag > max(0.5, flow_baseline.mean)).astype(np.float32)
                    flow_div_out = float(np.sum(np.maximum(div, 0.0) * move_mask) / (np.sum(move_mask) + 1e-6))

                    if not flow_baseline.ready:
                        flow_baseline.update(flow_mean)
                    fast_gate = flow_baseline.mean + FLOW_MEAN_Z * flow_baseline.std
                    flow_fast_frac = float(np.mean(mag > fast_gate)) if flow_baseline.ready else 0.0

                    cond_speed   = (flow_baseline.ready and flow_fast_frac >= FLOW_FAST_FRAC_MIN) \
                                   or (flow_p95 >= FLOW_P95_MIN) \
                                   or (flow_mean >= (flow_baseline.mean + FLOW_MEAN_Z*flow_baseline.std))
                    cond_pattern = (flow_coh >= FLOW_COH_MIN) or (flow_div_out >= FLOW_DIV_MIN)
                    stampede_label = 1 if (cond_speed and cond_pattern) else 0
                except cv2.error:
                    pass
            prev_gray_for_metrics = gray.copy()

            # --- Heads detection & tracking for crush/surge (collapse) ---
            head_pts, head_radii = detect_heads_gray(gray, detector)
            curr_count = len(head_pts)
            tracks = update_tracks(tracks, head_pts, head_radii, max_match_dist, window_frames)

            if prev_count is None:
                delta, y_heads = 0, 0
            else:
                delta = prev_count - curr_count
                r = (prev_count - curr_count) / max(1, prev_count)
                y_heads = 1 if (delta >= ABS_DROP or r >= REL_DROP) else 0

            # CNN
            x = preprocess_for_cnn(gray)
            p_cnn = float(model.predict(x, verbose=0)[0][0])
            y_cnn = 1 if p_cnn >= cnn_threshold else 0

            # Mode-aware intermediates
            final_crush = combine_labels(y_heads, y_cnn, COMBINE_RULE)
            final_stamp = combine_labels(stampede_label, y_cnn, COMBINE_RULE)

            # -------------------- XAI computations (strict head+torso) --------------------
            any_head_and_torso_down = False
            ht_cand_count = 0

            if XAI_ENABLED:
                cam_small = gradcam_heatmap(model, x)
                cam_up = upscale_cam(cam_small, W, H) if cam_small is not None else None

                if FLOW_ENABLED:
                    if prev_gray_for_flow is None or prev_gray_for_flow.shape != gray.shape:
                        down_mag = np.zeros((H, W), dtype=np.float32)
                    else:
                        try:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_gray_for_flow, gray, None,
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            # remove global camera motion for torso metric
                            vx = flow[..., 0]
                            vy = flow[..., 1]
                            gx, gy = np.median(vx), np.median(vy)
                            vy = vy - gy
                            down_mag = np.maximum(vy, 0.0).astype(np.float32)
                        except cv2.error:
                            down_mag = np.zeros((H, W), dtype=np.float32)
                    prev_gray_for_flow = gray.copy()
                else:
                    down_mag = np.zeros((H, W), dtype=np.float32)

                scene_mean_flow = float(down_mag.mean()) + 1e-6

                grid_boxes, cell_w, cell_h = make_grid(W, H, rows=GRID_ROWS, cols=GRID_COLS)

                cell_records = []
                for ((x0,y0,x1,y1), cid) in grid_boxes:
                    # Store normalized grid coordinates
                    cell_records.append({
                        "cell_id": cid,
                        "x0": float(x0)/W, "y0": float(y0)/H,
                        "x1": float(x1)/W, "y1": float(y1)/H,
                        "cand": 0, "sum_dy_norm": 0.0, "max_dy_norm": 0.0,
                        "torso_flow_accum": 0.0, "cnn_cell": 0.0, "heads": 0
                    })

                def cell_index_for(x, y):
                    c = min(GRID_COLS-1, max(0, int(x // max(1, cell_w))))
                    r = min(GRID_ROWS-1, max(0, int(y // max(1, cell_h))))
                    return r*GRID_COLS + c

                def neighbors_y_median(x, y, r_head):
                    ys = []
                    for t2 in tracks:
                        (x2, y2) = t2["pos"]; r2 = max(2.0, t2.get("r", r_head))
                        if abs(x2 - x) <= NEIGH_RADIUS_MULT * r_head and abs(y2 - y) <= 3.0 * r_head:
                            ys.append(y2)
                    if len(ys) < 3:
                        ys.extend([y] * (3 - len(ys)))
                    ys.sort()
                    return ys[len(ys)//2]

                # count heads per cell
                for tinfo in tracks:
                    (xh, yh) = tinfo["pos"]
                    ci = cell_index_for(xh, yh)
                    cell_records[ci]["heads"] += 1

                candidates = []
                for tinfo in tracks:
                    (xh, yh) = tinfo["pos"]; rhead = max(2.0, tinfo.get("r", 6.0))
                    if len(tinfo["hist"]) < (window_frames + 1):
                        tinfo["down_streak"] = 0
                        continue

                    y_then = tinfo["hist"][max(0, len(tinfo["hist"])-1-window_frames)][1]
                    dy = (yh - y_then)
                    dy_norm_abs = float(dy) / max(1.0, float(H))

                    y_med = neighbors_y_median(xh, yh, rhead)
                    rel_drop_rad = (yh - y_med) / rhead

                    x0r = int(max(0, xh - 1.2*rhead)); x1r = int(min(W, xh + 1.2*rhead))
                    yh0 = int(max(0, yh - 1.0*rhead)); yh1 = int(min(H, yh + 0.5*rhead))
                    yt0 = int(max(0, yh + 0.5*rhead)); yt1 = int(min(H, yh + 3.0*rhead))
                    torso_flow = float(down_mag[yt0:yt1, x0r:x1r].mean()) if yt1>yt0 else 0.0
                    head_flow  = float(down_mag[yh0:yh1, x0r:x1r].mean()) if yh1>yh0 else 0.0
                    torso_ratio = (torso_flow + 1e-6) / (head_flow + 1e-6)
                    torso_scene = (torso_flow + 1e-6) / scene_mean_flow

                    # STRONGER joint condition: head drop + relative drop + torso motion
                    cond_drop  = (dy_norm_abs >= HEAD_DOWN_MIN_DY_FRAC) or (dy >= HEAD_DOWN_MIN_DY_RAD * rhead)
                    cond_rel   = (rel_drop_rad >= NEIGH_REL_MIN_RAD)
                    cond_torso = (torso_ratio >= TORSO_RATIO_MIN) and (torso_scene >= TORSO_SCENE_MIN)

                    if cond_drop and cond_rel and cond_torso:
                        tinfo["down_streak"] = min(streak_frames+3, tinfo.get("down_streak", 0) + 1)
                    else:
                        tinfo["down_streak"] = max(0, tinfo.get("down_streak", 0) - 1)

                    if tinfo["down_streak"] >= streak_frames:
                        any_head_and_torso_down = True
                        ht_cand_count += 1
                        ci = cell_index_for(xh, yh)
                        rec = cell_records[ci]
                        rec["cand"] += 1
                        rec["sum_dy_norm"] += dy_norm_abs
                        rec["max_dy_norm"] = max(rec["max_dy_norm"], dy_norm_abs)
                        rec["torso_flow_accum"] += (torso_flow + 1e-6) / (scene_mean_flow + 1e-6)
                        # Store normalized candidate coordinates
                        candidates.append({
                            "x0": float(x0r)/W, "y0": float(yh0)/H,
                            "x1": float(x1r)/W, "y1": float(yt1)/H
                        })

                # fill cnn_cell & compute risk
                for rec in cell_records:
                    (x0c,y0c,x1c,y1c) = (rec["x0"], rec["y0"], rec["x1"], rec["y1"])
                    rec["cnn_cell"] = float(cam_up[y0c:y1c, x0c:x1c].mean()) if cam_up is not None else 0.0

                    heads_in_cell = max(1, rec["heads"])
                    down_in_cell  = rec["cand"]
                    frac_down     = float(down_in_cell) / float(heads_in_cell)

                    penalty = 0.0
                    if frac_down > MASS_DROP_PENALTY_START:
                        scale = min(1.0, (frac_down - MASS_DROP_PENALTY_START) / (1.0 - MASS_DROP_PENALTY_START))
                        penalty = MASS_DROP_PENALTY_STRENGTH * scale

                    hd_score = (down_in_cell + rec["sum_dy_norm"] + 0.5*rec["max_dy_norm"])
                    flow_norm = (rec["torso_flow_accum"] / max(1, down_in_cell)) if down_in_cell>0 else 0.0
                    risk_raw = (W_HEADDOWN * hd_score) + (W_FLOW * flow_norm) + (W_CAM * rec["cnn_cell"])
                    rec["risk"] = risk_raw * (1.0 - penalty)

                cands = [i for i,rc in enumerate(cell_records) if rc["cand"] > 0]
                best_i = max(cands, key=lambda i: cell_records[i]["risk"]) if cands else \
                         max(range(len(cell_records)), key=lambda i: cell_records[i]["risk"])
                best = cell_records[best_i]
                bx0,by0,bx1,by1 = best["x0"], best["y0"], best["x1"], best["y1"]
                best_risk = best["risk"]

                if (current_best is None) or (best_risk > float(current_best["risk_score"])) or (current_best and current_best.get("event_id") != event_id):
                    current_best = {
                        "event_id": event_id,
                        "frame": frame_bgr.copy(),
                        "frame_index": f,
                        "timecode": sec_to_tc(f / fps),
                        "zone_id": best["cell_id"],
                        "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,  # Normalized coordinates
                        "risk_score": best_risk,
                        "candidates": candidates.copy()
                    }

                for rc in cell_records:
                    events_z_rows.append({
                        "event_id": event_id, "frame": f, "timecode": sec_to_tc(f / fps), "zone_id": rc["cell_id"],
                        "x0": rc["x0"], "y0": rc["y0"], "x1": rc["x1"], "y1": rc["y1"],
                        "risk_score": rc["risk"],
                        "cand": rc["cand"],
                        "sum_dy_norm": rc["sum_dy_norm"],
                        "max_dy_norm": rc["max_dy_norm"],
                        "heads_in_cell": rc["heads"],
                        "cnn_cell": rc["cnn_cell"],
                    })

            heads_torso_label = 1 if any_head_and_torso_down else 0

            # ---------- FINAL LABEL LOGIC ----------
            motion_ok = (flow_fast_frac >= FLOW_MIN_FAST_FRAC) and (flow_coh >= FLOW_MIN_COH)
            strict_crush = 1 if (ht_cand_count >= HT_MIN_CAND and y_cnn == 1 and motion_ok) else 0

            mode = (detection_mode or "Hybrid").lower()
            if "stampede" in mode:
                final_label = final_stamp
            elif "crush" in mode or "surge" in mode:
                final_label = strict_crush
            else:
                final_label = 1 if (final_stamp == 1 or strict_crush == 1) else 0

            if QUIET_SCENE_SUPPRESS and final_label == 1:
                quiet_scene = (flow_p95 < QUIET_P95_MAX) and (flow_fast_frac < QUIET_FAST_FRAC_MAX) and (flow_coh < QUIET_COH_MAX)
                if quiet_scene and strict_crush == 1 and final_stamp == 0:
                    final_label = 0

            t = f / fps; tc = sec_to_tc(t)
            frames_rows.append((
                f, t, tc,
                curr_count, int(delta),
                p_cnn, y_cnn, y_heads, heads_torso_label,
                flow_mean, flow_p95, flow_coh, flow_div_out, flow_fast_frac,
                stampede_label,
                final_label
            ))

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
                    if current_best: save_current_best()
                in_event, start_f, start_t = False, None, None

            processed += 1
            if total_steps:
                frac = min(1.0, processed/total_steps)
                render_prog(frac, processed, total_steps)
            last_final_label = final_label

        # Draw overlays on every frame
        label_text = "Stampede" if last_final_label == 1 else "Safe"
        label_color = (0, 0, 255) if last_final_label == 1 else (0, 255, 0)
        cv2.putText(frame_bgr, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)

        # AI Monitoring overlay
        cv2.putText(frame_bgr, "AI Monitoring", (W - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Draw scattered dots based on motion
        if flow_baseline.ready:
            fast_gate = flow_baseline.mean + FLOW_MEAN_Z * flow_baseline.std
            rows, cols = np.where(mag > fast_gate)
            if len(rows) > 0:
                num_dots = int(50 + 450 * flow_fast_frac)
                num_dots = min(num_dots, len(rows))
                indices = np.random.choice(len(rows), num_dots, replace=False)
                for i in indices:
                    y_dot, x_dot = rows[i], cols[i]
                    cv2.circle(frame_bgr, (x_dot, y_dot), 2, (0, 255, 255), -1)  # yellow dots

        # Draw links if enabled
        if draw_links:
            for t in tracks:
                if len(t["hist"]) > 1:
                    prev_pos = t["hist"][-2]
                    curr_pos = t["pos"]
                    cv2.line(frame_bgr, (int(prev_pos[0]), int(prev_pos[1])), (int(curr_pos[0]), int(curr_pos[1])), (255, 0, 0), 1)

        out_video.write(frame_bgr)
        f += 1

    if in_event and start_f is not None:
        end_t = (f-1) / fps
        events_rows.append((start_f, f-1, start_t, end_t,
                            sec_to_tc(start_t), sec_to_tc(end_t), end_t-start_t))
        if current_best: save_current_best()

    cap.release()
    out_video.release()

    # Transcode to h264
    h264_path = labeled_path.replace(".mp4", "_h264.mp4")
    labeled_path, ok, err = transcode_to_h264(labeled_path, h264_path, fps)
    if not ok:
        st.warning(f"Transcoding failed: {err}")

    render_prog(1.0, total_steps, total_steps)
    df_frames = pd.DataFrame(frames_rows[1:], columns=frames_rows[0])
    df_events = pd.DataFrame(events_rows[1:], columns=events_rows[0])

    df_events_zones = pd.DataFrame(events_z_rows) if events_z_rows else pd.DataFrame(
        columns=["event_id","frame","timecode","zone_id","x0","y0","x1","y1",
                 "risk_score","cand","sum_dy_norm","max_dy_norm","heads_in_cell","cnn_cell"]
    )
    st.session_state["video_xai"] = {"events_zones": df_events_zones, "snapshots": snapshots}
    return df_frames, df_events, labeled_path

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
            df_frames, df_events, labeled_path = analyze_video(
                tmp.name, model, detection_mode=detection_mode
            )
        st.session_state["video_results"] = {
            "df_frames": df_frames,
            "df_events": df_events,
            "labeled_path": labeled_path,
        }
        st.session_state["detection_mode_label"] = detection_mode
        st.session_state["render_nonce"] = str(int(time.time() * 1e6))
        render_results(df_frames, df_events, labeled_path, key_seed=st.session_state["render_nonce"])
