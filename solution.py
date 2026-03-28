"""
Sentio Mind · Project 6 · Micro-Expression & Emotion Transition Timeline

Run:
  venv_p6\\Scripts\\python solution.py

Outputs:
  - emotion_timeline.html
  - emotion_timeline_output.json
  - demo.mp4 (best-effort)
"""

import json
import math
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_DISPLAY_NAME = "video_sample_1.mov"

VIDEO_PATH = Path("video_sample_1.mov")
if not VIDEO_PATH.exists():
    video_dir = Path("Video_1")
    if video_dir.exists():
        movs = sorted(video_dir.glob("*.mov"))
        if movs:
            VIDEO_PATH = movs[0]

REPORT_HTML_OUT = Path("emotion_timeline.html")
OUTPUT_JSON = Path("emotion_timeline_output.json")
DEMO_OUT = Path("demo.mp4")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANALYSIS_FPS = 8  # frames per second to analyse
MICRO_MAX_SEC = 0.5  # micro-expression must last less than this
MICRO_MIN_PROB = 0.40  # peak probability to qualify (0-1 scale in the README)
NEUTRAL_MIN = 0.50  # min neutral probability to count as "neutral before/after" (0-1 scale)

# Direct ONNX download. The ONNX Model Zoo migrated from GitHub to Hugging Face
# (GitHub LFS links are no longer served as of mid-2025).
# This points to the emotion-ferplus-8 model on the official HF mirror.
FERPLUS_ONNX_URL = (
    "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/"
    "emotion-ferplus-8.onnx"
)

# emotion-ferplus-8 output label order (8 classes; we drop contempt at index 7)
FERPLUS_LABEL_ORDER = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS = {
    "happy": "#22c55e",
    "neutral": "#94a3b8",
    "sad": "#3b82f6",
    "angry": "#ef4444",
    "fear": "#a855f7",
    "surprise": "#eab308",
    "disgust": "#f97316",
}


def _normalize_to_100(prob_dict: dict) -> dict:
    """Scale/round emotion probabilities so they sum to 100."""
    values = [float(prob_dict[e]) for e in EMOTIONS]
    total = sum(values)
    if total <= 0:
        return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

    scaled = [v / total * 100.0 for v in values]
    rounded = [int(round(v)) for v in scaled]
    diff = 100 - sum(rounded)
    if diff != 0:
        # Adjust rounding error on the max-prob emotion.
        max_idx = int(np.argmax(rounded))
        rounded[max_idx] = max(0, rounded[max_idx] + diff)
    return {e: rounded[i] for i, e in enumerate(EMOTIONS)}


def _download_ferplus_if_missing() -> Path | None:
    onnx = MODELS_DIR / "emotion_ferplus.onnx"
    if onnx.exists():
        return onnx
    try:
        print(f"Downloading FERPlus ONNX to {onnx} ...")
        req = urllib.request.Request(
            FERPLUS_ONNX_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(onnx, "wb") as f:
                shutil.copyfileobj(resp, f)
        return onnx
    except Exception as exc:
        print(f"WARNING: failed to download FERPlus ONNX: {exc}")
        return None


# ---------------------------------------------------------------------------
# EMOTION MODEL LOADER
# ---------------------------------------------------------------------------

def load_emotion_model():
    """
    Try DeepFace first.
    If not installed, try loading models/emotion_ferplus.onnx with OpenCV DNN.
    If neither available, return (None, 'fallback').
    Return: (model_object, model_type_string)
    """
    try:
        from deepface import DeepFace  # optional

        return DeepFace, "deepface"
    except Exception:
        pass

    onnx = _download_ferplus_if_missing()
    if onnx and onnx.exists():
        net = cv2.dnn.readNetFromONNX(str(onnx))
        return net, "ferplus"

    print("WARNING: no emotion model found. Using neutral baseline.")
    return None, "fallback"


# ---------------------------------------------------------------------------
# EMOTION ANALYSIS — ONE FACE CROP
# ---------------------------------------------------------------------------

def analyse_emotion(face_crop: np.ndarray, model, model_type: str) -> dict:
    """
    Return { emotion_name: probability_0_to_100, ... } for all 7 EMOTIONS.
    Probabilities must sum to 100 (round as needed).
    """
    if model_type == "deepface":
        try:
            res = model.analyze(
                face_crop, actions=["emotion"], enforce_detection=False, silent=True
            )
            # DeepFace can return list[dict] or dict.
            if isinstance(res, list) and res:
                res = res[0]
            emotions = res.get("emotion", {}) if isinstance(res, dict) else {}

            # Values are typically in 0-100 already; but handle both scales.
            prob_dict = {}
            for e in EMOTIONS:
                val = float(emotions.get(e, 0.0))
                prob_dict[e] = val

            max_val = max(prob_dict.values()) if prob_dict else 0.0
            # If values look like 0-1 probabilities, scale to 0-100.
            if max_val <= 1.5:
                prob_dict = {k: v * 100.0 for k, v in prob_dict.items()}

            return _normalize_to_100(prob_dict)
        except Exception:
            return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

    if model_type == "ferplus":
        try:
            if face_crop.ndim == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop
            # emotion-ferplus-8 expects 1x1x64x64 float input
            gray = cv2.resize(gray, (64, 64))

            blob = cv2.dnn.blobFromImage(
                gray, scalefactor=1.0, size=(64, 64), mean=(0,), swapRB=False
            )
            model.setInput(blob)
            preds = model.forward()
            flat = np.array(preds).reshape(-1)

            # Model output: 8 classes (neutral, happy, surprise, sad, angry, disgust,
            # fear, contempt). We use the first 7 (drop contempt).
            n = min(7, flat.size)
            scores = flat[:n].astype(np.float64)
            if n < 7:
                scores = np.pad(scores, (0, 7 - n))

            # Map model label order -> EMOTIONS dict
            raw = {FERPLUS_LABEL_ORDER[i]: float(scores[i]) for i in range(7)}
            prob_dict = {e: raw.get(e, 0.0) for e in EMOTIONS}

            # Handle 0-1 vs raw logit scores.
            if max(prob_dict.values()) <= 1.5:
                prob_dict = {k: v * 100.0 for k, v in prob_dict.items()}

            return _normalize_to_100(prob_dict)
        except Exception:
            return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

    # Fallback
    return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    """
    Find the largest face in the frame.
    Return (face_crop_ndarray, (x, y, w, h)) or (None, None) if no face found.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, None

    # Speed: run face detection on a smaller image, then scale bbox back.
    # This keeps the exact function signature intact but avoids expensive detection
    # on full-resolution frames.
    h, w = gray.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > 640:
        scale = 640.0 / float(max_side)
    if scale < 1.0:
        gray_small = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        gray_small = gray

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not Path(cascade_path).exists():
        return None, None

    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if faces is None or len(faces) == 0:
        return None, None

    # Pick largest face by area.
    x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))

    # Scale bbox back to original frame coordinates.
    if scale < 1.0 and scale > 0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    # Add a small margin so emotion model sees complete features.
    margin = int(0.15 * max(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(frame.shape[1], x + w + margin)
    y1 = min(frame.shape[0], y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return None, None

    crop = frame[y0:y1, x0:x1]
    return crop, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    """
    Return list of micro-expression dicts matching emotion_timeline.json schema.
    """
    micro = []
    if len(frame_series) < 3:
        return micro

    micro_max_frames = int(ANALYSIS_FPS * MICRO_MAX_SEC)  # e.g. 4
    peak_thresh = MICRO_MIN_PROB * 100.0
    neutral_thresh = NEUTRAL_MIN * 100.0

    next_id = 1
    i = 1
    while i < len(frame_series) - 1:
        prev_neutral = frame_series[i - 1]["emotions"]["neutral"] >= neutral_thresh
        if not prev_neutral:
            i += 1
            continue

        # Pick the strongest non-neutral emotion at i.
        non_neutral = [
            e
            for e in EMOTIONS
            if e != "neutral" and frame_series[i]["emotions"].get(e, 0) >= peak_thresh
        ]
        if not non_neutral:
            i += 1
            continue

        emotion = max(
            non_neutral, key=lambda e: float(frame_series[i]["emotions"].get(e, 0))
        )

        # Count consecutive frames where that emotion stays above threshold.
        j = i
        peak = 0.0
        while j < len(frame_series) and frame_series[j]["emotions"].get(emotion, 0) >= peak_thresh:
            peak = max(peak, float(frame_series[j]["emotions"].get(emotion, 0)))
            j += 1

        # Now find when neutral returns within MICRO_MAX_SEC (including tail frames).
        start_t = float(frame_series[i]["t"])
        max_end_t = start_t + MICRO_MAX_SEC

        k = i
        neutral_return_idx = None
        while k < len(frame_series) and float(frame_series[k]["t"]) <= max_end_t:
            if frame_series[k]["emotions"]["neutral"] >= neutral_thresh:
                neutral_return_idx = k
                break
            k += 1

        if neutral_return_idx is None:
            i = j + 1
            continue

        duration_frames = neutral_return_idx - i + 1
        duration_sec = duration_frames / float(ANALYSIS_FPS)

        # Definition: lasts less than 0.5 seconds (fewer than ANALYSIS_FPS*0.5 frames).
        if duration_sec >= MICRO_MAX_SEC or duration_frames >= micro_max_frames:
            i = neutral_return_idx + 1
            continue

        micro.append(
            {
                "id": next_id,
                "timestamp_sec": float(frame_series[i]["t"]),
                "duration_sec": float(round(duration_sec, 3)),
                "emotion": emotion,
                "peak_probability": float(round(peak, 3)),
                "followed_by": "neutral",
                "is_suppressed": True,
            }
        )
        next_id += 1
        i = neutral_return_idx + 1

    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    """
    A transition occurs when the dominant emotion changes and the new one
    holds for at least 0.5 seconds.
    """
    transitions = []
    if len(frame_series) < 2:
        return transitions

    def dominant_at(idx: int) -> str:
        emo = frame_series[idx]["emotions"]
        return max(emo, key=lambda e: emo.get(e, 0))

    dom_prev = dominant_at(0)
    i = 1
    while i < len(frame_series):
        dom_curr = dominant_at(i)
        if dom_curr == dom_prev:
            i += 1
            continue

        # Check how long dom_curr stays dominant.
        j = i
        while j < len(frame_series) and dominant_at(j) == dom_curr:
            j += 1

        frames_count = j - i
        duration_sec = frames_count / float(ANALYSIS_FPS)
        if duration_sec >= MICRO_MAX_SEC:
            transitions.append(
                {
                    "from_emotion": dom_prev,
                    "to_emotion": dom_curr,
                    "timestamp_sec": float(frame_series[i]["t"]),
                    "transition_duration_sec": float(round(duration_sec, 3)),
                }
            )
        dom_prev = dom_curr
        i = j

    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    """
    suppression_score = (micro_expression_count / total_expression_events) * 100
    total_expression_events = frames where any non-neutral emotion > 0.35
    """
    non_neutral_frames = 0
    for fs in frame_series:
        emo_probs = fs["emotions"]
        non_neutral_max = max(
            float(emo_probs.get(e, 0)) for e in EMOTIONS if e != "neutral"
        )
        if non_neutral_max > 35.0:  # 0.35 in README, scaled to 0-100
            non_neutral_frames += 1

    if non_neutral_frames <= 0:
        return 0

    score = (len(micro_expressions) / float(non_neutral_frames)) * 100.0
    return int(round(max(0.0, min(100.0, score))))


def compute_emotional_range(frame_series: list) -> int:
    """
    distinct = count of EMOTIONS that appeared with prob > 30 at least once
    range_score = min(100, (distinct / 7) * 100 + std_of_all_probs * 2)
    """
    appeared = set()
    all_probs = []
    for fs in frame_series:
        emo_probs = fs["emotions"]
        for e in EMOTIONS:
            p = float(emo_probs.get(e, 0))
            all_probs.append(p)
            if p > 30.0:
                appeared.add(e)

    distinct = len(appeared)
    std_val = float(np.std(all_probs)) if all_probs else 0.0
    range_score = (distinct / 7.0) * 100.0 + std_val * 2.0
    range_score = min(100.0, max(0.0, range_score))
    return int(round(range_score))


# ---------------------------------------------------------------------------
# HTML RIVER CHART (OFFLINE)
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(
    frame_series: list,
    micro_expressions: list,
    transitions: list,
    stats: dict,
    output_path: Path,
):
    emotion_order = EMOTIONS
    emotion_colors = EMOTION_COLORS

    # Compact JSON payloads to keep the HTML size reasonable.
    js_frame_series = json.dumps(frame_series, ensure_ascii=False, separators=(",", ":"))
    js_micro = json.dumps(micro_expressions, ensure_ascii=False, separators=(",", ":"))
    js_transitions = json.dumps(transitions, ensure_ascii=False, separators=(",", ":"))
    js_stats = json.dumps(stats, ensure_ascii=False, separators=(",", ":"))
    js_emotion_order = json.dumps(emotion_order, ensure_ascii=False)
    js_emotion_colors = json.dumps(emotion_colors, ensure_ascii=False, separators=(",", ":"))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Emotion Timeline</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #f1f5f9; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; font-weight: 600; }}
    .muted {{ color: #64748b; }}
    #tooltip {{
      position: fixed; z-index: 9999; display: none;
      background: rgba(15, 23, 42, 0.95); color: #fff;
      padding: 8px 10px; border-radius: 10px; font-size: 13px;
      pointer-events: none; max-width: 260px;
    }}
    .legend span {{
      display: inline-block; width: 12px; height: 12px; margin-right: 6px;
      border-radius: 3px; vertical-align: middle;
    }}
  </style>
</head>
<body>
  <h2 style="margin-top:0;">Micro-Expression & Emotion Transition Timeline</h2>
  <div class="grid">
    <div class="card">
      <div class="legend muted" id="legend"></div>
      <div style="margin-top:10px;">
        <svg id="riverSvg" width="100%" viewBox="0 0 1000 420" role="img" aria-label="Emotion river chart"></svg>
      </div>
      <div class="muted" style="margin-top:10px; font-size: 13px;">
        Dashed vertical lines mark micro-expression start timestamps.
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0; font-size: 16px;">Summary</h3>
      <div><b>Dominant emotion:</b> <span id="dominantEmotion"></span></div>
      <div><b>Suppression score:</b> <span id="suppressionScore"></span></div>
      <div><b>Emotional range score:</b> <span id="emotionalRangeScore"></span></div>
      <hr style="border:0; border-top:1px solid #eef2f7; margin: 12px 0;"/>
      <h4 style="margin:0 0 8px; font-size: 14px;">Time in State (%)</h4>
      <div id="timeInState"></div>
    </div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3 style="margin-top:0; font-size: 16px;">Transitions</h3>
    <div id="transitions"></div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3 style="margin-top:0; font-size: 16px;">Micro-Expressions</h3>
    <div id="microList" class="muted" style="font-size: 13px;"></div>
  </div>

  <div id="tooltip"></div>

  <script>
    const frameSeries = {js_frame_series};
    const microExpressions = {js_micro};
    const transitions = {js_transitions};
    const stats = {js_stats};
    const EMOTIONS = {js_emotion_order};
    const EMOTION_COLORS = {js_emotion_colors};

    const svg = document.getElementById('riverSvg');
    const tooltip = document.getElementById('tooltip');

    const W = 1000, H = 420;
    const margin = {{ l: 70, r: 20, t: 22, b: 54 }};
    const innerW = W - margin.l - margin.r;
    const innerH = H - margin.t - margin.b;

    const t0 = frameSeries.length ? frameSeries[0].t : 0;
    const t1 = frameSeries.length ? frameSeries[frameSeries.length - 1].t : 1;
    const denom = (t1 - t0) || 1;

    function xScale(t) {{
      return margin.l + ((t - t0) / denom) * innerW;
    }}
    function yScale(v) {{
      // v is 0..100
      return margin.t + (1 - (v / 100.0)) * innerH;
    }}

    function elSVG(name, attrs) {{
      const e = document.createElementNS('http://www.w3.org/2000/svg', name);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }}

    function setTooltip(html) {{
      tooltip.innerHTML = html;
    }}

    function showTooltip(evt, html) {{
      setTooltip(html);
      tooltip.style.display = 'block';
      tooltip.style.left = (evt.clientX + 12) + 'px';
      tooltip.style.top = (evt.clientY + 12) + 'px';
    }}

    function moveTooltip(evt) {{
      if (tooltip.style.display === 'block') {{
        tooltip.style.left = (evt.clientX + 12) + 'px';
        tooltip.style.top = (evt.clientY + 12) + 'px';
      }}
    }}

    function hideTooltip() {{
      tooltip.style.display = 'none';
    }}

    // Legend
    const legend = document.getElementById('legend');
    legend.innerHTML = EMOTIONS.map(e => (
      `<div style="display:inline-block; margin-right:14px;"><span style="background:${{EMOTION_COLORS[e]}}"></span>${{e}}</div>`
    )).join('');

    // Axes + grid
    const grid = elSVG('g', {{ }});
    const yTicks = [0, 25, 50, 75, 100];
    yTicks.forEach(v => {{
      const y = yScale(v);
      grid.appendChild(elSVG('line', {{ x1: margin.l, y1: y, x2: W - margin.r, y2: y, stroke: '#f1f5f9', 'stroke-width': 1 }}));
      const txt = elSVG('text', {{ x: margin.l - 10, y: y + 4, fill: '#64748b', 'font-size': 12, 'text-anchor': 'end' }});
      txt.textContent = v;
      grid.appendChild(txt);
    }});
    svg.appendChild(grid);

    const N = frameSeries.length;
    const xs = new Array(N);
    for (let i = 0; i < N; i++) xs[i] = xScale(frameSeries[i].t);

    // Prepare cumulative sums for each band.
    const probsByEmotion = EMOTIONS.map(e => frameSeries.map(fs => Number(fs.emotions[e] || 0)));
    const cumLow = [];
    const cumHigh = [];
    let cum = new Array(N).fill(0);

    for (let ei = 0; ei < EMOTIONS.length; ei++) {{
      const low = cum.slice();
      const high = new Array(N);
      for (let i = 0; i < N; i++) {{
        high[i] = low[i] + probsByEmotion[ei][i];
      }}
      cumLow.push(low);
      cumHigh.push(high);
      cum = high;
    }}

    function buildAreaPath(lowArr, highArr) {{
      if (!lowArr.length) return '';
      let d = `M ${{xs[0]}} ${{yScale(highArr[0])}}`;
      for (let i = 1; i < lowArr.length; i++) d += ` L ${{xs[i]}} ${{yScale(highArr[i])}}`;
      for (let i = lowArr.length - 1; i >= 0; i--) d += ` L ${{xs[i]}} ${{yScale(lowArr[i])}}`;
      d += ' Z';
      return d;
    }}

    // Draw stacked bands from bottom to top.
    // To preserve the exact ordering, draw earlier emotions first with opacity.
    for (let ei = 0; ei < EMOTIONS.length; ei++) {{
      const e = EMOTIONS[ei];
      const path = elSVG('path', {{
        d: buildAreaPath(cumLow[ei], cumHigh[ei]),
        fill: EMOTION_COLORS[e],
        opacity: 0.65,
        stroke: EMOTION_COLORS[e],
        'stroke-width': 0.5,
      }});
      svg.appendChild(path);
    }}

    // X-axis ticks (based on duration)
    const tSpan = t1 - t0;
    const approxTicks = 6;
    const step = Math.max(0.5, Math.round((tSpan / approxTicks) * 2) / 2);
    const firstTick = Math.ceil(t0 / step) * step;
    for (let tt = firstTick; tt <= t1 + 1e-9; tt += step) {{
      const x = xScale(tt);
      svg.appendChild(elSVG('line', {{ x1: x, y1: margin.t + innerH, x2: x, y2: margin.t + innerH + 6, stroke: '#cbd5e1', 'stroke-width': 1 }}));
      const txt = elSVG('text', {{ x: x, y: margin.t + innerH + 26, fill: '#64748b', 'font-size': 12, 'text-anchor': 'middle' }});
      txt.textContent = tt.toFixed(1);
      svg.appendChild(txt);
    }}

    // Micro-expression marker lines
    microExpressions.forEach(me => {{
      const x = xScale(me.timestamp_sec);
      const line = elSVG('line', {{
        x1: x, y1: margin.t, x2: x, y2: margin.t + innerH,
        stroke: '#0ea5e9',
        'stroke-width': 2,
        'stroke-dasharray': '6,6'
      }});
      line.style.cursor = 'help';
      const html = `
        <div style="font-weight:600; margin-bottom:4px;">Micro-expression</div>
        <div><b>Emotion:</b> ${{me.emotion}}</div>
        <div><b>Peak probability:</b> ${{me.peak_probability}}</div>
        <div><b>Duration:</b> ${{me.duration_sec}}s</div>
        <div class="muted"><b>t:</b> ${{Number(me.timestamp_sec).toFixed(2)}}s</div>
      `;
      line.addEventListener('mouseenter', (evt) => showTooltip(evt, html));
      line.addEventListener('mousemove', moveTooltip);
      line.addEventListener('mouseleave', hideTooltip);
      svg.appendChild(line);
    }});

    // Summary values
    document.getElementById('dominantEmotion').textContent = stats.dominant_emotion || 'neutral';
    document.getElementById('suppressionScore').textContent = stats.suppression_score ?? 0;
    document.getElementById('emotionalRangeScore').textContent = stats.emotional_range_score ?? 0;

    // Time in state table
    const timePct = stats.emotion_time_pct || {{}};
    const timeRows = EMOTIONS.map(e => (
      `<tr><td><span style="display:inline-block; width:10px; height:10px; border-radius:3px; background:${{EMOTION_COLORS[e]}}; margin-right:8px;"></span>${{e}}</td><td>${{(timePct[e] ?? 0).toFixed ? timePct[e].toFixed(1) : timePct[e]}}%</td></tr>`
    )).join('');
    document.getElementById('timeInState').innerHTML = `
      <table><thead><tr><th>Emotion</th><th>%</th></tr></thead><tbody>${{timeRows}}</tbody></table>
    `;

    // Transitions table
    if (transitions.length === 0) {{
      document.getElementById('transitions').innerHTML = '<div class="muted">No stable transitions detected.</div>';
    }} else {{
      const rows = transitions.map(tr => (
        `<tr><td>${{tr.from_emotion}}</td><td>${{tr.to_emotion}}</td><td>${{Number(tr.timestamp_sec).toFixed(2)}}s</td><td>${{Number(tr.transition_duration_sec).toFixed(2)}}s</td></tr>`
      )).join('');
      document.getElementById('transitions').innerHTML = `
        <table>
          <thead><tr><th>From</th><th>To</th><th>Timestamp</th><th>Duration</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    // Micro-expression list
    if (microExpressions.length === 0) {{
      document.getElementById('microList').innerHTML = 'No micro-expressions detected.';
    }} else {{
      const lines = microExpressions.slice(0, 25).map(me => (
        `<div><b>#${{me.id}}</b> ${{
          me.emotion
        }} at ${{Number(me.timestamp_sec).toFixed(2)}}s for ${{me.duration_sec}}s (peak ${{me.peak_probability}})</div>`
      )).join('');
      const more = microExpressions.length > 25 ? ` (showing first 25 of ${{microExpressions.length}})` : '';
      document.getElementById('microList').innerHTML = lines + more;
    }}
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def _make_demo_video(frame_series: list, cap_in: cv2.VideoCapture):
    """
    Best-effort: generate `demo.mp4` with dominant-emotion overlay.
    This is not part of the schema required by Sentio Mind ingestion,
    but it fulfills the assignment deliverable if video codecs are available.
    """
    fps_in = cap_in.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(DEMO_OUT), fourcc, fps_in, (w, h))
    if not writer.isOpened():
        return

    max_demo_sec = 100.0  # under 2 minutes
    frame_idx = 0

    # Map analyzed time -> dominant emotion using nearest index.
    # Analysis uses ~ANALYSIS_FPS timestamps in frame_series.
    while True:
        ret, frame = cap_in.read()
        if not ret:
            break
        t = frame_idx / fps_in
        if t > max_demo_sec:
            break

        if frame_series:
            ai = int(round(t * ANALYSIS_FPS))
            ai = max(0, min(len(frame_series) - 1, ai))
            emo_probs = frame_series[ai]["emotions"]
            dominant = max(emo_probs, key=lambda e: emo_probs.get(e, 0))
        else:
            dominant = "neutral"

        cv2.putText(
            frame,
            f"t={t:.2f}s  dom={dominant}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    writer.release()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find input video. Expected 'video_sample_1.mov' or a mov under 'Video_1/'."
        )

    model, model_type = load_emotion_model()
    print(f"Emotion model: {model_type}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps_in if fps_in > 0 else 0.0

    sample_every = max(1, int(round(fps_in / ANALYSIS_FPS)))
    frame_series = []

    frame_idx = 0
    _last_progress_t = -5.0  # track last printed second
    while True:
        # Use grab() to skip decoding on frames we don't analyze.
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % sample_every == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            ts = frame_idx / fps_in if fps_in > 0 else 0.0

            # Print progress every 5 seconds of video processed.
            if ts - _last_progress_t >= 5.0:
                pct = (ts / dur * 100) if dur > 0 else 0.0
                print(f"  Processing... t={ts:.1f}s / {dur:.1f}s  ({pct:.0f}%)", flush=True)
                _last_progress_t = ts

            face_crop, _bbox = detect_primary_face(frame)
            if face_crop is not None:
                emotions = analyse_emotion(face_crop, model, model_type)
            elif frame_series:
                emotions = frame_series[-1]["emotions"].copy()
            else:
                emotions = {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

            frame_series.append({"t": float(round(ts, 3)), "emotions": emotions})

        frame_idx += 1

    cap.release()
    print(f"Analysed {len(frame_series)} frames over {dur:.1f}s")

    micro_expressions = detect_micro_expressions(frame_series)
    transitions = detect_transitions(frame_series)
    suppression = compute_suppression_score(frame_series, micro_expressions)
    emo_range = compute_emotional_range(frame_series)

    emo_counts = {e: 0 for e in EMOTIONS}
    for fs in frame_series:
        dom = max(fs["emotions"], key=fs["emotions"].get)
        emo_counts[dom] += 1
    n = len(frame_series) or 1
    emo_time_pct = {e: float(round(c / n * 100.0, 1)) for e, c in emo_counts.items()}
    dominant = max(emo_time_pct, key=emo_time_pct.get)

    stats = {
        "source": "p6_emotion_timeline",
        "video": VIDEO_DISPLAY_NAME,
        "duration_sec": float(round(dur, 2)),
        "fps_analyzed": ANALYSIS_FPS,
        "dominant_emotion": dominant,
        "emotion_time_pct": emo_time_pct,
        "suppression_score": int(suppression),
        "emotional_range_score": int(emo_range),
        "micro_expressions": micro_expressions,
        "transitions": transitions,
        "frame_series": frame_series,
    }

    OUTPUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    generate_emotion_timeline_html(
        frame_series, micro_expressions, transitions, stats, REPORT_HTML_OUT
    )

    print()
    print("=" * 50)
    print(f"  Dominant emotion:     {dominant}  ({emo_time_pct[dominant]}%)")
    print(f"  Micro-expressions:    {len(micro_expressions)}")
    print(f"  Suppression score:    {suppression}")
    print(f"  Emotional range:      {emo_range}")
    print(f"  Transitions:          {len(transitions)}")
    print(f"  Report -> {REPORT_HTML_OUT}")
    print(f"  JSON   -> {OUTPUT_JSON}")
    print("=" * 50)


