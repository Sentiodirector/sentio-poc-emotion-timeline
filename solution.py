"""
solution.py
Sentio Mind · Project 6 · Micro-Expression & Emotion Transition Timeline

Detects sub-second micro-expressions, tracks emotion transitions,
computes suppression/range scores, and classifies Duchenne vs social smiles.

Run:  python solution.py
Deps: opencv-python, deepface, mediapipe, numpy, Pillow, tqdm
"""

import cv2
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("emotion_timeline.html")
OUTPUT_JSON     = Path("emotion_timeline_output.json")
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANALYSIS_FPS    = 8       # frames per second to analyse
MICRO_MAX_SEC   = 0.5     # micro-expression must last less than this
MICRO_MIN_PROB  = 0.40    # peak probability to qualify
NEUTRAL_MIN     = 0.50    # min neutral probability to count as "neutral before/after"

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


# ---------------------------------------------------------------------------
# EMOTION MODEL LOADER
# ---------------------------------------------------------------------------

def load_emotion_model():
    """
    Try DeepFace first.
    If not installed, try loading models/emotion_ferplus.onnx with OpenCV DNN.
    If neither available, return (None, 'fallback').
    Return: (model_object, model_type_string)
    TODO: implement
    """
    try:
        from deepface import DeepFace
        return DeepFace, "deepface"
    except Exception:
        pass

    onnx = MODELS_DIR / "emotion_ferplus.onnx"
    if onnx.exists():
        net = cv2.dnn.readNetFromONNX(str(onnx))
        return net, "ferplus"

    print("WARNING: no emotion model found. Using random baseline.")
    return None, "fallback"


# ---------------------------------------------------------------------------
# EMOTION ANALYSIS — ONE FACE CROP
# ---------------------------------------------------------------------------

def analyse_emotion(face_crop: np.ndarray, model, model_type: str) -> dict:
    if model_type == "deepface":
        try:
            result = model.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']

            total = sum(emotions.values()) or 1
            return {e: round(emotions.get(e, 0) / total * 100, 2) for e in EMOTIONS}

        except Exception:
            return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

    elif model_type == "ferplus":
        face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        blob = cv2.dnn.blobFromImage(face, 1.0, (48, 48), (0, 0, 0), swapRB=False, crop=False)
        model.setInput(blob)
        preds = model.forward()[0]

        preds = preds[:7]  # drop contempt
        preds = np.maximum(preds, 0)
        total = np.sum(preds) or 1
        probs = preds / total * 100

        return {EMOTIONS[i]: round(probs[i], 2) for i in range(7)}

    else:
        return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    # largest face
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face_crop = frame[y:y+h, x:x+w]

    return face_crop, (x, y, w, h)


# ---------------------------------------------------------------------------
# DUCHENNE SMILE DETECTOR (BONUS)
# Uses OpenCV eye detection to distinguish genuine vs social smiles.
# Genuine (Duchenne): zygomatic major + orbicularis oculi → mouth up AND eyes narrow
# Social: mouth up only, eyes stay open
# Method: Measure Eye Aspect Ratio (height/width) from Haar Cascade eye ROIs.
# ---------------------------------------------------------------------------

# Pre-load cascade once (module level)
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _init_face_mesh():
    """Initialize the Duchenne detector. Returns True if available."""
    if _EYE_CASCADE.empty():
        print("WARNING: eye cascade not found — Duchenne detection disabled.")
        return None
    print("Duchenne detector: OpenCV eye cascade loaded.")
    return True


def classify_smile_type(frame: np.ndarray, detector) -> str:
    """
    Classify a smile as 'duchenne' (genuine) or 'social' using eye openness.
    
    Duchenne (genuine) smile: orbicularis oculi contracts → eyes narrow/squint.
    Social smile: only mouth moves, eyes remain fully open.
    
    Method: Detect eyes via Haar Cascade, compute aspect ratio (height/width).
    - Narrowed eyes (Duchenne): aspect ratio < threshold
    - Open eyes (social smile): aspect ratio >= threshold
    
    Returns 'duchenne', 'social', or 'none'.
    """
    if detector is None:
        return "none"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face first to limit eye search region
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "none"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Search for eyes only in the upper half of the face
    face_upper = gray[y:y + h // 2, x:x + w]
    eyes = _EYE_CASCADE.detectMultiScale(face_upper, 1.1, 4, minSize=(w // 8, h // 16))

    if len(eyes) < 2:
        # Can't detect both eyes — inconclusive
        return "none"

    # Compute average Eye Aspect Ratio (height / width) for detected eyes
    # Sort by x to get left and right eyes
    eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
    ear_values = []
    for (ex, ey, ew, eh) in eyes_sorted:
        if ew > 0:
            ear_values.append(eh / ew)

    if not ear_values:
        return "none"

    avg_ear = sum(ear_values) / len(ear_values)

    # Duchenne threshold: genuine smiles narrow the eyes
    # Typical open eye ratio ≈ 0.45-0.60
    # Duchenne (squinted) eye ratio ≈ 0.25-0.40
    DUCHENNE_THRESHOLD = 0.42

    if avg_ear < DUCHENNE_THRESHOLD:
        return "duchenne"
    else:
        return "social"


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    micro = []
    i = 1
    fps = ANALYSIS_FPS

    while i < len(frame_series) - 1:
        curr = frame_series[i]
        prev = frame_series[i - 1]

        if prev["emotions"]["neutral"] >= NEUTRAL_MIN * 100:
            for emotion, prob in curr["emotions"].items():
                if emotion != "neutral" and prob >= MICRO_MIN_PROB * 100:

                    start = i
                    j = i

                    while j < len(frame_series) and frame_series[j]["emotions"][emotion] >= MICRO_MIN_PROB * 100:
                        j += 1

                    duration_frames = j - start
                    duration_sec = duration_frames / fps

                    if duration_sec < MICRO_MAX_SEC:
                        if j < len(frame_series) and frame_series[j]["emotions"]["neutral"] >= NEUTRAL_MIN * 100:

                            peak_prob = max([fs["emotions"][emotion] for fs in frame_series[start:j]] or [0])

                            micro.append({
                                "id": len(micro) + 1,
                                "timestamp_sec": frame_series[start]["t"],
                                "duration_sec": round(duration_sec, 3),
                                "emotion": emotion,
                                "peak_probability": round(peak_prob, 2),
                                "followed_by": "neutral",
                                "is_suppressed": True
                            })
                    i = j
                    break
        i += 1

    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    transitions = []
    fps = ANALYSIS_FPS

    for i in range(1, len(frame_series)):
        prev = max(frame_series[i-1]["emotions"], key=frame_series[i-1]["emotions"].get)
        curr = max(frame_series[i]["emotions"], key=frame_series[i]["emotions"].get)

        if prev != curr:
            transitions.append({
                "from_emotion": prev,
                "to_emotion": curr,
                "timestamp_sec": frame_series[i]["t"],
                "transition_duration_sec": round(1/fps, 3)
            })

    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    total_events = 0

    for fs in frame_series:
        for e, p in fs["emotions"].items():
            if e != "neutral" and p > 35:
                total_events += 1
                break

    if total_events == 0:
        return 0

    score = (len(micro_expressions) / total_events) * 100
    return int(min(100, score))


def compute_emotional_range(frame_series: list) -> int:
    distinct = set()
    all_probs = []

    for fs in frame_series:
        for e, p in fs["emotions"].items():
            all_probs.append(p)
            if p > 30:
                distinct.add(e)

    std = np.std(all_probs) if all_probs else 0

    score = (len(distinct) / 7) * 100 + std * 2
    return int(min(100, score))

# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def _load_chartjs_inline():
    """Load Chart.js from local file for offline bundling."""
    p = Path("chart.min.js")
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Fallback: try to download once
    import urllib.request
    try:
        urllib.request.urlretrieve(
            "https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js",
            str(p),
        )
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_transition_matrix(transitions):
    """Build a from→to count matrix from the transitions list."""
    matrix = {}
    for t in transitions:
        fr, to = t["from_emotion"], t["to_emotion"]
        matrix.setdefault(fr, {})
        matrix[fr][to] = matrix[fr].get(to, 0) + 1
    return matrix


def generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, output_path):

    times = [float(fs["t"]) for fs in frame_series]

    data = {e: [] for e in EMOTIONS}
    for fs in frame_series:
        for e in EMOTIONS:
            data[e].append(float(fs["emotions"][e]))

    chart_js_code = _load_chartjs_inline()
    trans_matrix = _build_transition_matrix(transitions)

    # Build micro-expression timestamps for dashed vertical lines
    micro_ts = [m["timestamp_sec"] for m in micro_expressions]
    micro_labels = [f"{m['emotion']} ({round(m['peak_probability'],1)}%)" for m in micro_expressions]

    # --- Emotion time % bars ---
    emo_pct = stats.get("emotion_time_pct", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Emotion Timeline — Sentio Mind</title>
<script>{chart_js_code}</script>
<style>
*,*::before,*::after{{box-sizing:border-box}}
:root{{
  --bg:#0d0d0d;--surface:#0a0a0a;--accent:#ff6600;
  --border:#2a2a2a;--text:#fff;--muted:#737373;
  --mono:'Consolas','Courier New',monospace;
  --sans:system-ui,-apple-system,'Segoe UI',sans-serif;
}}
body{{margin:0;font-family:var(--sans);background:var(--bg);color:var(--text);padding:50px 20px;display:flex;justify-content:center}}
.box{{width:100%;max-width:1100px;border:1px solid var(--border);padding:40px;position:relative;background:var(--surface)}}
.corner{{position:absolute;width:10px;height:10px;border-color:#555;border-style:solid}}
.tl{{top:-1px;left:-1px;border-width:1px 0 0 1px}}.tr{{top:-1px;right:-1px;border-width:1px 1px 0 0}}
.bl{{bottom:-1px;left:-1px;border-width:0 0 1px 1px}}.br{{bottom:-1px;right:-1px;border-width:0 1px 1px 0}}
.hdr{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--border);padding-bottom:20px;margin-bottom:30px}}
.hdr .sys{{font-family:var(--mono);color:var(--accent);font-size:.75rem;letter-spacing:1px;margin-bottom:5px}}
.hdr h1{{margin:0;font-size:2rem;font-weight:600}}
.hdr-r{{font-family:var(--mono);color:var(--muted);font-size:.7rem;text-align:right;line-height:1.5}}
.section-label{{font-family:var(--mono);color:var(--accent);font-size:.75rem;letter-spacing:1px;margin-bottom:12px}}
.chart-wrap{{margin-top:20px;padding:10px 0}}

/* Stats grid */
.stats{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--border);margin-top:40px}}
.stat{{padding:28px 20px;text-align:center;border-right:1px solid var(--border);display:flex;flex-direction:column;justify-content:center}}
.stat:last-child{{border-right:none}}
.stat-v{{color:var(--accent);font-size:2.2rem;font-weight:400;margin-bottom:6px}}
.stat-l{{font-family:var(--mono);color:var(--muted);font-size:.6rem;letter-spacing:1.5px;text-transform:uppercase}}

/* Emotion % bars */
.emo-bars{{margin-top:40px}}
.bar-row{{display:flex;align-items:center;margin-bottom:8px;font-family:var(--mono);font-size:.75rem}}
.bar-name{{width:90px;color:var(--muted);text-transform:uppercase}}
.bar-track{{flex:1;height:8px;background:#1a1a1a;border-radius:4px;overflow:hidden;margin:0 12px}}
.bar-fill{{height:100%;border-radius:4px}}
.bar-val{{width:45px;text-align:right;color:var(--accent)}}

/* Micro expressions */
ul.micro{{font-family:var(--mono);font-size:.72rem;color:var(--muted);list-style:none;padding:0;margin:0;max-height:300px;overflow-y:auto}}
ul.micro li{{padding:8px 0;border-bottom:1px dashed var(--border);display:flex;justify-content:space-between;gap:10px}}
ul.micro li:last-child{{border-bottom:none}}
ul.micro .emo{{color:var(--text)}}
ul.micro .val{{color:var(--accent)}}

/* Transition table */
.trans-table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.7rem;margin-top:12px}}
.trans-table th,.trans-table td{{padding:8px 6px;border:1px solid var(--border);text-align:center}}
.trans-table th{{background:#111;color:var(--accent);text-transform:uppercase;font-weight:400}}
.trans-table td{{color:var(--muted)}}
.trans-table td.has-val{{color:var(--text)}}
</style>
</head>
<body>
<div class="box">
  <div class="corner tl"></div><div class="corner tr"></div>
  <div class="corner bl"></div><div class="corner br"></div>

  <div class="hdr">
    <div>
      <div class="sys">SYS: EMOTION_ANALYSIS</div>
      <h1>Emotion Timeline</h1>
    </div>
    <div class="hdr-r">
      <div>DURATION: {stats['duration_sec']}s</div>
      <div>FPS_ANALYZED: {stats['fps_analyzed']}</div>
      <div>FRAMES: {len(frame_series)}</div>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat"><div class="stat-v">{str(stats['dominant_emotion']).upper()}</div><div class="stat-l">DOMINANT_EMOTION</div></div>
    <div class="stat"><div class="stat-v">{stats['suppression_score']}</div><div class="stat-l">SUPPRESSION_SCORE</div></div>
    <div class="stat"><div class="stat-v">{stats['emotional_range_score']}</div><div class="stat-l">EMOTIONAL_RANGE</div></div>
  </div>

  <!-- Emotion Time % Bars -->
  <div class="emo-bars" style="margin-top:40px;">
    <div class="section-label">EMOTION_TIME_DISTRIBUTION</div>
"""

    for e in EMOTIONS:
        pct = emo_pct.get(e, 0)
        html += f"""    <div class="bar-row">
      <span class="bar-name">{e}</span>
      <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{EMOTION_COLORS[e]}"></div></div>
      <span class="bar-val">{pct}%</span>
    </div>\n"""

    html += """  </div>

  <!-- River Chart -->
  <div class="chart-wrap">
    <div class="section-label">STACKED_EMOTION_RIVER_CHART</div>
    <canvas id="chart"></canvas>
  </div>

  <!-- Micro Expressions -->
  <div style="margin-top:40px">
    <div class="section-label">DETECTED_MICRO_EXPRESSIONS (""" + str(len(micro_expressions)) + """)</div>
    <ul class="micro">
"""

    if not micro_expressions:
        html += "      <li>NONE_DETECTED</li>\n"
    for m in micro_expressions:
        html += f'      <li><span>&gt; MICRO</span> <span class="emo">{str(m["emotion"]).upper()}</span> <span>T={m["timestamp_sec"]}s</span> <span>DUR={m["duration_sec"]}s</span> <span class="val">PEAK: {round(m["peak_probability"],2)}</span></li>\n'

    html += """    </ul>
  </div>

  <!-- Transition Table -->
  <div style="margin-top:40px">
    <div class="section-label">EMOTION_TRANSITION_MATRIX</div>
    <table class="trans-table">
      <tr><th>FROM \\ TO</th>"""

    for e in EMOTIONS:
        html += f"<th>{e[:3].upper()}</th>"
    html += "</tr>\n"

    for fr in EMOTIONS:
        html += f"      <tr><th>{fr[:3].upper()}</th>"
        for to in EMOTIONS:
            cnt = trans_matrix.get(fr, {}).get(to, 0)
            cls = ' class="has-val"' if cnt > 0 else ''
            html += f"<td{cls}>{cnt if cnt > 0 else '·'}</td>"
        html += "</tr>\n"

    html += """    </table>
  </div>

"""

    # --- Duchenne Smile Section ---
    ds = stats.get("duchenne_smiles", {})
    genuine = ds.get("genuine_frames", 0)
    social  = ds.get("social_frames", 0)
    gpct    = ds.get("genuine_pct", 0)
    events  = ds.get("events", [])

    html += f"""  <!-- Duchenne Smile Analysis (BONUS) -->
  <div style="margin-top:40px">
    <div class="section-label">DUCHENNE_SMILE_ANALYSIS ✦ BONUS</div>
    <div class="stats" style="margin-top:12px">
      <div class="stat"><div class="stat-v" style="color:#22c55e">{genuine}</div><div class="stat-l">GENUINE (DUCHENNE)</div></div>
      <div class="stat"><div class="stat-v" style="color:#eab308">{social}</div><div class="stat-l">SOCIAL SMILES</div></div>
      <div class="stat"><div class="stat-v" style="color:#22c55e">{gpct}%</div><div class="stat-l">GENUINE %</div></div>
    </div>
"""

    if events:
        html += '    <ul class="micro" style="margin-top:12px;max-height:150px">\n'
        for i, ev in enumerate(events):
            html += f'      <li><span>&gt; GENUINE_SMILE #{i+1}</span> <span class="emo">START={ev["start_sec"]}s</span> <span class="val">END={ev["end_sec"]}s</span></li>\n'
        html += "    </ul>\n"
    else:
        html += '    <div style="font-family:var(--mono);color:var(--muted);font-size:.72rem;margin-top:10px">NO_DUCHENNE_SMILES_DETECTED</div>\n'

    html += """  </div>

</div>

<script>
"""

    # Inject micro-expression timestamps for the dashed-line plugin
    html += f"const microTimestamps = {micro_ts};\n"
    html += f"const microLabels = {json.dumps(micro_labels)};\n"

    # Inject Duchenne event ranges for green band plugin
    html += f"const duchenneEvents = {json.dumps(events)};\n"


    html += """
// Custom plugin: draw dashed vertical lines at micro-expression timestamps
const microLinePlugin = {
  id: 'microLines',
  afterDraw(chart) {
    const ctx = chart.ctx;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const labels = chart.data.labels;
    ctx.save();
    microTimestamps.forEach((ts, idx) => {
      let closest = 0, minDiff = Infinity;
      labels.forEach((l, i) => { const d = Math.abs(l - ts); if (d < minDiff) { minDiff = d; closest = i; }});
      const x = xScale.getPixelForValue(closest);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#ff660066';
      ctx.lineWidth = 1;
      ctx.moveTo(x, yScale.top);
      ctx.lineTo(x, yScale.bottom);
      ctx.stroke();
    });
    ctx.restore();
  }
};

// Custom plugin: draw green bands for Duchenne (genuine) smile regions
const duchennePlugin = {
  id: 'duchenneSmiles',
  beforeDraw(chart) {
    if (!duchenneEvents || duchenneEvents.length === 0) return;
    const ctx = chart.ctx;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const labels = chart.data.labels;
    ctx.save();
    duchenneEvents.forEach(ev => {
      let s = 0, e = 0, d1 = Infinity, d2 = Infinity;
      labels.forEach((l, i) => { let d = Math.abs(l - ev.start_sec); if (d < d1) { d1 = d; s = i; }});
      labels.forEach((l, i) => { let d = Math.abs(l - ev.end_sec); if (d < d2) { d2 = d; e = i; }});
      const x1 = xScale.getPixelForValue(s);
      const x2 = xScale.getPixelForValue(e);
      ctx.fillStyle = 'rgba(34, 197, 94, 0.08)';
      ctx.fillRect(x1, yScale.top, x2 - x1, yScale.bottom - yScale.top);
      // Top label
      ctx.fillStyle = '#22c55e88';
      ctx.font = '9px Consolas, monospace';
      ctx.fillText('😊', x1 + 2, yScale.top + 12);
    });
    ctx.restore();
  }
};

const labels = """ + str(times) + """;
const datasets = [
"""

    for e in EMOTIONS:
        html += f"""{{
  label: '{e.upper()}',
  data: {data[e]},
  fill: true, tension: 0.4,
  borderColor: '{EMOTION_COLORS[e]}',
  backgroundColor: '{EMOTION_COLORS[e]}55',
  pointRadius: 0, borderWidth: 1
}},
"""

    html += """
];

new Chart(document.getElementById('chart'), {
  type: 'line',
  data: { labels, datasets },
  plugins: [microLinePlugin, duchennePlugin],
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#737373', font: { family: 'Consolas, Courier New, monospace', size: 10 } } },
      tooltip: {
        mode: 'index',
        backgroundColor: '#0a0a0a', titleColor: '#ff6600', bodyColor: '#e5e5e5',
        borderColor: '#2a2a2a', borderWidth: 1,
        titleFont: { family: 'Consolas, Courier New, monospace' },
        bodyFont: { family: 'Consolas, Courier New, monospace' }
      }
    },
    scales: {
      x: { stacked: true, ticks: { color: '#737373', font: { family: 'Consolas, Courier New, monospace', size: 9 }, maxTicksLimit: 20 }, grid: { color: '#1a1a1a' } },
      y: { stacked: true, ticks: { color: '#737373', font: { family: 'Consolas, Courier New, monospace', size: 10 } }, grid: { color: '#1a1a1a' }, beginAtZero: true, max: 100 }
    }
  }
});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report written to {output_path}")


def get_chart_js():
    """Return inline Chart.js source."""
    return _load_chartjs_inline()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tqdm import tqdm

    model, model_type = load_emotion_model()
    print(f"Emotion model: {model_type}")

    face_mesh = _init_face_mesh()
    print(f"Duchenne detector: {'enabled' if face_mesh else 'disabled'}")

    cap    = cv2.VideoCapture(str(VIDEO_PATH))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur    = total / fps_in

    sample_every = max(1, int(fps_in / ANALYSIS_FPS))
    frame_series = []

    frame_idx = 0
    pbar = tqdm(total=total, desc="Analysing frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            ts = frame_idx / fps_in
            face_crop, bbox = detect_primary_face(frame)
            if face_crop is not None:
                emotions = analyse_emotion(face_crop, model, model_type)
            elif frame_series:
                emotions = frame_series[-1]["emotions"].copy()
            else:
                emotions = {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

            # Duchenne smile classification (only when happy is dominant)
            smile_type = "none"
            dom_emo = max(emotions, key=emotions.get)
            if dom_emo == "happy" and emotions["happy"] >= 40 and face_mesh is not None:
                smile_type = classify_smile_type(frame, face_mesh)

            frame_series.append({"t": round(ts, 3), "emotions": emotions, "smile_type": smile_type})
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    print(f"Analysed {len(frame_series)} sampled frames over {dur:.1f}s")

    micro_expressions = detect_micro_expressions(frame_series)
    transitions       = detect_transitions(frame_series)
    suppression       = compute_suppression_score(frame_series, micro_expressions)
    emo_range         = compute_emotional_range(frame_series)

    # Emotion time percentages
    emo_counts = {e: 0 for e in EMOTIONS}
    for fs in frame_series:
        dom = max(fs["emotions"], key=fs["emotions"].get)
        emo_counts[dom] += 1
    n = len(frame_series) or 1
    emo_time_pct  = {e: round(c / n * 100, 1) for e, c in emo_counts.items()}
    dominant      = max(emo_time_pct, key=emo_time_pct.get)

    # Duchenne smile summary
    duchenne_count = sum(1 for fs in frame_series if fs.get("smile_type") == "duchenne")
    social_count   = sum(1 for fs in frame_series if fs.get("smile_type") == "social")
    total_smiles   = duchenne_count + social_count
    duchenne_pct   = round(duchenne_count / total_smiles * 100, 1) if total_smiles > 0 else 0

    duchenne_events = []
    in_duchenne = False
    start_t = 0
    for fs in frame_series:
        if fs.get("smile_type") == "duchenne" and not in_duchenne:
            in_duchenne = True
            start_t = fs["t"]
        elif fs.get("smile_type") != "duchenne" and in_duchenne:
            in_duchenne = False
            duchenne_events.append({"start_sec": start_t, "end_sec": fs["t"]})
    if in_duchenne:
        duchenne_events.append({"start_sec": start_t, "end_sec": frame_series[-1]["t"]})

    stats = {
        "source":               "p6_emotion_timeline",
        "video":                str(VIDEO_PATH),
        "duration_sec":         round(dur, 2),
        "fps_analyzed":         ANALYSIS_FPS,
        "dominant_emotion":     dominant,
        "emotion_time_pct":     emo_time_pct,
        "suppression_score":    suppression,
        "emotional_range_score": emo_range,
        "micro_expressions":    micro_expressions,
        "transitions":          transitions,
        "duchenne_smiles": {
            "genuine_frames": duchenne_count,
            "social_frames":  social_count,
            "genuine_pct":    duchenne_pct,
            "events":         duchenne_events,
        },
        "frame_series":         frame_series,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=float)

    generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, REPORT_HTML_OUT)

    # OpenCV detector doesn't need explicit cleanup

    print()
    print("=" * 50)
    print(f"  Dominant emotion:     {dominant}  ({emo_time_pct[dominant]}%)")
    print(f"  Micro-expressions:    {len(micro_expressions)}")
    print(f"  Suppression score:    {suppression}")
    print(f"  Emotional range:      {emo_range}")
    print(f"  Transitions:          {len(transitions)}")
    print(f"  Duchenne smiles:      {duchenne_count} genuine / {social_count} social  ({duchenne_pct}% genuine)")
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 50)
