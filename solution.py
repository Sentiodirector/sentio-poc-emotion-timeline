"""
emotion_timeline.py
Sentio Mind · Project 6 · Micro-Expression & Emotion Transition Timeline

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Place models/emotion_ferplus.onnx in the same folder if not using DeepFace.
"""

import cv2
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path(__file__).parent / "video_sample_1.mov"
REPORT_HTML_OUT = Path("emotion_timeline.html")
OUTPUT_JSON     = Path("emotion_timeline_output.json")
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANALYSIS_FPS    = 8       # frames per second to analyse
MICRO_MAX_SEC   = 0.5     # micro-expression must last less than this
MICRO_MIN_PROB  = 0.40    # peak probability to qualify
NEUTRAL_MIN     = 0.50    # min neutral probability to count as "neutral before/after"
EMA_ALPHA       = 0.35    # smoothing factor for exponential moving average

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

_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


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
        from deepface import DeepFace
        return DeepFace, "deepface"
    except ImportError:
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
            result = model.analyze(
                face_crop, actions=['emotion'], enforce_detection=False
            )
            emotions = result[0]['emotion']
            total = sum(emotions.values()) or 1
            return {e: round(emotions.get(e, 0) / total * 100, 2) for e in EMOTIONS}
        except Exception:
            return {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    elif model_type == "ferplus":
        face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        blob = cv2.dnn.blobFromImage(face, 1.0, (48, 48), (0, 0, 0), swapRB=False, crop=False)
        model.setInput(blob)
        preds = model.forward()[0]
        preds = preds[:7]
        preds = np.maximum(preds, 0)
        total = np.sum(preds) or 1
        probs = preds / total * 100
        return {EMOTIONS[i]: round(float(probs[i]), 2) for i in range(7)}

    else:
        return {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}


def smooth_emotions(current: dict, previous: dict, alpha: float = EMA_ALPHA) -> dict:
    """Exponential moving average to reduce frame-to-frame jitter."""
    if previous is None:
        return current
    smoothed = {}
    for e in EMOTIONS:
        smoothed[e] = round(alpha * current[e] + (1 - alpha) * previous[e], 2)
    return smoothed


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = frame[y:y+h, x:x+w]
    return face_crop, (x, y, w, h)


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    micro = []
    fps = ANALYSIS_FPS
    max_frames = int(fps * MICRO_MAX_SEC)
    i = 1

    while i < len(frame_series) - 1:
        prev = frame_series[i - 1]

        if prev["emotions"]["neutral"] < NEUTRAL_MIN * 100:
            i += 1
            continue

        curr = frame_series[i]
        for emotion in EMOTIONS:
            if emotion == "neutral":
                continue
            prob = curr["emotions"][emotion]
            if prob < MICRO_MIN_PROB * 100:
                continue

            start = i
            j = i
            while (j < len(frame_series)
                   and frame_series[j]["emotions"][emotion] >= MICRO_MIN_PROB * 100):
                j += 1

            duration_frames = j - start
            if duration_frames >= max_frames:
                i = j
                break

            peak_prob = max(fs["emotions"][emotion] for fs in frame_series[start:j])

            if j < len(frame_series):
                after = frame_series[j]
                after_dominant = max(after["emotions"], key=after["emotions"].get)
                after_neutral = after["emotions"]["neutral"] >= NEUTRAL_MIN * 100
            else:
                after_dominant = "neutral"
                after_neutral = True

            micro.append({
                "id": len(micro) + 1,
                "timestamp_sec": frame_series[start]["t"],
                "duration_sec": round(duration_frames / fps, 3),
                "emotion": emotion,
                "peak_probability": round(peak_prob, 2),
                "followed_by": after_dominant,
                "is_suppressed": after_neutral,
            })
            i = j
            break
        else:
            i += 1

    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    transitions = []
    fps = ANALYSIS_FPS

    for i in range(1, len(frame_series)):
        prev_dom = max(frame_series[i-1]["emotions"], key=frame_series[i-1]["emotions"].get)
        curr_dom = max(frame_series[i]["emotions"], key=frame_series[i]["emotions"].get)

        if prev_dom != curr_dom:
            transitions.append({
                "from_emotion": prev_dom,
                "to_emotion": curr_dom,
                "timestamp_sec": frame_series[i]["t"],
                "transition_duration_sec": round(1 / fps, 3),
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

    std = float(np.std(all_probs)) if all_probs else 0.0
    score = (len(distinct) / 7) * 100 + std * 2
    return int(min(100, score))


# ---------------------------------------------------------------------------
# TRANSITION FREQUENCY MATRIX
# ---------------------------------------------------------------------------

def build_transition_matrix(transitions: list) -> dict:
    matrix = {e: {e2: 0 for e2 in EMOTIONS} for e in EMOTIONS}
    for t in transitions:
        matrix[t["from_emotion"]][t["to_emotion"]] += 1
    return matrix


# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, output_path):
    times = [float(fs["t"]) for fs in frame_series]

    data = {e: [] for e in EMOTIONS}
    for fs in frame_series:
        for e in EMOTIONS:
            data[e].append(float(fs["emotions"][e]))

    chartjs_path = Path(__file__).parent / "chart.umd.min.js"
    if chartjs_path.exists():
        chartjs_code = chartjs_path.read_text(encoding="utf-8")
    else:
        chartjs_code = "console.error('Chart.js not found — download chart.umd.min.js');"

    micro_json = json.dumps(micro_expressions, default=float)
    trans_matrix = build_transition_matrix(transitions)

    emo_pct = stats["emotion_time_pct"]
    sorted_emo = sorted(emo_pct.items(), key=lambda x: x[1], reverse=True)

    # Build transition matrix HTML rows
    trans_rows = ""
    for e_from in EMOTIONS:
        cells = ""
        for e_to in EMOTIONS:
            v = trans_matrix[e_from][e_to]
            opacity = min(1.0, v / max(1, max(trans_matrix[e_from].values()))) if v > 0 else 0
            bg = f"rgba(255,255,255,{opacity * 0.3:.2f})" if v > 0 else "transparent"
            cells += f'<td style="background:{bg};text-align:center;">{v if v > 0 else "·"}</td>'
        trans_rows += f'<tr><td style="font-weight:600;color:{EMOTION_COLORS[e_from]}">{e_from}</td>{cells}</tr>\n'

    trans_header = "".join(
        f'<th style="color:{EMOTION_COLORS[e]};font-size:0.75rem;writing-mode:vertical-rl;'
        f'text-orientation:mixed;padding:4px 8px;">{e}</th>'
        for e in EMOTIONS
    )

    # Micro-expression list items
    micro_items = ""
    for m in micro_expressions:
        color = EMOTION_COLORS.get(m["emotion"], "#fff")
        suppressed = "suppressed → neutral" if m["is_suppressed"] else f"→ {m['followed_by']}"
        micro_items += (
            f'<div class="micro-item">'
            f'<span class="micro-dot" style="background:{color}"></span>'
            f'<span class="micro-emo">{m["emotion"]}</span>'
            f'<span class="micro-time">{m["timestamp_sec"]}s</span>'
            f'<span class="micro-dur">{m["duration_sec"]}s</span>'
            f'<span class="micro-peak">peak {m["peak_probability"]:.0f}%</span>'
            f'<span class="micro-tag {"micro-suppressed" if m["is_suppressed"] else "micro-open"}">{suppressed}</span>'
            f'</div>\n'
        )

    # Emotion distribution bars
    dist_bars = ""
    for emo, pct in sorted_emo:
        color = EMOTION_COLORS[emo]
        dist_bars += (
            f'<div class="dist-row">'
            f'<span class="dist-label" style="color:{color}">{emo}</span>'
            f'<div class="dist-track"><div class="dist-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="dist-pct">{pct}%</span>'
            f'</div>\n'
        )

    datasets_js = ",\n".join(
        f'{{"label":"{e}","data":{data[e]},"fill":true,"tension":0.4,'
        f'"borderColor":"{EMOTION_COLORS[e]}","backgroundColor":"{EMOTION_COLORS[e]}44",'
        f'"borderWidth":1.5,"pointRadius":0}}'
        for e in EMOTIONS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emotion Timeline — Sentio Mind</title>
<script>{chartjs_code}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0b1120;color:#e2e8f0;min-height:100vh}}
.header{{background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);padding:32px 0 24px;text-align:center;border-bottom:1px solid rgba(255,255,255,.06)}}
.header h1{{font-size:1.8rem;font-weight:700;letter-spacing:-0.02em}}
.header p{{color:#64748b;font-size:.85rem;margin-top:4px}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px 20px 48px}}
.grid{{display:grid;gap:20px}}
.grid-stats{{grid-template-columns:repeat(4,1fr)}}
.grid-2col{{grid-template-columns:1fr 1fr}}
@media(max-width:768px){{.grid-stats,.grid-2col{{grid-template-columns:1fr}}}}
.card{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:24px;backdrop-filter:blur(12px)}}
.card-title{{font-size:.85rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px}}
.stat-card{{text-align:center;padding:28px 16px}}
.stat-val{{font-size:2rem;font-weight:700;line-height:1}}
.stat-label{{font-size:.78rem;color:#64748b;margin-top:6px}}
.gauge{{position:relative;width:80px;height:80px;margin:0 auto 12px}}
.gauge svg{{transform:rotate(-90deg)}}
.gauge-bg{{fill:none;stroke:rgba(255,255,255,.08);stroke-width:6}}
.gauge-fill{{fill:none;stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .8s ease}}
.gauge-text{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:1.1rem;font-weight:700}}
.chart-wrap{{position:relative}}
.chart-wrap canvas{{max-height:400px}}
.micro-item{{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;font-size:.82rem;transition:background .15s}}
.micro-item:hover{{background:rgba(255,255,255,.06)}}
.micro-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.micro-emo{{font-weight:600;min-width:60px}}
.micro-time{{color:#64748b;min-width:50px}}
.micro-dur{{color:#475569;min-width:40px}}
.micro-peak{{color:#94a3b8;min-width:60px}}
.micro-tag{{font-size:.72rem;padding:2px 8px;border-radius:10px;font-weight:500}}
.micro-suppressed{{background:rgba(239,68,68,.15);color:#f87171}}
.micro-open{{background:rgba(34,197,94,.15);color:#4ade80}}
.micro-scroll{{max-height:320px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.1) transparent}}
.dist-row{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
.dist-label{{font-size:.82rem;font-weight:600;min-width:65px;text-align:right}}
.dist-track{{flex:1;height:10px;background:rgba(255,255,255,.06);border-radius:5px;overflow:hidden}}
.dist-fill{{height:100%;border-radius:5px;transition:width .6s ease}}
.dist-pct{{font-size:.78rem;color:#94a3b8;min-width:40px}}
table.matrix{{width:100%;border-collapse:separate;border-spacing:3px;font-size:.78rem}}
table.matrix th,table.matrix td{{padding:6px;border-radius:4px}}
table.matrix td{{color:#cbd5e1}}
.section-gap{{margin-top:20px}}
</style>
</head>
<body>

<div class="header">
  <h1>Emotion Timeline Analysis</h1>
  <p>Sentio Mind · Micro-Expression & Transition Detection · {stats['duration_sec']}s video @ {stats['fps_analyzed']} fps</p>
</div>

<div class="wrap">

<!-- Stats Row -->
<div class="grid grid-stats" style="margin-bottom:20px">
  <div class="card stat-card">
    <div class="stat-val" style="color:{EMOTION_COLORS[stats['dominant_emotion']]}">{stats['dominant_emotion'].title()}</div>
    <div class="stat-label">Dominant Emotion</div>
  </div>
  <div class="card stat-card">
    <div class="gauge">
      <svg viewBox="0 0 36 36"><circle class="gauge-bg" cx="18" cy="18" r="15.9"/><circle class="gauge-fill" cx="18" cy="18" r="15.9" stroke="#ef4444" stroke-dasharray="{stats['suppression_score']},100"/></svg>
      <div class="gauge-text">{stats['suppression_score']}</div>
    </div>
    <div class="stat-label">Suppression Score</div>
  </div>
  <div class="card stat-card">
    <div class="gauge">
      <svg viewBox="0 0 36 36"><circle class="gauge-bg" cx="18" cy="18" r="15.9"/><circle class="gauge-fill" cx="18" cy="18" r="15.9" stroke="#3b82f6" stroke-dasharray="{stats['emotional_range_score']},100"/></svg>
      <div class="gauge-text">{stats['emotional_range_score']}</div>
    </div>
    <div class="stat-label">Emotional Range</div>
  </div>
  <div class="card stat-card">
    <div class="stat-val">{len(micro_expressions)}</div>
    <div class="stat-label">Micro-Expressions</div>
  </div>
</div>

<!-- River Chart -->
<div class="card" style="margin-bottom:20px">
  <div class="card-title">Emotion River Chart</div>
  <div class="chart-wrap"><canvas id="riverChart"></canvas></div>
</div>

<!-- Distribution + Transitions -->
<div class="grid grid-2col section-gap">
  <div class="card">
    <div class="card-title">Emotion Distribution</div>
    {dist_bars}
  </div>
  <div class="card">
    <div class="card-title">Transition Matrix</div>
    <div style="overflow-x:auto">
    <table class="matrix">
      <tr><th></th>{trans_header}</tr>
      {trans_rows}
    </table>
    </div>
  </div>
</div>

<!-- Micro-Expressions -->
<div class="card section-gap">
  <div class="card-title">Micro-Expressions ({len(micro_expressions)} detected)</div>
  <div class="micro-scroll">
    {micro_items if micro_items else '<p style="color:#475569;font-size:.85rem">No micro-expressions detected.</p>'}
  </div>
</div>

</div>

<script>
const labels = {json.dumps(times, default=float)};
const microEvents = {micro_json};

const microLinePlugin = {{
  id: 'microLines',
  afterDraw(chart) {{
    const ctx = chart.ctx;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    microEvents.forEach(m => {{
      const idx = labels.findIndex(l => Math.abs(l - m.timestamp_sec) < 0.001);
      if (idx < 0) return;
      const x = xScale.getPixelForValue(idx);
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 1;
      ctx.moveTo(x, yScale.top);
      ctx.lineTo(x, yScale.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#fff';
      ctx.font = '600 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(m.emotion, x, yScale.top - 4);
      ctx.restore();
    }});
  }}
}};

const config = {{
  type: 'line',
  data: {{
    labels: labels.map(t => t.toFixed(1) + 's'),
    datasets: [{datasets_js}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{
        labels: {{ color: '#94a3b8', usePointStyle: true, pointStyle: 'circle', padding: 16, font: {{ size: 11 }} }}
      }},
      tooltip: {{
        mode: 'index',
        backgroundColor: 'rgba(15,23,42,0.95)',
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        callbacks: {{
          label: function(ctx) {{
            return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%';
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        stacked: true,
        ticks: {{ color: '#475569', maxTicksLimit: 20, font: {{ size: 10 }} }},
        grid: {{ color: 'rgba(255,255,255,0.03)' }}
      }},
      y: {{
        stacked: true,
        beginAtZero: true,
        max: 100,
        ticks: {{ color: '#475569', callback: v => v + '%', font: {{ size: 10 }} }},
        grid: {{ color: 'rgba(255,255,255,0.03)' }}
      }}
    }}
  }},
  plugins: [microLinePlugin]
}};

new Chart(document.getElementById('riverChart'), config);
</script>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, model_type = load_emotion_model()
    print(f"Emotion model: {model_type}")

    cap    = cv2.VideoCapture(str(VIDEO_PATH))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur    = total / fps_in

    sample_every = max(1, int(fps_in / ANALYSIS_FPS))
    expected_samples = total // sample_every
    frame_series = []
    prev_smoothed = None

    print(f"Video: {total} frames, {dur:.1f}s, sampling every {sample_every} frames (~{expected_samples} samples)")

    frame_idx = 0
    analysed  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            analysed += 1
            if analysed % 10 == 0 or analysed == 1:
                print(f"  Processing frame {analysed}/{expected_samples} (t={frame_idx/fps_in:.1f}s) ...", flush=True)
            ts = frame_idx / fps_in
            face_crop, bbox = detect_primary_face(frame)
            if face_crop is not None:
                raw_emotions = analyse_emotion(face_crop, model, model_type)
            elif frame_series:
                raw_emotions = frame_series[-1]["emotions"].copy()
            else:
                raw_emotions = {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}

            emotions = smooth_emotions(raw_emotions, prev_smoothed)
            prev_smoothed = emotions
            frame_series.append({"t": round(ts, 3), "emotions": emotions})
        frame_idx += 1
    cap.release()

    print(f"Analysed {len(frame_series)} frames over {dur:.1f}s")

    micro_expressions = detect_micro_expressions(frame_series)
    transitions       = detect_transitions(frame_series)
    suppression       = compute_suppression_score(frame_series, micro_expressions)
    emo_range         = compute_emotional_range(frame_series)

    emo_counts = {e: 0 for e in EMOTIONS}
    for fs in frame_series:
        dom = max(fs["emotions"], key=fs["emotions"].get)
        emo_counts[dom] += 1
    n = len(frame_series) or 1
    emo_time_pct = {e: round(c / n * 100, 1) for e, c in emo_counts.items()}
    dominant     = max(emo_time_pct, key=emo_time_pct.get)

    stats = {
        "source":                "p6_emotion_timeline",
        "video":                 str(VIDEO_PATH),
        "duration_sec":          round(dur, 2),
        "fps_analyzed":          ANALYSIS_FPS,
        "dominant_emotion":      dominant,
        "emotion_time_pct":      emo_time_pct,
        "suppression_score":     suppression,
        "emotional_range_score": emo_range,
        "micro_expressions":     micro_expressions,
        "transitions":           transitions,
        "frame_series":          frame_series,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=float)

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
