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
VIDEO_PATH      = Path(r"/Users/saurabhkuntal/Downloads/sentio-poc-emotion-timeline-main-sol/sentio-poc-emotion-timeline-main/video_sample_1.mov")
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
            result = model.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']

            total = sum(emotions.values()) or 1
            return {e: round(emotions.get(e, 0) / total * 100, 2) for e in EMOTIONS}

        except:
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

def generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, output_path):

    times = [float(fs["t"]) for fs in frame_series]

    data = {e: [] for e in EMOTIONS}
    for fs in frame_series:
        for e in EMOTIONS:
            data[e].append(float(fs["emotions"][e]))

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Sentio Mind · Emotion Timeline</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}}

.container {{
    width: 95%;
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}}

h1 {{
    text-align: center;
    font-size: 2.8rem;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
}}

.card {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    transition: 0.3s;
}}

.card:hover {{
    transform: translateY(-5px);
}}

.stat-value {{
    font-size: 2rem;
    font-weight: bold;
}}

.stat-label {{
    color: #94a3b8;
    margin-top: 5px;
}}

.chart-container {{
    margin-top: 30px;
    padding: 20px;
}}

ul {{
    margin-top: 10px;
}}

li {{
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.05);
}}

.badge {{
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    background: #22c55e33;
    color: #22c55e;
}}

.footer {{
    text-align: center;
    margin-top: 30px;
    color: #64748b;
    font-size: 0.9rem;
}}
</style>
</head>

<body>

<div class="container">

<h1>🧠 Emotion Intelligence Dashboard</h1>

<div class="grid">
    <div class="card">
        <div class="stat-value">{stats['dominant_emotion']}</div>
        <div class="stat-label">Dominant Emotion</div>
    </div>
    <div class="card">
        <div class="stat-value">{stats['suppression_score']}</div>
        <div class="stat-label">Suppression Score</div>
    </div>
    <div class="card">
        <div class="stat-value">{stats['emotional_range_score']}</div>
        <div class="stat-label">Emotional Range</div>
    </div>
    <div class="card">
        <div class="stat-value">{len(micro_expressions)}</div>
        <div class="stat-label">Micro Expressions</div>
    </div>
</div>

<div class="card chart-container">
    <canvas id="chart"></canvas>
</div>

<div class="card">
<h2>⚡ Micro Expression Events</h2>
<ul>
"""

    # ✅ CORRECT PLACE (inside function)
    for m in micro_expressions:
        html += f"""
<li>
    <strong>{m['emotion']}</strong> at {m['timestamp_sec']}s
    <span class="badge">Peak {round(m['peak_probability'], 2)}%</span>
</li>
"""

    html += """
</ul>
</div>

<div class="footer">
    Built with ❤️ using Computer Vision · Sentio Mind Project
</div>

</div>

<script>

const labels = """ + str(times) + """;

const datasets = [
"""

    for e in EMOTIONS:
        html += f"""
{{
    label: '{e}',
    data: {data[e]},
    fill: true,
    tension: 0.4,
    borderWidth: 2,
    borderColor: '{EMOTION_COLORS[e]}',
    backgroundColor: '{EMOTION_COLORS[e]}33',
}},
"""

    html += """
];

new Chart(document.getElementById('chart'), {
    type: 'line',
    data: {
        labels: labels,
        datasets: datasets
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                labels: {
                    color: 'white',
                    font: { size: 12 }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#94a3b8' }
            },
            y: {
                ticks: { color: '#94a3b8' },
                beginAtZero: true,
                max: 100
            }
        }
    }
});

</script>

</body>
</html>
"""

    # ✅ Write file
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
    frame_series = []

    frame_idx = 0
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
                # Interpolate: copy last known
                emotions = frame_series[-1]["emotions"].copy()
            else:
                emotions = {e: (100 if e == "neutral" else 0) for e in EMOTIONS}
            frame_series.append({"t": round(ts, 3), "emotions": emotions})
        frame_idx += 1
    cap.release()

    print(f"Analysed {len(frame_series)} frames over {dur:.1f}s")

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
        "frame_series":         frame_series,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=float)

    generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, REPORT_HTML_OUT)

    print()
    print("=" * 50)
    print(f"  Dominant emotion:     {dominant}  ({emo_time_pct[dominant]}%)")
    print(f"  Micro-expressions:    {len(micro_expressions)}")
    print(f"  Suppression score:    {suppression}")
    print(f"  Emotional range:      {emo_range}")
    print(f"  Transitions:          {len(transitions)}")
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 50)
