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
import urllib.request

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
    except ImportError:
        pass

    onnx = MODELS_DIR / "emotion_ferplus.onnx"
    if onnx.exists():
        try:
            net = cv2.dnn.readNetFromONNX(str(onnx))
            return net, "ferplus"
        except Exception as e:
            print(f"Error loading ONNX: {e}")

    print("WARNING: no emotion model found. Using random baseline.")
    return None, "fallback"


# ---------------------------------------------------------------------------
# EMOTION ANALYSIS — ONE FACE CROP
# ---------------------------------------------------------------------------

def analyse_emotion(face_crop: np.ndarray, model, model_type: str) -> dict:
    if model_type == "deepface":
        try:
            res = model.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(res, list):
                res = res[0]
            emotions = res.get('emotion', {})
            total = sum(emotions.values())
            if total > 0:
                return {e: float((emotions.get(e, 0) / total) * 100) for e in EMOTIONS}
        except Exception:
            pass

    elif model_type == "ferplus":
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)) # FerPlus ONNX usually 64x64 or 48x48
            blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0,0,0), swapRB=False, crop=False)
            model.setInput(blob)
            preds = model.forward()[0]
            # FerPlus labels: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
            mapping = {0:"neutral", 1:"happy", 2:"surprise", 3:"sad", 4:"angry", 5:"disgust", 6:"fear"}
            exp_preds = np.exp(preds - np.max(preds))
            sm = exp_preds / np.sum(exp_preds)
            emotions = {mapping[i]: float(sm[i])*100 for i in range(7)}
            return emotions
        except Exception as e:
            print(f"Emotion error: {e}")
            pass

    return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

# Use Haar Cascade as it's more reliable across different environments
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_primary_face(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None
    
    # Pick largest face
    best_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = best_face
    return frame[y:y+h, x:x+w].copy(), (x, y, w, h)


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    micro = []
    n = len(frame_series)
    in_micro = False
    current_emo = None
    start_idx = 0
    micro_counter = 1
    
    for i in range(1, n):
        emotions = frame_series[i]['emotions']
        prev_emotions = frame_series[i-1]['emotions']
        
        if not in_micro:
            for em in EMOTIONS:
                if em == 'neutral': continue
                if emotions[em] >= MICRO_MIN_PROB * 100:
                    if prev_emotions['neutral'] >= NEUTRAL_MIN * 100:
                        in_micro = True
                        current_emo = em
                        start_idx = i
                        break
        else:
            if emotions[current_emo] < MICRO_MIN_PROB * 100:
                if emotions['neutral'] >= NEUTRAL_MIN * 100:
                    duration = frame_series[i]['t'] - frame_series[start_idx]['t']
                    if duration < MICRO_MAX_SEC:
                        peak_prob = max(frame_series[j]['emotions'][current_emo] for j in range(start_idx, i))
                        micro.append({
                            "id": micro_counter,
                            "timestamp_sec": float(frame_series[start_idx]['t']),
                            "duration_sec": float(round(duration, 3)),
                            "emotion": current_emo,
                            "peak_probability": float(round(peak_prob, 1)),
                            "followed_by": "neutral",
                            "is_suppressed": True
                        })
                        micro_counter += 1
                in_micro = False
                current_emo = None
                
    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    transitions = []
    if not frame_series: return transitions
    current_dom = max(frame_series[0]['emotions'], key=frame_series[0]['emotions'].get)
    i = 1
    n = len(frame_series)
    while i < n:
        dom = max(frame_series[i]['emotions'], key=frame_series[i]['emotions'].get)
        if dom != current_dom:
            new_dom = dom
            start_i = i
            j = i
            while j < n and max(frame_series[j]['emotions'], key=frame_series[j]['emotions'].get) == new_dom:
                j += 1
            duration = frame_series[j-1]['t'] - frame_series[start_i]['t']
            if duration >= 0.5:
                transitions.append({
                    "from_emotion": current_dom,
                    "to_emotion": new_dom,
                    "timestamp_sec": float(frame_series[start_i]['t']),
                    "transition_duration_sec": float(round((frame_series[start_i]['t'] - frame_series[start_i-1]['t']), 3))
                })
                current_dom = new_dom
            i = j
        else:
            i += 1
    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    total_expression_events = 0
    for frame in frame_series:
        for em in EMOTIONS:
            if em != "neutral" and frame['emotions'][em] > 35.0:
                total_expression_events += 1
                break
    if total_expression_events == 0:
        return 0
    score = (len(micro_expressions) / total_expression_events) * 100
    return int(min(100, round(score)))


def compute_emotional_range(frame_series: list) -> int:
    if not frame_series: return 0
    appeared = set()
    all_probs = []
    for frame in frame_series:
        for em in EMOTIONS:
            prob = frame['emotions'][em]
            all_probs.append(prob)
            if prob > 30.0:
                appeared.add(em)
    distinct = len(appeared)
    std_of_all_probs = np.std(all_probs)
    range_score = min(100, (distinct / 7) * 100 + std_of_all_probs * 2)
    return int(round(range_score))


# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(frame_series: list, micro_expressions: list,
                                    transitions: list, stats: dict,
                                    output_path: Path):
    """
    Write emotion_timeline.html — self-contained, offline, no CDN.
    Uses custom SVG for a premium 'river chart' look.
    """
    if not frame_series:
        with open(output_path, "w") as f: f.write("No data")
        return

    # Chart Dimensions
    W, H = 1000, 400
    PAD = 50
    CHART_W = W - 2*PAD
    CHART_H = H - 2*PAD
    
    times = [f['t'] for f in frame_series]
    max_t = max(times) if times else 1
    
    def get_x(t): return PAD + (t / max_t) * CHART_W
    def get_y(val): return PAD + CHART_H - (val / 100.0) * CHART_H

    # Generate Emotion Paths
    paths = []
    # We need to compute stacked areas
    # Bottom baseline is 0
    current_bottoms = [0.0] * len(frame_series)
    
    for em in EMOTIONS:
        # Top line for this emotion
        top_pts = []
        bottom_pts = []
        for i, f in enumerate(frame_series):
            prob = f['emotions'].get(em, 0)
            top = current_bottoms[i] + prob
            top_pts.append((get_x(f['t']), get_y(top)))
            bottom_pts.append((get_x(f['t']), get_y(current_bottoms[i])))
            current_bottoms[i] = top
            
        # Create Path: top line left-to-right, then bottom line right-to-left
        d = f"M {top_pts[0][0]},{top_pts[0][1]} "
        for p in top_pts[1:]:
            d += f"L {p[0]},{p[1]} "
        for p in reversed(bottom_pts):
            d += f"L {p[0]},{p[1]} "
        d += "Z"
        
        paths.append({
            "emotion": em,
            "d": d,
            "color": EMOTION_COLORS[em]
        })

    # Micro-expression markers (vertical lines)
    markers = []
    for m in micro_expressions:
        x = get_x(m['timestamp_sec'])
        markers.append(f'<line x1="{x}" y1="{PAD}" x2="{x}" y2="{PAD+CHART_H}" stroke="#ef4444" stroke-width="2" stroke-dasharray="5,5" />')
        markers.append(f'<text x="{x+5}" y="{PAD+20}" fill="#ef4444" font-size="12" font-weight="bold">{m["emotion"].upper()}</text>')

    svg_paths_html = "".join([f'<path d="{p["d"]}" fill="{p["color"]}" opacity="0.8"><title>{p["emotion"]}</title></path>' for p in paths])
    markers_html = "".join(markers)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sentio Mind · Emotion Timeline</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; line-height: 1.5; }}
        .header {{ background: #1e293b; padding: 2rem; border-bottom: 1px solid #334155; }}
        .container {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid #334155; }}
        h1, h2 {{ margin-top: 0; color: #38bdf8; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
        .stat-item {{ background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #334155; }}
        .stat-label {{ font-size: 0.875rem; color: #94a3b8; }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; color: #f1f5f9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; }}
        .dot {{ width: 12px; height: 12px; border-radius: 2px; }}
        svg {{ background: #0f172a; border-radius: 8px; width: 100%; height: auto; }}
        .micro-tag {{ color: #ef4444; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <div style="max-width: 1100px; margin: auto;">
            <p style="color: #94a3b8; margin: 0; font-size: 0.875rem;">Sentio Mind · POC Project 6</p>
            <h1>Emotion Transition Timeline</h1>
        </div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Duration</div>
                <div class="stat-value">{stats.get('duration_sec', 0)}s</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Dominant</div>
                <div class="stat-value" style="color: {EMOTION_COLORS.get(stats.get('dominant_emotion'), '#fff')}">{stats.get('dominant_emotion', 'neutral').capitalize()}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Suppression Score</div>
                <div class="stat-value">{stats.get('suppression_score', 0)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Emotional Range</div>
                <div class="stat-value">{stats.get('emotional_range_score', 0)}</div>
            </div>
        </div>

        <div class="card" style="margin-top: 2rem;">
            <h2>River Chart Timeline</h2>
            <div class="legend">
                {''.join([f'<div class="legend-item"><div class="dot" style="background:{EMOTION_COLORS[e]}"></div>{e.capitalize()}</div>' for e in EMOTIONS])}
            </div>
            <svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
                <!-- X Axis -->
                <line x1="{PAD}" y1="{PAD+CHART_H}" x2="{PAD+CHART_W}" y2="{PAD+CHART_H}" stroke="#334155" />
                <text x="{PAD}" y="{PAD+CHART_H+20}" fill="#94a3b8" font-size="12">0s</text>
                <text x="{PAD+CHART_W-20}" y="{PAD+CHART_H+20}" fill="#94a3b8" font-size="12">{max_t}s</text>
                
                <!-- Y Axis -->
                <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{PAD+CHART_H}" stroke="#334155" />
                <text x="{PAD-35}" y="{PAD+10}" fill="#94a3b8" font-size="12">100%</text>
                <text x="{PAD-25}" y="{PAD+CHART_H}" fill="#94a3b8" font-size="12">0%</text>

                {svg_paths_html}
                {markers_html}
            </svg>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-top: 1rem;">
                * Vertical red dashed lines indicate detected <span class="micro-tag">Micro-Expressions</span>.
            </p>
        </div>

        <div class="card">
            <h2>Transition Log</h2>
            <table>
                <thead>
                    <tr><th>From</th><th>To</th><th>Timestamp</th><th>Duration</th></tr>
                </thead>
                <tbody>
                    { ''.join([f"<tr><td>{t['from_emotion'].capitalize()}</td><td>{t['to_emotion'].capitalize()}</td><td>{t['timestamp_sec']}s</td><td>{t['transition_duration_sec']}s</td></tr>" for t in transitions]) if transitions else "<tr><td colspan='4' style='text-align:center; color:#94a3b8'>No major transitions detected</td></tr>" }
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>Micro-Expression Details</h2>
            <table>
                <thead>
                    <tr><th>ID</th><th>Emotion</th><th>Start</th><th>Duration</th><th>Peak Prob</th></tr>
                </thead>
                <tbody>
                    { ''.join([f"<tr><td>{m['id']}</td><td class='micro-tag'>{m['emotion'].upper()}</td><td>{m['timestamp_sec']}s</td><td>{m['duration_sec']}s</td><td>{m['peak_probability']}%</td></tr>" for m in micro_expressions]) if micro_expressions else "<tr><td colspan='5' style='text-align:center; color:#94a3b8'>No micro-expressions detected</td></tr>" }
                </tbody>
            </table>
        </div>
    </div>
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
    emo_time_pct  = {e: float(round(c / n * 100, 1)) for e, c in emo_counts.items()}
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
        json.dump(stats, f, indent=2)

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
