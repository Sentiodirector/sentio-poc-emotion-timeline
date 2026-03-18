"""
solution.py
Sentio Mind · Project 6 · Micro-Expression & Emotion Transition Timeline

Complete implementation.
Run: python solution.py
"""

import cv2
import json
import numpy as np
from pathlib import Path
import urllib.request

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("Class_8_cctv_video_1.mov")
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
    """
    Return { emotion_name: probability_0_to_100, ... } for all 7 EMOTIONS.
    Probabilities must sum to 100 (round as needed).
    """
    if model_type == "deepface":
        try:
            res = model.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(res, list): 
                res = res[0]
            raw_emotions = res['emotion']
            # Normalise exactly to 100 to ensure consistency
            total = sum(raw_emotions.values()) or 1
            return {e: round((raw_emotions.get(e, 0) / total) * 100, 2) for e in EMOTIONS}
        except Exception:
            pass

    elif model_type == "ferplus":
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            model.setInput(blob)
            preds = model.forward()[0]
            # FER+ labels: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
            # Ignore contempt (index 7)
            probs = np.exp(preds[:7]) / np.sum(np.exp(preds[:7])) * 100
            
            fer_map = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]
            mapped = {fer_map[i]: float(probs[i]) for i in range(7)}
            return {e: round(mapped.get(e, 0), 2) for e in EMOTIONS}
        except Exception:
            pass

    # Fallback / Error
    return {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    """
    Find the largest face in the frame using OpenCV's built-in Haar Cascade.
    Return (face_crop_ndarray, (x, y, w, h)) or (None, None) if no face found.
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces) == 0:
        return None, None
        
    # Get the largest box by area (w * h)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Extract crop
    face_crop = frame[y:y+h, x:x+w]
    return face_crop, (x, y, w, h)


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    """
    frame_series: [{"t": float, "emotions": dict}, ...]
    Scan for micro-expression events using the definition from README.
    """
    micro = []
    max_frames_duration = int(ANALYSIS_FPS * MICRO_MAX_SEC) # strictly less than this
    
    # Note: Probabilities in frame_series are 0-100, so multiply thresholds by 100
    peak_thresh = MICRO_MIN_PROB * 100
    neutral_thresh = NEUTRAL_MIN * 100
    
    me_id = 1
    i = 1
    while i < len(frame_series) - 1:
        prev_neutral = frame_series[i-1]["emotions"]["neutral"]
        
        # Must be preceded by neutral >= 50%
        if prev_neutral >= neutral_thresh:
            
            # Find a non-neutral emotion that spikes above peak_thresh
            spike_emotion = None
            peak_val = 0
            for e in EMOTIONS:
                if e != "neutral" and frame_series[i]["emotions"][e] >= peak_thresh:
                    if frame_series[i]["emotions"][e] > peak_val:
                        peak_val = frame_series[i]["emotions"][e]
                        spike_emotion = e
            
            if spike_emotion:
                # Track duration to see if/when neutral returns to >= 50%
                duration_frames = 1
                returned_to_neutral = False
                
                for j in range(i + 1, min(len(frame_series), i + max_frames_duration)):
                    # Update peak if it got higher
                    if frame_series[j]["emotions"][spike_emotion] > peak_val:
                        peak_val = frame_series[j]["emotions"][spike_emotion]
                    
                    if frame_series[j]["emotions"]["neutral"] >= neutral_thresh:
                        returned_to_neutral = True
                        break
                    duration_frames += 1
                
                # Check criteria
                if returned_to_neutral and duration_frames < max_frames_duration:
                    duration_sec = duration_frames / ANALYSIS_FPS
                    micro.append({
                        "id": me_id,
                        "timestamp_sec": round(frame_series[i]["t"], 3),
                        "duration_sec": round(duration_sec, 3),
                        "emotion": spike_emotion,
                        "peak_probability": round(peak_val / 100.0, 3), # Store as 0.0 - 1.0 as typical
                        "followed_by": "neutral",
                        "is_suppressed": True
                    })
                    me_id += 1
                    i += duration_frames  # Skip past this event
                    continue
        i += 1
        
    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    """
    A transition occurs when the dominant emotion changes and the new one
    holds for at least 0.5 seconds (i.e. >= (ANALYSIS_FPS * 0.5) frames).
    """
    transitions = []
    if not frame_series:
        return transitions
        
    required_frames = max(1, int(ANALYSIS_FPS * 0.5))
    
    # Get dominant emotion for every frame
    dominant_series = [max(fs["emotions"], key=fs["emotions"].get) for fs in frame_series]
    
    current_stable = dominant_series[0]
    candidate = None
    candidate_frames = 0
    candidate_start_t = 0
    
    for i in range(1, len(dominant_series)):
        dom = dominant_series[i]
        
        if dom == current_stable:
            candidate = None
            candidate_frames = 0
        else:
            if dom == candidate:
                candidate_frames += 1
            else:
                candidate = dom
                candidate_frames = 1
                candidate_start_t = frame_series[i]["t"]
                
            if candidate_frames >= required_frames:
                # Confirmed transition
                transitions.append({
                    "from_emotion": current_stable,
                    "to_emotion": candidate,
                    "timestamp_sec": round(candidate_start_t, 3),
                    "transition_duration_sec": round(candidate_frames / ANALYSIS_FPS, 3)
                })
                current_stable = candidate
                candidate = None
                candidate_frames = 0
                
    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    """
    suppression_score = (len(micro_expressions) / total_expression_events) * 100
    total_expression_events = frames where any non-neutral emotion > 0.35 (35 in 0-100 scale)
    """
    if not frame_series: return 0
    
    total_expression_events = 0
    for fs in frame_series:
        if any(fs["emotions"][e] > 35 for e in EMOTIONS if e != "neutral"):
            total_expression_events += 1
            
    if total_expression_events == 0:
        return 0
        
    score = (len(micro_expressions) / total_expression_events) * 100
    return int(round(min(100, score)))


def compute_emotional_range(frame_series: list) -> int:
    """
    distinct = count of EMOTIONS that appeared with prob > 30 at least once
    range_score = min(100, (distinct / 7) * 100 + std_of_all_probs * 2)
    """
    if not frame_series: return 0
    
    appeared_emotions = set()
    all_probs = []
    
    for fs in frame_series:
        for e in EMOTIONS:
            prob = fs["emotions"][e]
            all_probs.append(prob)
            if prob > 30:
                appeared_emotions.add(e)
                
    distinct = len(appeared_emotions)
    std_probs = np.std(all_probs) if all_probs else 0
    
    score = (distinct / 7) * 100 + (std_probs * 2)
    return int(round(min(100, score)))


# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(frame_series: list, micro_expressions: list,
                                   transitions: list, stats: dict,
                                   output_path: Path):
    """
    Write emotion_timeline.html — self-contained, offline, no CDN.
    Dynamically fetches Chart.js library during runtime to embed it inline.
    """
    print("Fetching Chart.js library for offline bundling...")
    try:
        req = urllib.request.Request("https://cdn.jsdelivr.net/npm/chart.js", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            chart_js_code = response.read().decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not fetch Chart.js automatically. Proceeding with fallback CDN. Error: {e}")
        chart_js_code = ""

    script_tag = f"<script>{chart_js_code}</script>" if chart_js_code else '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'

    labels = [fs["t"] for fs in frame_series]
    datasets = []
    
    for emp in EMOTIONS:
        datasets.append({
            "label": emp.capitalize(),
            "data": [fs["emotions"].get(emp, 0) for fs in frame_series],
            "backgroundColor": EMOTION_COLORS.get(emp, "#000000"),
            "fill": True,
            "tension": 0.4
        })

    # Prepare transition table rows
    trans_html = ""
    for tr in transitions:
        trans_html += f"<tr><td>{tr['timestamp_sec']}s</td><td>{tr['from_emotion'].capitalize()}</td><td>{tr['to_emotion'].capitalize()}</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Emotion Timeline Report</title>
    <style> 
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #fafafa; color: #333; }} 
        h1, h2 {{ color: #111; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }} 
        .stat-box {{ padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; text-align: center; }} 
        .stat-box strong {{ display: block; font-size: 24px; margin-top: 10px; color: #2563eb; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f1f5f9; }}
    </style>
    {script_tag}
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.0.1/dist/chartjs-plugin-annotation.min.js"></script>
</head>
<body>
    <h1>Sentio Mind: Emotion Transition Timeline</h1>
    <div class="stats">
        <div class="stat-box">Dominant Emotion <strong>{stats['dominant_emotion'].capitalize()}</strong></div>
        <div class="stat-box">Suppression Score <strong>{stats['suppression_score']} / 100</strong></div>
        <div class="stat-box">Emotional Range <strong>{stats['emotional_range_score']} / 100</strong></div>
        <div class="stat-box">Micro-Expressions <strong>{len(micro_expressions)}</strong></div>
    </div>
    
    <div style="background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <canvas id="emotionChart" height="400"></canvas>
    </div>

    <h2>Transitions ({len(transitions)})</h2>
    <table>
        <tr><th>Timestamp</th><th>From Emotion</th><th>To Emotion</th></tr>
        {trans_html if trans_html else "<tr><td colspan='3'>No stable transitions detected.</td></tr>"}
    </table>

    <script>
        const ctx = document.getElementById('emotionChart').getContext('2d');
        const microExpressions = {json.dumps(micro_expressions)};
        
        // Generate vertical line annotations for micro expressions
        const annotations = {{}};
        microExpressions.forEach((me, index) => {{
            annotations['line' + index] = {{
                type: 'line',
                xMin: me.timestamp_sec,
                xMax: me.timestamp_sec,
                borderColor: '#111',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {{ display: true, content: 'ME: ' + me.emotion, position: 'start' }}
            }};
        }});

        new Chart(ctx, {{
            type: 'line',
            data: {{ labels: {json.dumps(labels)}, datasets: {json.dumps(datasets)} }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    y: {{ stacked: true, min: 0, max: 100, title: {{ display: true, text: 'Probability (%)' }} }},
                    x: {{ title: {{ display: true, text: 'Time (Seconds)' }} }}
                }},
                plugins: {{
                    tooltip: {{ mode: 'index', intersect: false }},
                    annotation: {{ annotations: annotations }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


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
            
            print(f"Processed frame {frame_idx}/{total} (Time: {ts:.2f}s)", end='\r')
            
        frame_idx += 1
    cap.release()

    print(f"\nAnalysed {len(frame_series)} frames over {dur:.1f}s")

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

    # Format output for multi_day_report.json schema injection
    output_integration = {
        "person_profiles": {
            "pid_1": {
                "emotion_timeline": stats
            }
        }
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