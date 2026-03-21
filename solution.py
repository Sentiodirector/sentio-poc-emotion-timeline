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
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path(r"E:\sentio_assignment\sentio-poc-emotion-timeline\Dataset_Assignment\Video_1\Class_8_cctv_video_1.mov")
PROFILES_DIR    = Path(r"E:\sentio_assignment\sentio-poc-emotion-timeline\Dataset_Assignment\Profiles_1")
REPORT_HTML_OUT = Path("emotion_timeline.html")
OUTPUT_JSON     = Path("emotion_timeline_output.json")
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANALYSIS_FPS            = 10      # frames per second to analyse
MICRO_MAX_SEC           = 0.5     # micro-expression must last less than this
MICRO_MIN_PROB          = 0.40    # peak probability to qualify
NEUTRAL_MIN             = 0.50    # min neutral probability to count as "neutral before/after"
TRANSITION_HOLD_SEC     = 0.5     # min duration for emotion to hold to be a transition
EXPRESSION_THRESH       = 35.0    # threshold (0-100) to consider a frame 'expressive'
DISTINCT_EMOTION_THRESH = 30.0    # threshold (0-100) for distinct emotion range score
FACE_MATCH_TOLERANCE    = 0.6     # face recognition tolerance factor

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

    Implementation:
      - Import DeepFace and pre-build the Emotion model so it is loaded into
        memory only once, preventing severe memory leaks during frame-by-frame
        video processing.
      - Initialize an MTCNN detector alongside the emotion model.
      - Return both as a dict bundled under "deepface" model_type.
    """
    try:
        from deepface import DeepFace
        from mtcnn import MTCNN

        # Pre-warm the DeepFace emotion model into memory (one-time load)
        # deepface 0.0.93 requires task="facial_attribute" for Emotion model
        DeepFace.build_model("Emotion", task="facial_attribute")

        # Initialize MTCNN face detector (one-time allocation)
        face_detector = MTCNN()

        model_bundle = {
            "deepface": DeepFace,
            "face_detector": face_detector,
        }
        return model_bundle, "deepface"
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

    For model_type == "deepface":
      Use DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
      enforce_detection=False is CRITICAL because MTCNN has already cropped
      the face — DeepFace would otherwise try to re-detect a face inside the
      crop and fail on tightly-cropped regions.

    For model_type == "ferplus":
      Resize to 48×48 grayscale, create blob, run net.forward()

    For model_type == "fallback":
      Return neutral=100, all others=0.
    """
    if model_type == "deepface":
        DeepFace = model["deepface"]
        try:
            result = DeepFace.analyze(
                face_crop,
                actions=["emotion"],
                enforce_detection=False,   # CRITICAL: face already cropped by MTCNN
                silent=True,
            )
            # DeepFace.analyze returns a list of dicts (one per face found)
            if isinstance(result, list):
                result = result[0]

            raw_emotions = result.get("emotion", {})

            # Map to our canonical 7 emotions, default 0 for any missing
            probs = {}
            for e in EMOTIONS:
                probs[e] = raw_emotions.get(e, 0.0)

            # Normalize to sum to exactly 100
            total = sum(probs.values())
            if total > 0:
                probs = {e: round(v / total * 100, 2) for e, v in probs.items()}
            else:
                probs = {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}

            # Fix rounding residual so sum == 100
            diff = 100.0 - sum(probs.values())
            if diff != 0:
                dominant = max(probs, key=probs.get)
                probs[dominant] = round(probs[dominant] + diff, 2)

            return probs

        except Exception:
            return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}

    elif model_type == "ferplus":
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (64, 64))
        model.setInput(blob)
        output = model.forward()[0]

        # FERPlus: 8 outputs (7 emotions + contempt) — drop contempt (index 7)
        scores = output[:7]
        # Softmax
        exp_s = np.exp(scores - np.max(scores))
        softmax = exp_s / exp_s.sum()
        probs = {EMOTIONS[i]: round(float(softmax[i]) * 100, 2) for i in range(7)}

        diff = 100.0 - sum(probs.values())
        if diff != 0:
            dominant = max(probs, key=probs.get)
            probs[dominant] = round(probs[dominant] + diff, 2)

        return probs

    else:
        # fallback
        return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    """
    Find the largest face in the frame using MTCNN.
    Return (face_crop_ndarray, (x, y, w, h)) or (None, None) if no face found.

    Implementation:
      - Convert BGR → RGB (MTCNN expects RGB input).
      - Run MTCNN detection.
      - If multiple faces, select the largest by bounding box area
        (assignment specifies close-up/medium shot of a single person).
      - Crop and return the face region as a BGR numpy array.
    """
    # Access the globally-loaded model bundle
    global _face_detector
    if _face_detector is None:
        return None, None

    # MTCNN expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = _face_detector.detect_faces(rgb_frame)

    if not detections:
        return None, None

    # Select the largest face by bounding box area
    best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = best["box"]

    # Clamp coordinates to frame boundaries (MTCNN can sometimes return negatives)
    fh, fw = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, fw - x)
    h = min(h, fh - y)

    if w <= 0 or h <= 0:
        return None, None

    face_crop = frame[y : y + h, x : x + w]
    return face_crop, (x, y, w, h)


# Global reference set during model loading
_face_detector = None


# ---------------------------------------------------------------------------
# MICRO-EXPRESSION DETECTION
# ---------------------------------------------------------------------------

def detect_micro_expressions(frame_series: list) -> list:
    """
    frame_series: [{"t": float, "emotions": dict}, ...]
    Scan for micro-expression events using the definition from README.

    Algorithm:
      Slide over frames. When a non-neutral emotion exceeds MICRO_MIN_PROB:
        - check that the previous frame had neutral >= NEUTRAL_MIN
        - count how many consecutive frames this emotion stays above threshold
        - if it returns to neutral (>= NEUTRAL_MIN) within MICRO_MAX_SEC -> micro-expression

    Return list of micro-expression dicts matching emotion_timeline.json schema.
    """
    micro = []
    n = len(frame_series)
    if n < 3:
        return micro

    # Config thresholds are 0-1; emotions are 0-100 percentages
    spike_thresh = MICRO_MIN_PROB * 100   # 40.0
    neutral_thresh = NEUTRAL_MIN * 100    # 50.0

    i = 1  # start at 1 so we can look back at i-1
    micro_id = 1

    while i < n - 1:
        prev_frame = frame_series[i - 1]
        curr_frame = frame_series[i]

        # Check: previous frame must be neutral-dominant (neutral >= threshold)
        if prev_frame["emotions"].get("neutral", 0) < neutral_thresh:
            i += 1
            continue

        # Check: current frame has a non-neutral emotion spiking above threshold
        spike_emotion = None
        spike_prob = 0.0
        for emo in EMOTIONS:
            if emo == "neutral":
                continue
            prob = curr_frame["emotions"].get(emo, 0)
            if prob >= spike_thresh and prob > spike_prob:
                spike_emotion = emo
                spike_prob = prob

        if spike_emotion is None:
            i += 1
            continue

        # Found a spike — trace how long it lasts
        start_t = curr_frame["t"]
        peak_prob = spike_prob
        j = i + 1

        while j < n:
            emo_val = frame_series[j]["emotions"].get(spike_emotion, 0)
            if emo_val >= spike_thresh:
                peak_prob = max(peak_prob, emo_val)
                j += 1
            else:
                break

        # j is now the first frame AFTER the spike ends
        if j >= n:
            # Spike runs to end of video — no return to neutral
            i = j
            continue

        end_t = frame_series[j]["t"]
        duration = end_t - start_t

        # Duration must be < MICRO_MAX_SEC (0.5s)
        if duration >= MICRO_MAX_SEC:
            i = j
            continue

        # The frame after the spike must return to neutral
        after_frame = frame_series[j]
        if after_frame["emotions"].get("neutral", 0) < neutral_thresh:
            i = j
            continue

        # Determine what emotion follows
        followed_by = max(after_frame["emotions"], key=after_frame["emotions"].get)

        micro.append({
            "id": micro_id,
            "timestamp_sec": round(start_t, 3),
            "duration_sec": round(duration, 3),
            "emotion": spike_emotion,
            "peak_probability": round(peak_prob, 2),
            "followed_by": followed_by,
            "is_suppressed": True,  # returning to neutral implies suppression
        })
        micro_id += 1
        i = j + 1  # skip past this micro-expression

    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    """
    A transition occurs when the dominant emotion changes and the new one
    holds for at least 0.5 seconds.

    Return list of transition dicts matching emotion_timeline.json schema.
    """
    transitions = []
    n = len(frame_series)
    if n < 2:
        return transitions

    # Compute dominant emotion per frame
    dominants = []
    for fs in frame_series:
        dom = max(fs["emotions"], key=fs["emotions"].get)
        dominants.append(dom)

    # Walk through frames tracking stable emotion segments
    current_emotion = dominants[0]
    segment_start_t = frame_series[0]["t"]

    i = 1
    while i < n:
        if dominants[i] != current_emotion:
            # Potential transition — check if the new emotion holds for >= 0.5s
            new_emotion = dominants[i]
            transition_t = frame_series[i]["t"]
            j = i + 1

            while j < n and dominants[j] == new_emotion:
                j += 1

            # Duration of the new emotion segment
            if j < n:
                hold_duration = frame_series[j]["t"] - transition_t
            else:
                hold_duration = frame_series[n - 1]["t"] - transition_t

            if hold_duration >= TRANSITION_HOLD_SEC:
                # Genuine transition
                transition_duration = transition_t - segment_start_t
                transitions.append({
                    "from_emotion": current_emotion,
                    "to_emotion": new_emotion,
                    "timestamp_sec": round(transition_t, 3),
                    "transition_duration_sec": round(
                        frame_series[i]["t"] - frame_series[i - 1]["t"], 3
                    ),
                })
                current_emotion = new_emotion
                segment_start_t = transition_t
                i = j
            else:
                # Flicker — skip past it
                i = j
        else:
            i += 1

    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    """
    suppression_score = (len(micro_expressions) / total_expression_events) * 100
    total_expression_events = frames where any non-neutral emotion > 35.0

    Return int 0-100.

    Heuristic:
      - Count "expression events" as frames where any non-neutral emotion
        exceeds 35% (0.35 on 0-1 scale = 35.0 on 0-100 scale).
      - Ratio = micro_expression_count / expression_event_count.
      - If no expression events exist, fall back to a frequency-per-minute
        heuristic: (micro_count / duration_minutes) * 10, capped at 100.
    """
    n_micro = len(micro_expressions)
    if n_micro == 0:
        return 0

    # Count frames with expressive (non-neutral) content above EXPRESSION_THRESH
    total_expression_events = 0
    for fs in frame_series:
        for emo in EMOTIONS:
            if emo == "neutral":
                continue
            if fs["emotions"].get(emo, 0) > EXPRESSION_THRESH:
                total_expression_events += 1
                break  # count each frame only once

    if total_expression_events > 0:
        score = (n_micro / total_expression_events) * 100
    else:
        # Fallback: frequency-based heuristic
        if len(frame_series) >= 2:
            duration_min = (frame_series[-1]["t"] - frame_series[0]["t"]) / 60.0
        else:
            duration_min = 1.0
        duration_min = max(duration_min, 1 / 60)  # avoid division by zero
        score = (n_micro / duration_min) * 10

    return min(100, max(0, int(round(score))))


def compute_emotional_range(frame_series: list) -> int:
    """
    distinct = count of EMOTIONS that appeared with prob > 30.0 at least once
    range_score = min(100, (distinct / 7) * 100 + std_of_all_probs * 2)

    Return int 0-100.
    """
    if not frame_series:
        return 0

    distinct_emotions = set()
    all_probs = []

    for fs in frame_series:
        for emo in EMOTIONS:
            if emo == "neutral":
                continue
            prob = fs["emotions"].get(emo, 0)
            all_probs.append(prob)
            if prob > DISTINCT_EMOTION_THRESH:
                distinct_emotions.add(emo)

    distinct_count = len(distinct_emotions)
    base_score = (distinct_count / 7) * 100

    # Calculate standard deviation of all non-neutral probabilities to gauge variance
    if all_probs:
        std_dev = np.std(all_probs)
    else:
        std_dev = 0.0

    range_score = base_score + (std_dev * 2.0)
    return min(100, max(0, int(round(range_score))))


# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(frame_series: list, micro_expressions: list,
                                    transitions: list, stats: dict,
                                    output_path: Path):
    """
    Write emotion_timeline.html — self-contained, offline, no CDN.
    """
    import urllib.request
    import json

    print("Fetching Chart.js dependencies for offline mode...")
    chartjs_code = urllib.request.urlopen("https://cdn.jsdelivr.net/npm/chart.js").read().decode('utf-8')
    annotation_code = urllib.request.urlopen("https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation").read().decode('utf-8')

    # Prepare datasets
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    labels = [fs["t"] for fs in frame_series]
    
    datasets = []
    for emo in emotions:
        data = [fs["emotions"].get(emo, 0.0) for fs in frame_series]
        # Add slight transparency to the fill color
        color_hex = EMOTION_COLORS.get(emo, "#000000")
        datasets.append({
            "label": emo.capitalize(),
            "data": data,
            "borderColor": color_hex,
            "backgroundColor": color_hex + "80", 
            "fill": True,
            "tension": 0.4
        })

    # Prepare annotations for Micro-Expressions
    annotations = {}
    for idx, me in enumerate(micro_expressions):
        t_val = me.get("timestamp_sec", 0.0)
        annotations[f"line{idx}"] = {
            "type": "line",
            "xMin": t_val,
            "xMax": t_val,
            "borderColor": "#ffffff",
            "borderWidth": 2,
            "borderDash": [5, 5],
            "label": {
                "display": True,
                "content": f"ME: {me.get('emotion')}",
                "position": "end",
                "backgroundColor": "rgba(255, 0, 0, 0.8)",
                "color": "#fff"
            }
        }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Timeline</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        h1 {{ margin: 0; padding: 0; font-size: 28px; font-weight: 600; color: #38bdf8; }}
        p {{ color: #cbd5e1; font-size: 14px; margin-top: 8px; }}
        .chart-container {{ position: relative; height: 75vh; width: 100%; max-width: 1400px; margin: 0 auto; background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    </style>
    <script>{chartjs_code}</script>
    <script>{annotation_code}</script>
</head>
<body>
    <div class="header">
        <h1>Micro-Expression & Emotion Transition Rivera</h1>
        <p>Subject: Unknown | Dominant: {stats.get("dominant_emotion", "Unknown")} | MEs Detected: {len(micro_expressions)}</p>
    </div>
    <div class="chart-container">
        <canvas id="emotionChart"></canvas>
    </div>

    <script>
        const ctx = document.getElementById('emotionChart').getContext('2d');
        const data = {{
            labels: {json.dumps(labels)},
            datasets: {json.dumps(datasets)}
        }};

        const config = {{
            type: 'line',
            data: data,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    annotation: {{
                        annotations: {json.dumps(annotations)}
                    }},
                    tooltip: {{
                        position: 'nearest'
                    }},
                    legend: {{
                        labels: {{ color: '#f8fafc' }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'category',
                        title: {{
                            display: true,
                            text: 'Time (Seconds)',
                            color: '#cbd5e1'
                        }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    y: {{
                        stacked: true,
                        title: {{
                            display: true,
                            text: 'Probability (%)',
                            color: '#cbd5e1'
                        }},
                        min: 0,
                        max: 100,
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }};

        new Chart(ctx, config);
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated offline HTML report: {output_path}")


# ---------------------------------------------------------------------------
# SUBJECT IDENTIFICATION
# ---------------------------------------------------------------------------

def identify_subject(profiles_dir: Path, first_video_frame: np.ndarray) -> str:
    """
    Compare the face in the first video frame against profile images.
    Returns the filename (without extension) of the matched profile.
    """
    import face_recognition
    import os

    # 1. Get face encoding from the first video frame
    rgb_frame = cv2.cvtColor(first_video_frame, cv2.COLOR_BGR2RGB)
    video_encodings = face_recognition.face_encodings(rgb_frame)
    
    if not video_encodings:
        print("WARNING: No face found in the first frame for identification.")
        return "Unknown"
    
    video_encoding = video_encodings[0]

    # 2. Iterate through profiles
    print(f"Identifying subject against profiles in {profiles_dir} ...")
    for filename in os.listdir(profiles_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        filepath = os.path.join(profiles_dir, filename)
        profile_img = face_recognition.load_image_file(filepath)
        profile_encodings = face_recognition.face_encodings(profile_img)
        
        if not profile_encodings:
            continue
            
        profile_encoding = profile_encodings[0]
        match = face_recognition.compare_faces([profile_encoding], video_encoding, tolerance=FACE_MATCH_TOLERANCE)
        
        if match[0]:
            matched_id = os.path.splitext(filename)[0]
            print(f"[ MATCH ] found: {matched_id}")
            return matched_id

    print("[ NO MATCH ] No matching profile found.")
    return "Unknown"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 1. Identify Subject First ---
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read video.")
        sys.exit(1)
        
    subject_id = identify_subject(PROFILES_DIR, first_frame)
    print(f"Subject Identified: {subject_id}")
    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 2. Initialize Models ---
    model, model_type = load_emotion_model()
    print(f"Emotion model: {model_type}")

    # Store the face detector globally so detect_primary_face can use it
    if model_type == "deepface" and isinstance(model, dict):
        _face_detector = model["face_detector"]

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur    = total / fps_in

    sample_every = max(1, int(fps_in / ANALYSIS_FPS))
    frame_series = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % int(fps_in * 2) == 0:
                print(f"Processing frame {frame_idx}/{total}...")
                
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
    finally:
        cap.release()
        cv2.destroyAllWindows()

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
        json.dump(stats, f, indent=2)

    generate_emotion_timeline_html(frame_series, micro_expressions, transitions, stats, REPORT_HTML_OUT)

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
