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

    For model_type == "ferplus":
      Resize to 48×48 grayscale, create blob, run net.forward()
      The model returns 8 values (7 emotions + contempt) — drop contempt, normalise to 100.

    For model_type == "fallback":
      Return neutral=100, all others=0.

    TODO: implement each branch
    """
    # TODO
    return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# FACE DETECTION — PRIMARY FACE IN FRAME
# ---------------------------------------------------------------------------

def detect_primary_face(frame: np.ndarray):
    """
    Find the largest face in the frame.
    Return (face_crop_ndarray, (x, y, w, h)) or (None, None) if no face found.
    Use Haar cascade or MediaPipe Face Detection.
    TODO: implement
    """
    # TODO
    return None, None


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
        - if it returns to neutral (>= NEUTRAL_MIN) within MICRO_MAX_SEC → it's a micro-expression

    Return list of micro-expression dicts matching emotion_timeline.json schema.
    TODO: implement
    """
    micro = []
    # TODO
    return micro


# ---------------------------------------------------------------------------
# EMOTION TRANSITIONS
# ---------------------------------------------------------------------------

def detect_transitions(frame_series: list) -> list:
    """
    A transition occurs when the dominant emotion changes and the new one
    holds for at least 0.5 seconds.

    Return list of transition dicts matching emotion_timeline.json schema.
    TODO: implement
    """
    transitions = []
    # TODO
    return transitions


# ---------------------------------------------------------------------------
# SUPPRESSION & RANGE SCORES
# ---------------------------------------------------------------------------

def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    """
    suppression_score = (len(micro_expressions) / total_expression_events) * 100
    total_expression_events = frames where any non-neutral emotion > 0.35

    Return int 0–100.
    TODO: implement
    """
    # TODO
    return 0


def compute_emotional_range(frame_series: list) -> int:
    """
    distinct = count of EMOTIONS that appeared with prob > 30 at least once
    range_score = min(100, (distinct / 7) * 100 + std_of_all_probs * 2)

    Return int 0–100.
    TODO: implement
    """
    # TODO
    return 50


# ---------------------------------------------------------------------------
# HTML RIVER CHART
# ---------------------------------------------------------------------------

def generate_emotion_timeline_html(frame_series: list, micro_expressions: list,
                                    transitions: list, stats: dict,
                                    output_path: Path):
    """
    Write emotion_timeline.html — self-contained, offline, no CDN.

    Must include:
      1. Stacked area river chart (Chart.js bundled inline or D3.js inline)
         7 emotion bands, X = seconds, Y = stacked probability 0–100
         Vertical dashed lines at each micro-expression event
      2. Stat summary: dominant emotion, time-in-state table, suppression score, range score
      3. Transitions table: from → to, timestamp, duration

    To bundle Chart.js offline: download chart.umd.min.js and paste between <script> tags.
    TODO: implement
    """
    # TODO
    pass


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
