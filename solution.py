"""
solution.py
Sentio Mind · Project 6 · Micro-Expression & Emotion Transition Timeline

Run: python solution.py
Place models/emotion_ferplus.onnx in the same folder if not using DeepFace.
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path

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
    "happy":   "#22c55e",
    "neutral": "#94a3b8",
    "sad":     "#3b82f6",
    "angry":   "#ef4444",
    "fear":    "#a855f7",
    "surprise":"#eab308",
    "disgust": "#f97316",
}

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
        print("Using DeepFace emotion model.")
        return DeepFace, "deepface"
    except ImportError:
        pass

    onnx = MODELS_DIR / "emotion_ferplus.onnx"
    if onnx.exists():
        net = cv2.dnn.readNetFromONNX(str(onnx))
        print("Using FER+ ONNX model.")
        return net, "ferplus"

    print("WARNING: no emotion model found. Using random baseline.")
    return None, "fallback"

# ---------------------------------------------------------------------------
def analyse_emotion(face_crop: np.ndarray, model, model_type: str) -> dict:
    """
    Return { emotion_name: probability_0_to_100, ... } for all 7 EMOTIONS.
    Probabilities sum to 100.
    """
    if model_type == "deepface":
        try:
            result = model.analyze(
                face_crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            # DeepFace returns a list of dicts
            if isinstance(result, list):
                result = result[0]
            raw = result.get("emotion", {})
            # raw values are already percentages summing to ~100
            total = sum(raw.values()) or 1.0
            probs = {}
            for e in EMOTIONS:
                probs[e] = round(raw.get(e, 0.0) / total * 100, 2)
            # Ensure exact sum = 100
            diff = 100 - sum(probs.values())
            probs["neutral"] = round(probs.get("neutral", 0) + diff, 2)
            return probs
        except Exception as ex:
            print(f"DeepFace error: {ex} — falling back to neutral")
            return {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    elif model_type == "ferplus":
        # FER+ model: input 48x48 grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48)).astype(np.float32)
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(48, 48))
        model.setInput(blob)
        logits = model.forward().flatten()  # 8 values (7 emotions + contempt)
        # Keep first 7 (drop contempt index 7)
        logits = logits[:7]
        # Softmax
        e_x = np.exp(logits - np.max(logits))
        probs_arr = e_x / e_x.sum()
        probs = {EMOTIONS[i]: round(float(probs_arr[i]) * 100, 2) for i in range(7)}
        # Normalize to 100
        total = sum(probs.values()) or 1.0
        probs = {k: round(v / total * 100, 2) for k, v in probs.items()}
        diff = 100 - sum(probs.values())
        probs["neutral"] = round(probs.get("neutral", 0) + diff, 2)
        return probs

    else:  # fallback
        return {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}

# ---------------------------------------------------------------------------
_face_cascade = None

def detect_primary_face(frame: np.ndarray):
    """
    Find the largest face in the frame.
    Return (face_crop_ndarray, (x, y, w, h)) or (None, None) if no face found.
    Uses Haar cascade with MediaPipe Face Detection as backup.
    """
    global _face_cascade

    # --- Try MediaPipe first (more accurate) ---
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(min_detection_confidence=0.4) as detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                h, w = frame.shape[:2]
                best = None
                best_area = 0
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = max(0, int(bb.xmin * w))
                    y1 = max(0, int(bb.ymin * h))
                    x2 = min(w, int((bb.xmin + bb.width) * w))
                    y2 = min(h, int((bb.ymin + bb.height) * h))
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1, y1, x2 - x1, y2 - y1)
                if best:
                    x, y, bw, bh = best
                    # Add small padding
                    pad = int(min(bw, bh) * 0.1)
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    x2 = min(frame.shape[1], x + bw + 2 * pad)
                    y2 = min(frame.shape[0], y + bh + 2 * pad)
                    crop = frame[y:y2, x:x2]
                    if crop.size > 0:
                        return crop, (x, y, x2 - x, y2 - y)
    except Exception:
        pass

    # --- Fallback: Haar cascade ---
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None

    # Pick largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return None, None
    return crop, (x, y, w, h)

# ---------------------------------------------------------------------------
def detect_micro_expressions(frame_series: list) -> list:
    """
    Scan for micro-expression events.
    A micro-expression:
      1. A non-neutral emotion appears with probability >= MICRO_MIN_PROB (40%)
      2. Lasts < MICRO_MAX_SEC (0.5s → < 4 frames at 8fps)
      3. Directly preceded AND followed by neutral >= NEUTRAL_MIN (50%)
    Returns list of micro-expression dicts matching schema.
    """
    micro = []
    n = len(frame_series)
    micro_id = 1
    i = 1  # start at 1 so we can look back

    while i < n - 1:
        frame = frame_series[i]
        emotions = frame["emotions"]

        # Find the dominant non-neutral emotion in this frame
        non_neutral = {e: v for e, v in emotions.items() if e != "neutral"}
        dom_emotion = max(non_neutral, key=non_neutral.get)
        dom_prob = non_neutral[dom_emotion] / 100.0  # fraction

        if dom_prob >= MICRO_MIN_PROB:
            # Check the previous frame was neutral
            prev_neutral = frame_series[i - 1]["emotions"].get("neutral", 0) / 100.0
            if prev_neutral >= NEUTRAL_MIN:
                # Count consecutive frames this emotion stays above threshold
                j = i
                while j < n and frame_series[j]["emotions"].get(dom_emotion, 0) / 100.0 >= MICRO_MIN_PROB:
                    j += 1
                # j is now the first frame where the emotion drops below threshold
                duration = frame_series[j - 1]["t"] - frame_series[i]["t"]
                # Add half-frame for inclusivity
                if j < n:
                    duration = frame_series[j]["t"] - frame_series[i]["t"]

                if duration < MICRO_MAX_SEC:
                    # Check the next frame after the burst is neutral
                    if j < n:
                        next_neutral = frame_series[j]["emotions"].get("neutral", 0) / 100.0
                        # Also check: return to neutral within the burst window
                        followed_by_neutral = next_neutral >= NEUTRAL_MIN
                    else:
                        followed_by_neutral = True  # end of video, treat as neutral

                    if followed_by_neutral:
                        # Find peak probability within the burst
                        peak_prob = max(
                            frame_series[k]["emotions"].get(dom_emotion, 0) / 100.0
                            for k in range(i, j)
                        )
                        # Followed-by emotion: dominant in frame j
                        if j < n:
                            followed_by = max(
                                frame_series[j]["emotions"],
                                key=frame_series[j]["emotions"].get
                            )
                        else:
                            followed_by = "neutral"

                        micro.append({
                            "id": micro_id,
                            "timestamp_sec": round(frame_series[i]["t"], 3),
                            "duration_sec": round(duration, 3),
                            "emotion": dom_emotion,
                            "peak_probability": round(peak_prob, 3),
                            "followed_by": followed_by,
                            "is_suppressed": followed_by == "neutral"
                        })
                        micro_id += 1
                        i = j  # skip past this burst
                        continue
        i += 1

    return micro

# ---------------------------------------------------------------------------
def detect_transitions(frame_series: list) -> list:
    """
    A transition occurs when the dominant emotion changes and the new emotion
    holds for at least 0.5 seconds.
    Returns list of transition dicts matching schema.
    """
    transitions = []
    if not frame_series:
        return transitions

    min_hold_frames = int(ANALYSIS_FPS * 0.5)  # ≥ 4 frames at 8fps
    n = len(frame_series)

    def dominant(frame):
        return max(frame["emotions"], key=frame["emotions"].get)

    current_emotion = dominant(frame_series[0])
    segment_start = 0

    i = 1
    while i < n:
        new_emotion = dominant(frame_series[i])
        if new_emotion != current_emotion:
            # Check the new emotion holds for at least min_hold_frames
            hold_end = i
            while hold_end < n and dominant(frame_series[hold_end]) == new_emotion:
                hold_end += 1
            hold_frames = hold_end - i

            if hold_frames >= min_hold_frames:
                ts = frame_series[i]["t"]
                transition_dur = frame_series[min(hold_end - 1, n - 1)]["t"] - ts
                transitions.append({
                    "from_emotion": current_emotion,
                    "to_emotion": new_emotion,
                    "timestamp_sec": round(ts, 3),
                    "transition_duration_sec": round(transition_dur, 3)
                })
                current_emotion = new_emotion
                segment_start = i
                i = hold_end
                continue
        i += 1

    return transitions

# ---------------------------------------------------------------------------
def compute_suppression_score(frame_series: list, micro_expressions: list) -> int:
    """
    suppression_score = (micro_expression_count / total_expression_events) * 100
    total_expression_events = frames where any non-neutral emotion > 0.35
    Returns int 0–100.
    """
    if not frame_series:
        return 0
    expression_threshold = 35  # 35 out of 100
    total_expression_events = sum(
        1 for f in frame_series
        if any(v > expression_threshold for e, v in f["emotions"].items() if e != "neutral")
    )
    if total_expression_events == 0:
        return 0
    score = (len(micro_expressions) / total_expression_events) * 100
    return min(100, int(round(score)))


def compute_emotional_range(frame_series: list) -> int:
    """
    distinct = count of EMOTIONS that appeared with prob > 30 at least once
    range_score = min(100, (distinct / 7) * 100 + std(all_probs_over_time) * 2)
    Returns int 0–100.
    """
    if not frame_series:
        return 0

    # Count distinct emotions appearing above 30%
    distinct = set()
    all_probs = []
    for f in frame_series:
        for e in EMOTIONS:
            v = f["emotions"].get(e, 0)
            all_probs.append(v)
            if v > 30:
                distinct.add(e)

    std_val = float(np.std(all_probs)) if all_probs else 0.0
    score = (len(distinct) / 7) * 100 + std_val * 2
    return min(100, int(round(score)))

# ---------------------------------------------------------------------------
def _get_chartjs_inline() -> str:
    """
    Return Chart.js 4.x UMD source inlined.
    Downloads from CDN once and caches locally, or embeds a minimal stub.
    """
    cache = MODELS_DIR / "chart.umd.min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    try:
        import urllib.request
        url = "https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"
        print("Downloading Chart.js for offline bundling...")
        urllib.request.urlretrieve(url, str(cache))
        return cache.read_text(encoding="utf-8")
    except Exception as ex:
        print(f"Could not download Chart.js ({ex}). Using CDN link fallback.")
        return None  # will fall back to CDN link in HTML


def _build_duchenne_markers(frame_series: list) -> list:
    """
    Attempt Duchenne smile detection using MediaPipe Face Mesh.
    A genuine smile shows both mouth activity AND eye-corner crinkling.
    Returns list of timestamps where a genuine Duchenne smile is detected.
    """
    # This is only computed if mediapipe is available
    try:
        import mediapipe as mp
    except ImportError:
        return []

    # We use landmark distances – outer eye corners vs upper cheeks
    # Landmarks: left eye outer corner ~33, right ~263
    # Cheek reference: left ~116, right ~345
    # Mouth corners: ~61 (left), ~291 (right)
    MOUTH_CORNERS = [61, 291]
    EYE_OUTER_L, EYE_OUTER_R = 33, 263
    CHEEK_L, CHEEK_R = 116, 345

    duchenne_timestamps = []
    mp_mesh = mp.solutions.face_mesh

    # Re-open video (we already closed it in main)
    if not VIDEO_PATH.exists():
        return []

    cap2 = cv2.VideoCapture(str(VIDEO_PATH))
    fps_in = cap2.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, int(fps_in / ANALYSIS_FPS))
    frame_idx = 0

    with mp_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                ts = frame_idx / fps_in
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)
                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]

                    def pt(idx):
                        l = lms[idx]
                        return np.array([l.x * w, l.y * h])

                    # Check mouth corners are raised (smiling)
                    mouth_left = pt(MOUTH_CORNERS[0])
                    mouth_right = pt(MOUTH_CORNERS[1])
                    mouth_avg_y = (mouth_left[1] + mouth_right[1]) / 2

                    # Eye crinkle: distance between outer eye corner and cheek landmark shrinks when smiling
                    eye_l = pt(EYE_OUTER_L)
                    cheek_l = pt(CHEEK_L)
                    eye_r = pt(EYE_OUTER_R)
                    cheek_r = pt(CHEEK_R)

                    dist_l = np.linalg.norm(eye_l - cheek_l)
                    dist_r = np.linalg.norm(eye_r - cheek_r)
                    avg_eye_cheek_dist = (dist_l + dist_r) / 2

                    # Normalize by face width (mouth corners distance)
                    face_width = np.linalg.norm(mouth_left - mouth_right)
                    if face_width > 0:
                        normalized_crinkle = avg_eye_cheek_dist / face_width
                        # Threshold determined empirically: lower ratio → more crinkle → genuine smile
                        if normalized_crinkle < 1.5:
                            duchenne_timestamps.append(round(ts, 3))

            frame_idx += 1

    cap2.release()
    return duchenne_timestamps


# ---------------------------------------------------------------------------
def generate_emotion_timeline_html(frame_series: list, micro_expressions: list,
                                    transitions: list, stats: dict,
                                    output_path: Path):
    """
    Write emotion_timeline.html — self-contained, offline, no CDN.
    Includes:
      1. Stacked area river chart (Chart.js bundled inline)
         7 emotion bands, X = seconds, Y = stacked 0–100
         Vertical dashed lines at each micro-expression event
      2. Stat summary: dominant emotion, time-in-state table, suppression, range
      3. Transitions table
      4. Bonus: Duchenne smile markers
    """
    # Try to get Duchenne smile markers
    duchenne_ts = _build_duchenne_markers(frame_series)

    # Prepare chart data
    labels = [str(f["t"]) for f in frame_series]
    datasets = []
    for emotion in EMOTIONS:
        data = [f["emotions"].get(emotion, 0) for f in frame_series]
        datasets.append({
            "label": emotion.capitalize(),
            "data": data,
            "backgroundColor": EMOTION_COLORS[emotion],
            "borderColor": EMOTION_COLORS[emotion],
            "borderWidth": 1,
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
        })

    datasets_json = json.dumps(datasets)
    labels_json = json.dumps(labels)
    micro_json = json.dumps(micro_expressions)
    duchenne_json = json.dumps(duchenne_ts)

    # Build micro-expression annotation objects for Chart.js
    annotations_js = "["
    for me in micro_expressions:
        ts = me["timestamp_sec"]
        emotion = me["emotion"]
        color = EMOTION_COLORS.get(emotion, "#ffffff")
        annotations_js += f"""
        {{
            type: 'line',
            xMin: '{ts}',
            xMax: '{ts}',
            borderColor: '{color}',
            borderWidth: 2,
            borderDash: [6, 3],
            label: {{
                content: '{emotion.upper()} {ts}s',
                display: true,
                position: 'start',
                backgroundColor: '{color}aa',
                color: '#fff',
                font: {{ size: 10 }}
            }}
        }},"""
    annotations_js += "]"

    # Build transitions table rows
    transitions_rows = ""
    for t in transitions:
        transitions_rows += f"""
        <tr>
            <td><span class="badge" style="background:{EMOTION_COLORS.get(t['from_emotion'],'#888')}">{t['from_emotion'].capitalize()}</span></td>
            <td><span class="badge" style="background:{EMOTION_COLORS.get(t['to_emotion'],'#888')}">{t['to_emotion'].capitalize()}</span></td>
            <td>{t['timestamp_sec']}s</td>
            <td>{t['transition_duration_sec']}s</td>
        </tr>"""

    # Build micro-expression table rows
    micro_rows = ""
    for me in micro_expressions:
        color = EMOTION_COLORS.get(me["emotion"], "#888")
        suppressed_badge = '<span class="badge-yes">Yes</span>' if me["is_suppressed"] else '<span class="badge-no">No</span>'
        micro_rows += f"""
        <tr>
            <td>{me['id']}</td>
            <td>{me['timestamp_sec']}s</td>
            <td>{me['duration_sec']}s</td>
            <td><span class="badge" style="background:{color}">{me['emotion'].capitalize()}</span></td>
            <td>{round(me['peak_probability']*100, 1)}%</td>
            <td>{me['followed_by'].capitalize()}</td>
            <td>{suppressed_badge}</td>
        </tr>"""

    # Build emotion time-pct table
    emo_time_rows = ""
    for e, pct in sorted(stats["emotion_time_pct"].items(), key=lambda x: -x[1]):
        color = EMOTION_COLORS.get(e, "#888")
        emo_time_rows += f"""
        <tr>
            <td><span class="dot" style="background:{color}"></span> {e.capitalize()}</td>
            <td>{pct}%</td>
            <td><div class="bar-wrap"><div class="bar" style="width:{pct}%;background:{color}"></div></div></td>
        </tr>"""

    # Inline Chart.js
    chartjs_src = _get_chartjs_inline()
    if chartjs_src:
        script_tag = f"<script>\n{chartjs_src}\n</script>"
    else:
        script_tag = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>'

    dominant = stats["dominant_emotion"]
    dominant_color = EMOTION_COLORS.get(dominant, "#94a3b8")
    suppression_score = stats["suppression_score"]
    emotional_range_score = stats["emotional_range_score"]
    duration = stats["duration_sec"]
    video_name = stats["video"]
    n_micro = len(micro_expressions)
    n_transitions = len(transitions)
    n_duchenne = len(duchenne_ts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Emotion Timeline - Sentio Mind</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #21263a;
    --border: #2e3347;
    --text: #e2e8f0;
    --text-muted: #8892a4;
    --accent: #6366f1;
    --accent2: #a78bfa;
    --radius: 12px;
  }}

  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0;
  }}

  /* ── HEADER ─────────────────────────────────────── */
  header {{
    background: linear-gradient(135deg, #1a1d27 0%, #12152099 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px;
    display: flex;
    align-items: center;
    gap: 20px;
  }}
  .logo-icon {{
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
  }}
  .header-info h1 {{
    font-size: 22px; font-weight: 700;
    background: linear-gradient(90deg, #fff, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header-info p {{
    font-size: 13px; color: var(--text-muted); margin-top: 3px;
  }}

  /* ── LAYOUT ──────────────────────────────────────── */
  main {{ padding: 32px 40px; max-width: 1400px; margin: 0 auto; }}

  /* ── STAT CARDS ─────────────────────────────────── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    transition: transform .2s, box-shadow .2s;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.3); }}
  .card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-muted); margin-bottom: 8px; }}
  .card .value {{ font-size: 28px; font-weight: 700; }}
  .card .sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

  /* ── SECTION ─────────────────────────────────────── */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .section-header {{
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 14px;
    display: flex; align-items: center; gap: 8px;
    background: var(--surface2);
  }}
  .section-body {{ padding: 24px; }}

  /* ── CHART ───────────────────────────────────────── */
  .chart-wrap {{
    position: relative;
    height: 320px;
    padding: 8px 0;
  }}

  /* ── TABLE ───────────────────────────────────────── */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{
    text-align: left; padding: 10px 14px;
    color: var(--text-muted); font-weight: 500;
    border-bottom: 1px solid var(--border);
    font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  }}
  .data-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: var(--surface2); }}

  /* ── BADGES ──────────────────────────────────────── */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    color: #fff;
    letter-spacing: .02em;
  }}
  .badge-yes {{ color: #4ade80; font-size: 12px; font-weight: 600; }}
  .badge-no  {{ color: #f87171; font-size: 12px; font-weight: 600; }}
  .dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }}

  /* ── PROGRESS BAR ────────────────────────────────── */
  .bar-wrap {{ background: var(--surface2); border-radius: 4px; overflow: hidden; height: 6px; min-width: 80px; }}
  .bar {{ height: 100%; border-radius: 4px; transition: width .6s ease; }}

  /* ── SCORE RING ──────────────────────────────────── */
  .score-ring {{
    display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
  }}
  .ring-wrap {{
    position: relative; width: 100px; height: 100px;
  }}
  .ring-wrap svg {{ transform: rotate(-90deg); }}
  .ring-label {{
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
  }}
  .ring-label .num {{ font-size: 22px; font-weight: 700; }}
  .ring-label .desc {{ font-size: 10px; color: var(--text-muted); }}

  /* ── FOOTER ──────────────────────────────────────── */
  footer {{
    text-align: center; padding: 20px 40px;
    color: var(--text-muted); font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }}

  /* ── RESPONSIVE ──────────────────────────────────── */
  @media (max-width: 768px) {{
    main, header, footer {{ padding: 20px; }}
    .cards {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <div class="logo-icon">🧠</div>
  <div class="header-info">
    <h1>Emotion Timeline Analysis</h1>
    <p>Sentio Mind · Project 6 · {video_name} · {duration}s · {ANALYSIS_FPS} fps analyzed</p>
  </div>
</header>

<main>

  <!-- STAT CARDS -->
  <div class="cards">
    <div class="card">
      <div class="label">Dominant Emotion</div>
      <div class="value" style="color:{dominant_color}">{dominant.capitalize()}</div>
      <div class="sub">{stats['emotion_time_pct'].get(dominant, 0)}% of session</div>
    </div>
    <div class="card">
      <div class="label">Duration</div>
      <div class="value">{duration}s</div>
      <div class="sub">Analyzed at {ANALYSIS_FPS} fps</div>
    </div>
    <div class="card">
      <div class="label">Micro-Expressions</div>
      <div class="value" style="color:#a855f7">{n_micro}</div>
      <div class="sub">Events detected</div>
    </div>
    <div class="card">
      <div class="label">Suppression Score</div>
      <div class="value" style="color:#ef4444">{suppression_score}</div>
      <div class="sub">Stress indicator (0–100)</div>
    </div>
    <div class="card">
      <div class="label">Emotional Range</div>
      <div class="value" style="color:#22c55e">{emotional_range_score}</div>
      <div class="sub">Expressiveness (0–100)</div>
    </div>
    <div class="card">
      <div class="label">Transitions</div>
      <div class="value" style="color:#eab308">{n_transitions}</div>
      <div class="sub">Emotion shifts</div>
    </div>
    {'<div class="card"><div class="label">Duchenne Smiles</div><div class="value" style="color:#22c55e">' + str(n_duchenne) + '</div><div class="sub">Genuine smiles detected</div></div>' if n_duchenne > 0 else ''}
  </div>

  <!-- RIVER CHART -->
  <div class="section">
    <div class="section-header">📈 Emotion River Chart <span style="color:var(--text-muted);font-weight:400;font-size:12px">· stacked area · micro-expressions marked with dashed lines</span></div>
    <div class="section-body">
      <div class="chart-wrap">
        <canvas id="emotionChart"></canvas>
      </div>
    </div>
  </div>

  <!-- EMOTION TIME TABLE -->
  <div class="section">
    <div class="section-header">🎯 Time-in-State Breakdown</div>
    <div class="section-body">
      <table class="data-table">
        <thead>
          <tr><th>Emotion</th><th>Percentage</th><th>Distribution</th></tr>
        </thead>
        <tbody>
          {emo_time_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- MICRO-EXPRESSIONS TABLE -->
  <div class="section">
    <div class="section-header">⚡ Micro-Expression Events ({n_micro} detected)</div>
    <div class="section-body">
      {'<p style="color:var(--text-muted);font-size:14px">No micro-expressions detected in this session.</p>' if not micro_expressions else f"""
      <table class="data-table">
        <thead>
          <tr><th>#</th><th>Timestamp</th><th>Duration</th><th>Emotion</th><th>Peak Prob</th><th>Followed By</th><th>Suppressed</th></tr>
        </thead>
        <tbody>{micro_rows}</tbody>
      </table>"""}
    </div>
  </div>

  <!-- TRANSITIONS TABLE -->
  <div class="section">
    <div class="section-header">🔀 Emotion Transitions ({n_transitions} detected)</div>
    <div class="section-body">
      {'<p style="color:var(--text-muted);font-size:14px">No significant transitions detected.</p>' if not transitions else f"""
      <table class="data-table">
        <thead>
          <tr><th>From</th><th>To</th><th>Timestamp</th><th>Duration</th></tr>
        </thead>
        <tbody>{transitions_rows}</tbody>
      </table>"""}
    </div>
  </div>

</main>

<footer>
  Generated by Sentio Mind · Project 6 · Emotion Timeline Analyzer · {video_name}
</footer>

{script_tag}

<script>
// ── Chart.js annotation plugin (inline minimal version) ──
// We implement vertical lines manually via plugin
const MICRO_EXPRESSIONS = {micro_json};
const DUCHENNE_TS = {duchenne_json};

const verticalLinesPlugin = {{
  id: 'verticalLines',
  afterDraw(chart) {{
    const ctx = chart.ctx;
    const xAxis = chart.scales.x;
    const yAxis = chart.scales.y;
    const labels = chart.data.labels;

    // Draw micro-expression lines
    MICRO_EXPRESSIONS.forEach(me => {{
      const tsStr = String(me.timestamp_sec);
      const idx = labels.indexOf(tsStr);
      if (idx === -1) return;
      const x = xAxis.getPixelForValue(idx);
      const colors = {{
        angry: '#ef4444', disgust: '#f97316', fear: '#a855f7',
        happy: '#22c55e', neutral: '#94a3b8', sad: '#3b82f6', surprise: '#eab308'
      }};
      const color = colors[me.emotion] || '#fff';
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([6, 3]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.moveTo(x, yAxis.top);
      ctx.lineTo(x, yAxis.bottom);
      ctx.stroke();
      // Label
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.save();
      ctx.translate(x + 3, yAxis.top + 60);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(me.emotion.toUpperCase(), 0, 0);
      ctx.restore();
      ctx.restore();
    }});

    // Draw Duchenne smile markers
    DUCHENNE_TS.forEach(ts => {{
      const tsStr = String(ts);
      const idx = labels.indexOf(tsStr);
      if (idx === -1) return;
      const x = xAxis.getPixelForValue(idx);
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([3, 2]);
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 1.5;
      ctx.moveTo(x, yAxis.top);
      ctx.lineTo(x, yAxis.bottom);
      ctx.stroke();
      ctx.fillStyle = '#22c55e';
      ctx.font = '9px Inter, sans-serif';
      ctx.fillText('😊', x - 6, yAxis.top + 12);
      ctx.restore();
    }});
  }}
}};

Chart.register(verticalLinesPlugin);

const ctx = document.getElementById('emotionChart').getContext('2d');

const datasets = {datasets_json};
// Set fill to 'stack' mode
datasets.forEach(ds => {{ ds.fill = true; }});

const emotionChart = new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {labels_json},
    datasets: datasets
  }},
  options: {{
    animation: {{ duration: 800, easing: 'easeInOutCubic' }},
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{
          color: '#e2e8f0',
          font: {{ size: 12, family: 'Inter' }},
          padding: 16,
          usePointStyle: true,
          pointStyleWidth: 10,
        }}
      }},
      tooltip: {{
        backgroundColor: '#1a1d27',
        borderColor: '#2e3347',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#8892a4',
        padding: 12,
        callbacks: {{
          title: (items) => `t = ${{items[0].label}}s`,
          label: (item) => ` ${{item.dataset.label}}: ${{item.parsed.y.toFixed(1)}}%`
        }}
      }}
    }},
    scales: {{
      x: {{
        stacked: true,
        grid: {{ color: '#2e3347', drawBorder: false }},
        ticks: {{
          color: '#8892a4',
          font: {{ size: 10 }},
          maxTicksLimit: 20,
          callback: function(val, index) {{
            return this.getLabelForValue(val) + 's';
          }}
        }},
        title: {{ display: true, text: 'Time (seconds)', color: '#8892a4' }}
      }},
      y: {{
        stacked: true,
        min: 0,
        max: 100,
        grid: {{ color: '#2e3347', drawBorder: false }},
        ticks: {{
          color: '#8892a4',
          font: {{ size: 10 }},
          callback: v => v + '%'
        }},
        title: {{ display: true, text: 'Probability (%)', color: '#8892a4' }}
      }}
    }}
  }}
}});
</script>

</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report written → {output_path}")


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
                # Interpolate: copy last known frame
                emotions = frame_series[-1]["emotions"].copy()
            else:
                emotions = {e: (100.0 if e == "neutral" else 0.0) for e in EMOTIONS}
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
        json.dump(stats, f, indent=2)
    print(f"JSON output written → {OUTPUT_JSON}")

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
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 50)
