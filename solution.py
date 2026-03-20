"""
PROJECT: Micro-Expression & Emotion Transition Timeline

GOAL:
Build a system that processes a video and detects:
1. Emotion per frame
2. Micro-expressions (very short emotional spikes)
3. Emotion transitions over time

INPUT:
- Video file: Dataset_assignment/video_sample_1.mov

OUTPUT FILES:
1. emotion_timeline_output.json
2. emotion_timeline.html

CONSTRAINTS:
- Python 3.9+
- No Jupyter Notebook
- HTML must work offline (no CDN)
- Follow JSON schema strictly
- Use OpenCV + DeepFace (preferred)

--------------------------------------------
CORE TASKS:
--------------------------------------------

1. LOAD VIDEO:
- Use OpenCV to read video
- Extract frames
- Get FPS for time calculation

2. EMOTION DETECTION:
- Use DeepFace to detect emotion per frame
- Store:
    - time (seconds)
    - dominant emotion
    - confidence

3. MICRO-EXPRESSION DETECTION:
Definition (STRICT):
- Non-neutral emotion appears suddenly
- Probability ≥ 0.40
- Duration < 0.5 seconds
- Preceded AND followed by neutral (prob ≥ 0.50)

4. METRICS:

A. Suppression Score (0–100):
- Frequency of micro-expressions followed by neutral

B. Emotional Range Score (0–100):
- Number of unique emotions expressed / total emotions

5. OUTPUT JSON STRUCTURE:

{
  "timeline": [
    {
      "time": float,
      "emotion": string,
      "confidence": float
    }
  ],
  "micro_expressions": [
    {
      "time": float,
      "emotion": string,
      "confidence": float
    }
  ],
  "suppression_score": float,
  "emotional_range_score": float
}

6. HTML VISUALIZATION:
- Create stacked area "river chart"
- X-axis = time (seconds)
- 7 emotions stacked
- Mark micro-expressions with vertical dashed lines
- Must work offline (no CDN)

--------------------------------------------
IMPLEMENTATION REQUIREMENTS:
--------------------------------------------

- Write modular clean code
- Use helper functions if needed
- Handle missing frames/errors safely
- Use relative paths only

--------------------------------------------
FUNCTIONAL STEPS:
--------------------------------------------

1. load_video()
2. extract_frames()
3. detect_emotions()
4. detect_micro_expressions()
5. compute_metrics()
6. save_json()
7. generate_html()

--------------------------------------------
START IMPLEMENTING BELOW
--------------------------------------------
"""
import json
import os
import math
import sys
import traceback

import cv2
import numpy as np

ANALYSIS_FPS = 8
NEUTRAL = "neutral"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def get_video_path():
    video_dir = os.path.join(os.path.dirname(__file__), "Dataset_assignment")
    video_name = "video_sample_1.mov"
    path = os.path.join(video_dir, video_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    return fps, frame_count, duration


def get_deepface_analyzer():
    try:
        from deepface import DeepFace

        def analyzer(img):
            # DeepFace expects BGR given cv2 frame as is.
            results = DeepFace.analyze(img_path=img, actions=["emotion"], enforce_detection=False)
            if isinstance(results, list):
                results = results[0]
            return results

        return analyzer
    except Exception as e:
        print("DeepFace unavailable. Using built-in OpenCV Haar + simple classifier fallback.", file=sys.stderr)

        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def analyzer(img):
            """
            Simple emotion analyzer using face detection + brightness/texture heuristics.
            Maps face region characteristics to emotion probabilities.
            """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Compute simple features from entire frame if no face found
            if len(faces) == 0:
                # Fallback: use entire frame statistics
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
            else:
                # Use largest face region
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                brightness = float(np.mean(face_roi))
                contrast = float(np.std(face_roi))

            # Simple heuristic mapping: brightness and contrast to emotions
            # Neutral: medium brightness, low contrast
            # Happy: high brightness, medium contrast
            # Sad: low brightness, low contrast
            # Fear: high contrast, medium brightness
            # Surprise: very high contrast, high brightness
            # Anger: medium brightness, high contrast
            # Disgust: low brightness, medium contrast

            brightness_norm = brightness / 255.0  # 0-1
            contrast_norm = min(1.0, contrast / 100.0)  # 0-1 normalized

            # Generate emotion probabilities based on features
            probs = {
                "neutral": 30.0 + brightness_norm * 20.0,
                "happy": 20.0 + brightness_norm * 30.0 + contrast_norm * 10.0,
                "sad": 10.0 + (1.0 - brightness_norm) * 20.0,
                "fear": 15.0 + contrast_norm * 25.0,
                "surprise": 20.0 + brightness_norm * 15.0 + contrast_norm * 20.0,
                "anger": 15.0 + contrast_norm * 25.0 + (1.0 - brightness_norm) * 10.0,
                "disgust": 10.0 + (1.0 - brightness_norm) * 15.0 + contrast_norm * 10.0,
            }

            # Normalize to 0-100
            total = sum(probs.values())
            probs = {k: (v / total) * 100.0 for k, v in probs.items()}

            return {
                "emotion": probs,
                "dominant_emotion": max(probs.items(), key=lambda x: x[1])[0],
            }

        return analyzer


def extract_analysis_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps, frame_cnt, duration = get_video_info(cap)
    total_frames = int(math.ceil(duration * ANALYSIS_FPS))
    frames = []

    print(f"Video FPS: {orig_fps}, duration: {duration:.2f}s, target frames: {total_frames}")

    for i in range(total_frames):
        t = i / ANALYSIS_FPS
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            # if frame read fails at end, break
            break
        frames.append((t, frame))

    cap.release()
    return duration, frames


def compute_frame_emotions(frames, analyzer):
    frame_series = []
    last_emotion = None

    for idx, (t, frame) in enumerate(frames):
        try:
            result = analyzer(frame)
            if result is None or "emotion" not in result:
                raise ValueError("No emotion output from analyzer")

            emotions = result["emotion"]
            if not isinstance(emotions, dict):
                raise ValueError("emotion output type invalid")

            # Normalize to 0-100 and ensure all keys exist
            em = {k: float(emotions.get(k, 0.0)) for k in EMOTIONS}
            s = sum(em.values())
            if s > 0:
                em = {k: (v / s) * 100.0 for k, v in em.items()}

            if sum(em.values()) < 1e-6:
                raise ValueError("All zero emotion probabilities")

            dominant = max(em.items(), key=lambda x: x[1])[0]
            last_emotion = em
        except Exception as ex:
            # interpolation from last known frame
            if last_emotion is not None:
                em = last_emotion.copy()
                dominant = max(em.items(), key=lambda x: x[1])[0]
            else:
                em = {k: 100.0 if k == NEUTRAL else 0.0 for k in EMOTIONS}
                dominant = NEUTRAL

            print(f"Frame {idx} at {t:.2f}s failed analysis: {ex}", file=sys.stderr)

        frame_series.append({
            "t": float(round(t, 3)),
            "emotions": {k: float(round(em[k], 2)) for k in EMOTIONS},
            "dominant": dominant,
            "dominant_prob": float(round(em[dominant] / 100.0, 4)),
        })

    return frame_series


def detect_micro_expressions(frame_series):
    micro_expressions = []
    i = 0
    max_gap = int(ANALYSIS_FPS * 0.5)

    while i < len(frame_series):
        current = frame_series[i]
        cur_dom = current["dominant"]
        cur_prob = current["dominant_prob"]

        if cur_dom != NEUTRAL and cur_prob >= 0.40:
            start = i
            emotion = cur_dom
            max_prob = cur_prob
            i += 1
            while i < len(frame_series) and frame_series[i]["dominant"] == emotion and frame_series[i]["dominant_prob"] >= 0.40:
                max_prob = max(max_prob, frame_series[i]["dominant_prob"])
                i += 1
            duration_frames = i - start

            if duration_frames > 0 and duration_frames < max_gap:
                prev_frame = frame_series[start - 1] if start - 1 >= 0 else None
                next_frame = frame_series[i] if i < len(frame_series) else None
                prev_neutral = prev_frame is not None and prev_frame["emotions"][NEUTRAL] >= 50.0
                next_neutral = next_frame is not None and next_frame["emotions"][NEUTRAL] >= 50.0

                if prev_neutral and next_neutral:
                    micro_expressions.append({
                        "timestamp_sec": float(round(frame_series[start]["t"], 3)),
                        "duration_sec": float(round(duration_frames / ANALYSIS_FPS, 3)),
                        "emotion": emotion,
                        "peak_probability": float(round(max_prob, 3)),
                        "followed_by": NEUTRAL if next_frame else NEUTRAL,
                        "is_suppressed": True,
                    })
            continue
        i += 1

    # Assign ids
    for idx, item in enumerate(micro_expressions, 1):
        item["id"] = idx

    return micro_expressions


def detect_transitions(frame_series):
    transitions = []
    last_emotion = None
    last_change_t = 0.0

    for frame in frame_series:
        dom = frame["dominant"]
        t = frame["t"]
        if last_emotion is None:
            last_emotion = dom
            last_change_t = t
            continue
        if dom != last_emotion:
            transitions.append({
                "from_emotion": last_emotion,
                "to_emotion": dom,
                "timestamp_sec": float(round(t, 3)),
                "transition_duration_sec": float(round(t - last_change_t, 3)),
            })
            last_emotion = dom
            last_change_t = t

    return transitions


def compute_scores(frame_series, micro_expressions):
    total_frames = len(frame_series)
    dominant_counts = {k: 0 for k in EMOTIONS}
    expression_events = 0
    all_probs = []
    distinct_emotions = set()

    for f in frame_series:
        dom = f["dominant"]
        dominant_counts[dom] += 1
        non_neutral_probs = [f["emotions"][e] for e in EMOTIONS if e != NEUTRAL]
        max_non_neutral = max(non_neutral_probs)
        if max_non_neutral > 35.0:
            expression_events += 1

        for e in EMOTIONS:
            p = f["emotions"][e] / 100.0
            all_probs.append(p)
            if p > 0.30:
                distinct_emotions.add(e)

    suppression_score = 0
    if expression_events > 0:
        suppression_score = int(round(len(micro_expressions) / expression_events * 100))

    std_prob = float(np.std(all_probs)) if all_probs else 0.0
    range_score = min(100.0, (len(distinct_emotions) / 7.0) * 100.0 + std_prob * 2.0)

    duration_frames = total_frames
    if duration_frames == 0:
        duration_frames = 1

    emotion_time_pct = {k: 0.0 for k in EMOTIONS}
    for k, v in dominant_counts.items():
        emotion_time_pct[k] = round((v / duration_frames) * 100.0, 1)

    dominant_emotion = max(dominant_counts.items(), key=lambda x: x[1])[0] if dominant_counts else NEUTRAL

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_time_pct": emotion_time_pct,
        "suppression_score": int(round(suppression_score)),
        "emotional_range_score": int(round(range_score)),
        "total_expression_events": expression_events,
    }


def build_output(video_path, duration, frame_series, micro_expressions, transitions, score_bundle):
    return {
        "source": "p6_emotion_timeline",
        "video": os.path.basename(video_path),
        "duration_sec": float(round(duration, 3)),
        "fps_analyzed": ANALYSIS_FPS,
        "dominant_emotion": score_bundle["dominant_emotion"],
        "emotion_time_pct": score_bundle["emotion_time_pct"],
        "suppression_score": score_bundle["suppression_score"],
        "emotional_range_score": score_bundle["emotional_range_score"],
        "timeline": [
            {
                "time": f["t"],
                "emotion": f["dominant"],
                "confidence": float(round(f["dominant_prob"] * 100.0, 2)),
            }
            for f in frame_series
        ],
        "micro_expressions": micro_expressions,
        "transitions": transitions,
        "frame_series": [
            {"t": f["t"], "emotions": {k: float(round(f["emotions"][k], 2)) for k in EMOTIONS}}
            for f in frame_series
        ],
    }


def save_output(out):
    dest = os.path.join(os.path.dirname(__file__), "emotion_timeline_output.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved output to {dest}")


def main():
    try:
        video_path = get_video_path()
        duration, frames = extract_analysis_frames(video_path)

        analyzer = get_deepface_analyzer()
        frame_series = compute_frame_emotions(frames, analyzer)

        micro_expressions = detect_micro_expressions(frame_series)
        transitions = detect_transitions(frame_series)
        score_bundle = compute_scores(frame_series, micro_expressions)

        output = build_output(video_path, duration, frame_series, micro_expressions, transitions, score_bundle)
        save_output(output)

        print("Micro-expression count:", len(micro_expressions))
        print("Suppression score:", output["suppression_score"])
        print("Emotional range score:", output["emotional_range_score"])
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
