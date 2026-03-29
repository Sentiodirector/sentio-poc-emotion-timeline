import cv2
from deepface import DeepFace
import numpy as np
import json
import os

# =========================
# CONFIG
# =========================
video_path = "Class_8_cctv_video_1.mov"
ANALYSIS_FPS = 10

print("Video exists:", os.path.exists(video_path))


# =========================
# FRAME EXTRACTION
# =========================
def extract_frames(video_path, analysis_fps=10):
    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 25

    frame_interval = int(original_fps / analysis_fps)

    frames = []
    timestamps = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            time_sec = frame_id / original_fps
            frames.append(frame)
            timestamps.append(time_sec)

        frame_id += 1

    cap.release()
    return frames, timestamps


# =========================
# EMOTION DETECTION
# =========================
def get_emotion(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        return result[0]['emotion']

    except Exception:
        return {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 100
        }


# =========================
# NORMALIZE (0-1)
# =========================
def normalize_emotions(emotion_list):
    for frame in emotion_list:
        frame['emotions'] = {
            k: float(v)/100 for k, v in frame['emotions'].items()
        }
    return emotion_list


# =========================
# SMOOTHING
# =========================
def smooth_emotions(emotion_list, window=3):
    smoothed = []

    for i in range(len(emotion_list)):
        avg = {}
        count = 0

        for j in range(max(0, i-window), min(len(emotion_list), i+window)):
            for k, v in emotion_list[j]['emotions'].items():
                avg[k] = avg.get(k, 0) + v
            count += 1

        avg = {k: v/count for k, v in avg.items()}

        smoothed.append({
            "time": emotion_list[i]['time'],
            "emotions": avg
        })

    return smoothed


# =========================
# MICRO EXPRESSION DETECTION
# =========================
def detect_micro_expressions(emotion_list, fps=10):
    micro_events = []
    max_frames = int(fps * 0.5)

    i = 1

    while i < len(emotion_list) - 1:
        emotions = emotion_list[i]['emotions']

        dominant = max(
            [(k, v) for k, v in emotions.items() if k != "neutral"],
            key=lambda x: x[1]
        )

        emotion, prob = dominant

        if prob >= 0.40:
            start = i

            while i < len(emotion_list) and emotion_list[i]['emotions'][emotion] >= 0.40:
                i += 1

            duration = i - start

            if duration <= max_frames:
                if (
                    emotion_list[start-1]['emotions']['neutral'] >= 0.50 and
                    emotion_list[min(i, len(emotion_list)-1)]['emotions']['neutral'] >= 0.50
                ):
                    micro_events.append({
                        "time": emotion_list[start]['time'],
                        "emotion": emotion,
                        "duration": duration / fps
                    })
        else:
            i += 1

    return micro_events


# =========================
# METRICS
# =========================
def suppression_score(micro_events, total_frames):
    if total_frames == 0:
        return 0
    score = (len(micro_events) / total_frames) * 100
    return min(100, score * 10)


def emotional_range(emotion_list):
    emotions_seen = set()

    for frame in emotion_list:
        dominant = max(frame['emotions'], key=frame['emotions'].get)
        emotions_seen.add(dominant)

    return (len(emotions_seen) / 7) * 100


# =========================
# MAIN
# =========================
frames, timestamps = extract_frames(video_path, ANALYSIS_FPS)

print("Total sampled frames:", len(frames))

emotion_list = []

for i, frame in enumerate(frames):
    print(f"Processing frame {i+1}/{len(frames)}")

    emotions = get_emotion(frame)

    emotion_list.append({
        "time": timestamps[i],
        "emotions": emotions
    })

print("\nEmotion extraction complete!")


# =========================
# POST PROCESSING
# =========================
emotion_list = normalize_emotions(emotion_list)
emotion_list = smooth_emotions(emotion_list)

micro_events = detect_micro_expressions(emotion_list)

supp_score = suppression_score(micro_events, len(emotion_list))
range_score = emotional_range(emotion_list)

print("\nMicro-expressions detected:", len(micro_events))
for e in micro_events[:5]:
    print(e)

print("Suppression Score:", supp_score)
print("Emotional Range:", range_score)


# =========================
# SAVE JSON
# =========================
output = {
    "micro_expressions": micro_events,
    "suppression_score": supp_score,
    "emotional_range": range_score
}

with open("emotion_timeline_output.json", "w") as f:
    json.dump(output, f, indent=4)

print("\nJSON saved successfully!")