import cv2
import json
import numpy as np
import os
from deepface import DeepFace
from collections import deque

VIDEO_PATH = r"C:\Emotion_timeline\Class_8_cctv_video_1.mov"
PROFILE_DIR = "Profiles_1"   # folder with 10 profile images
OUTPUT_JSON = "emotion_timeline_output.json"

EMOTION_THRESHOLD = 0.40
NEUTRAL_THRESHOLD = 0.50
SIMILARITY_THRESHOLD = 0.50
# ---------------------------
# FACE RECOGNITION UTILS
# ---------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def encode_profiles(profile_dir):
    encodings = {}

    for img_name in os.listdir(profile_dir):
        path = os.path.join(profile_dir, img_name)

        try:
            embedding = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            encodings[img_name] = embedding
            print(f"Encoded: {img_name}")

        except:
            print(f"Skipping: {img_name}")

    return encodings


def identify_face(face_img, profile_encodings):
    try:
        embedding = DeepFace.represent(
            face_img,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
    except:
        return None, 0

    best_match = None
    best_score = -1

    for name, ref_emb in profile_encodings.items():
        score = cosine_similarity(embedding, ref_emb)

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def process_video(video_path, profile_encodings):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    timeline = []
    frame_idx = 0

    target_person = None
    tracker = None
    bbox = None

    cached_faces = []

    last_emotion_time = 0
    last_probs = {e: 0 for e in ['angry','disgust','fear','happy','sad','surprise','neutral']}
    last_emotion = "neutral"
    emotion_buffer = deque(maxlen=3)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ---------------------------
        # FRAME SKIP (IMPORTANT)
        # ---------------------------
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue

        selected_face = None

        # ---------------------------
        # TRACKING MODE
        # ---------------------------
        if tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)
                h_frame, w_frame = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                face_img = frame[y:y+h, x:x+w]
                selected_face = face_img
            else:
                tracker = None  # fallback to detection

        # ---------------------------
        # DETECTION MODE
        # ---------------------------
        if tracker is None:

            # detect every 5 frames OR until target found
            if frame_idx % 10 == 0 or target_person is None:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="retinaface",
                    enforce_detection=False
                )
                cached_faces = faces
            else:
                faces = cached_faces

            faces = sorted(
                faces,
                key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"],
                reverse=True
            )[:2]

            for face in faces:
                face_img = (face["face"] * 255).astype("uint8")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                # ---------------------------
                # RECOGNITION (ONLY FIRST FRAMES)
                # ---------------------------
                if target_person is None and frame_idx < 30:
                    name, score = identify_face(face_img, profile_encodings)

                    if name and score > SIMILARITY_THRESHOLD:
                        target_person = name
                        print(f"🎯 Target locked: {target_person}")

                        x = face["facial_area"]["x"]
                        y = face["facial_area"]["y"]
                        w = face["facial_area"]["w"]
                        h = face["facial_area"]["h"]

                        bbox = (x, y, w, h)

                        # INIT TRACKER
                        tracker = cv2.legacy.TrackerKCF_create()
                        tracker.init(frame, bbox)

                        selected_face = face_img
                        break

                elif target_person is not None:
                    name, score = identify_face(face_img, profile_encodings)

                    if name == target_person and score > SIMILARITY_THRESHOLD:
                        selected_face = face_img

                        x = face["facial_area"]["x"]
                        y = face["facial_area"]["y"]
                        w = face["facial_area"]["w"]
                        h = face["facial_area"]["h"]

                        bbox = (x, y, w, h)

                        tracker = cv2.legacy.TrackerKCF_create()
                        tracker.init(frame, bbox)

                        break

        # ---------------------------
        # PERIODIC RE-DETECTION (ANTI-DRIFT)
        # ---------------------------
        if frame_idx % 50 == 0:
            tracker = None

        # ---------------------------
        # EMOTION ANALYSIS
        # ---------------------------
        if selected_face is not None:
            try:
                if (time_sec - last_emotion_time) >= 0.5:
                    if selected_face is not None and selected_face.size > 0:
                        face_resized = cv2.resize(selected_face, (224, 224))

                    result = DeepFace.analyze(
                        face_resized,
                        actions=['emotion'],
                        enforce_detection=False
                    )[0]

                    # Add to buffer
                    emotion_buffer.append(result['emotion'])

                    # Smooth probabilities
                    avg_probs = {}
                    for key in result['emotion']:
                        avg_probs[key] = np.mean([e[key] for e in emotion_buffer])

                    last_probs = {k: float(v) for k, v in avg_probs.items()}
                    last_emotion = max(avg_probs, key=avg_probs.get)

                    last_emotion_time = float(time_sec)

            except:
                pass

        timeline.append({
            "time": time_sec,
            "emotion": last_emotion,
            "probabilities": last_probs
        })

        frame_idx += 1

    cap.release()
    return timeline, fps


# ---------------------------
# SEGMENTS
# ---------------------------
def build_segments(timeline):
    segments = []
    current = None

    for entry in timeline:
        if current is None:
            current = {
                "emotion": entry["emotion"],
                "start": entry["time"],
                "end": entry["time"],
                "probs": [entry["probabilities"]]
            }
        elif entry["emotion"] == current["emotion"]:
            current["end"] = entry["time"]
            current["probs"].append(entry["probabilities"])
        else:
            segments.append(current)
            current = {
                "emotion": entry["emotion"],
                "start": entry["time"],
                "end": entry["time"],
                "probs": [entry["probabilities"]]
            }

    if current:
        segments.append(current)

    return segments


# ---------------------------
# MICRO EXPRESSIONS
# ---------------------------
def detect_micro_expressions(segments):
    micros = []

    for i in range(1, len(segments) - 1):
        seg, prev_seg, next_seg = segments[i], segments[i-1], segments[i+1]

        duration = seg["end"] - seg["start"]
        frame_count = len(seg["probs"])

        # average probability of the segment's emotion
        avg_prob = np.mean([p.get(seg["emotion"], 0) for p in seg["probs"]])

        # neutral context before & after
        prev_n = np.mean([p.get("neutral", 0) for p in prev_seg["probs"]])
        next_n = np.mean([p.get("neutral", 0) for p in next_seg["probs"]])

        # consistency: how stable the emotion is within the segment
        consistency = np.mean([
            p.get(seg["emotion"], 0) >= EMOTION_THRESHOLD
            for p in seg["probs"]
        ])

        if (
            seg["emotion"] != "neutral" and
            avg_prob >= EMOTION_THRESHOLD and
            0.2 < duration < 0.8 and   # avoids too short & too long
            frame_count >= 2 and       # removes 1-frame noise
            prev_n >= NEUTRAL_THRESHOLD and
            next_n >= NEUTRAL_THRESHOLD and
            consistency >= 0.6         # ensures stability
        ):
            micros.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "emotion": seg["emotion"],
                "max_prob": round(avg_prob, 2)
            })

    return micros

# ---------------------------
# METRICS
# ---------------------------
def compute_metrics(timeline, micro_expressions):
    emotions = [t["emotion"] for t in timeline]

    unique_emotions = set(emotions)
    emotional_range_score = (len(unique_emotions) / 7) * 100

    suppression_score = (len(micro_expressions) / max(len(timeline), 1)) * 100

    return {
        "suppression_score": round(suppression_score, 2),
        "emotional_range_score": round(emotional_range_score, 2)
    }


# ---------------------------
# SAVE JSON
# ---------------------------
def save_output(timeline, micro_expressions, metrics):
    output = {
        "timeline": timeline,
        "micro_expressions": micro_expressions,
        "metrics": metrics
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # print("Encoding profiles...")
    profile_encodings = encode_profiles(PROFILE_DIR)

    print("Processing video...")
    timeline, fps = process_video(VIDEO_PATH, profile_encodings)

    print("Building segments...")
    segments = build_segments(timeline)

    print("Detecting micro-expressions...")
    micro_expressions = detect_micro_expressions(segments)

    print("Computing metrics...")
    metrics = compute_metrics(timeline, micro_expressions)

    print("Saving output...")
    save_output(timeline, micro_expressions, metrics)

    print("Done!")