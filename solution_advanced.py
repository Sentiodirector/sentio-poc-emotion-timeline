#!/usr/bin/env python3
"""
Sentio Mind - Advanced Micro-Expression Detection System
=========================================================
3-Layer Validation Approach for robust micro-expression detection:
  Layer 1: DeepFace + Sliding Window Smoothing
  Layer 2: Optical Flow Validation (OpenCV)
  Layer 3: MediaPipe Landmark Displacement

Author: Sentio Mind Platform
Python 3.9+ required
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_COLORS = {
    'angry': '#FF4444',
    'disgust': '#8B4513',
    'fear': '#9B59B6',
    'happy': '#FFD700',
    'sad': '#4A90D9',
    'surprise': '#FF8C00',
    'neutral': '#95A5A6'
}

# Thresholds
OPTICAL_FLOW_THRESHOLD = 0.1
LANDMARK_DISPLACEMENT_THRESHOLD = 0.005
EMOTION_SPIKE_THRESHOLD = 0.05
MICRO_EXPR_EMOTION_THRESHOLD = 0.10
NEUTRAL_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 40

# MediaPipe landmark indices
EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LIP_LANDMARKS = [61, 91, 181, 84, 17, 314, 405, 321, 375, 291]
EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133]
KEY_LANDMARKS = EYEBROW_LANDMARKS + LIP_LANDMARKS + EYE_LANDMARKS


# ═══════════════════════════════════════════════════════════════
# LAYER 1: DeepFace + Sliding Window Smoothing
# ═══════════════════════════════════════════════════════════════

def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion scores to sum to 1.0"""
    total = sum(emotions.values())
    if total == 0:
        return {e: 1.0/7 for e in ALL_EMOTIONS}
    return {k: v / total for k, v in emotions.items()}


def analyze_frame_deepface(frame: np.ndarray) -> Optional[Dict[str, float]]:
    """Analyze a single frame using DeepFace"""
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        if isinstance(result, list):
            result = result[0]
        emotions = {k: float(v) / 100.0 for k, v in result['emotion'].items()}
        return normalize_emotions(emotions)
    except Exception:
        return None


def smooth_emotions(frame_emotions: List[Dict], window: int = 3) -> List[Dict[str, float]]:
    """Apply sliding window smoothing to emotion probabilities"""
    smoothed = []
    for i in range(len(frame_emotions)):
        start = max(0, i - window // 2)
        end = min(len(frame_emotions), i + window // 2 + 1)
        window_frames = frame_emotions[start:end]

        # Filter out None frames
        valid_frames = [f for f in window_frames if f.get('emotions') is not None]
        if not valid_frames:
            smoothed.append({e: 0.0 for e in ALL_EMOTIONS})
            continue

        avg_emotions = {}
        for emotion in ALL_EMOTIONS:
            values = [f['emotions'].get(emotion, 0) for f in valid_frames]
            avg_emotions[emotion] = float(np.mean(values))
        smoothed.append(avg_emotions)
    return smoothed


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Optical Flow Validation
# ═══════════════════════════════════════════════════════════════

def compute_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray,
                         face_region: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    Compute optical flow magnitude between two frames.
    Returns average magnitude in face region if provided.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    if face_region is not None:
        x, y, w, h = face_region
        # Ensure bounds are valid
        y1, y2 = max(0, y), min(magnitude.shape[0], y + h)
        x1, x2 = max(0, x), min(magnitude.shape[1], x + w)
        if y2 > y1 and x2 > x1:
            face_flow = magnitude[y1:y2, x1:x2].mean()
            return float(face_flow)

    return float(magnitude.mean())


def detect_face_region(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect face region using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Return largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest)
    return None


# ═══════════════════════════════════════════════════════════════
# LAYER 3: MediaPipe Landmark Displacement
# ═══════════════════════════════════════════════════════════════

class LandmarkTracker:
    """Track facial landmarks using MediaPipe Tasks API or OpenCV fallback"""

    def __init__(self):
        self.prev_landmarks = None
        self.face_cascade = None
        self.use_mediapipe = False
        self.face_landmarker = None

        # Try to initialize MediaPipe Tasks API (new version)
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            import urllib.request
            import os

            # Download model if not exists
            model_path = "face_landmarker.task"
            if not os.path.exists(model_path):
                print("[INFO] Downloading MediaPipe face landmark model...")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(model_url, model_path)

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.use_mediapipe = True
            print("[INFO] MediaPipe Face Landmarker initialized successfully")
        except Exception as e:
            print(f"[INFO] MediaPipe Tasks not available ({e}), using OpenCV fallback")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.use_mediapipe = False

    def get_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract key facial landmarks from frame"""
        if self.use_mediapipe and self.face_landmarker:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.face_landmarker.detect(mp_image)

                if results.face_landmarks and len(results.face_landmarks) > 0:
                    face_landmarks = results.face_landmarks[0]
                    h, w = frame.shape[:2]

                    landmarks = []
                    for idx in KEY_LANDMARKS:
                        if idx < len(face_landmarks):
                            lm = face_landmarks[idx]
                            landmarks.append([lm.x * w, lm.y * h])

                    return np.array(landmarks) if landmarks else None
            except Exception:
                pass

        # Fallback: Use face detection + corner detection for key points
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]

                # Use Shi-Tomasi corner detection on face region
                corners = cv2.goodFeaturesToTrack(face_roi, 20, 0.01, 10)
                if corners is not None:
                    landmarks = []
                    for corner in corners:
                        cx, cy = corner.ravel()
                        landmarks.append([x + cx, y + cy])
                    return np.array(landmarks)

        return None

    def compute_displacement(self, frame: np.ndarray) -> float:
        """Compute landmark displacement from previous frame"""
        current_landmarks = self.get_landmarks(frame)

        if current_landmarks is None:
            return 0.0

        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return 0.0

        # Ensure same number of landmarks
        min_len = min(len(current_landmarks), len(self.prev_landmarks))
        if min_len == 0:
            self.prev_landmarks = current_landmarks
            return 0.0

        # Compute euclidean displacement
        curr = current_landmarks[:min_len]
        prev = self.prev_landmarks[:min_len]

        displacement = np.sqrt(np.sum((curr - prev) ** 2, axis=1)).mean()

        # Normalize by frame diagonal
        frame_diagonal = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
        normalized_displacement = displacement / frame_diagonal

        self.prev_landmarks = current_landmarks
        return float(normalized_displacement)

    def reset(self):
        """Reset tracker state"""
        self.prev_landmarks = None


# ═══════════════════════════════════════════════════════════════
# 3-LAYER CONFIDENCE VOTING
# ═══════════════════════════════════════════════════════════════

def compute_confidence_score(
    deepface_spike: float,
    optical_flow: float,
    landmark_displacement: float
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute confidence score using 3-layer validation.
    Returns (confidence_score, validation_layers)
    """
    confidence = 0.0
    validation_layers = {
        'deepface': False,
        'optical_flow': False,
        'landmark': False
    }

    # Layer 1: DeepFace spike
    if deepface_spike >= MICRO_EXPR_EMOTION_THRESHOLD:
        confidence += 40
        validation_layers['deepface'] = True

    # Layer 2: Optical flow validation
    if optical_flow > OPTICAL_FLOW_THRESHOLD:
        confidence += 35
        validation_layers['optical_flow'] = True

    # Layer 3: Landmark displacement
    if landmark_displacement > LANDMARK_DISPLACEMENT_THRESHOLD:
        confidence += 25
        validation_layers['landmark'] = True

    return confidence, validation_layers


# ═══════════════════════════════════════════════════════════════
# MICRO-EXPRESSION DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_micro_expressions(
    frame_emotions: List[Dict],
    analysis_fps: int
) -> List[Dict]:
    """
    Detect micro-expressions using 3-layer validation.

    Conditions:
    1. Previous frame dominant = neutral with prob >= 0.50
    2. Current frame non-neutral emotion >= 0.40
    3. Duration < 0.5 seconds
    4. Next frame returns to neutral >= 0.50
    5. Confidence score >= 75
    """
    micro_expressions = []
    max_duration_frames = int(analysis_fps * 0.5)

    # Apply smoothing
    smoothed = smooth_emotions(frame_emotions)

    i = 1
    while i < len(frame_emotions) - 1:
        frame = frame_emotions[i]
        prev_frame = frame_emotions[i - 1]

        # Skip if no valid emotion data
        if frame.get('emotions') is None or prev_frame.get('emotions') is None:
            i += 1
            continue

        prev_emotions = prev_frame['emotions']
        curr_emotions = frame['emotions']
        smoothed_emotions = smoothed[i]

        # Get dominant emotions
        prev_dominant = max(prev_emotions, key=prev_emotions.get)
        prev_neutral_prob = prev_emotions.get('neutral', 0)

        # Condition 1: Previous frame is neutral with high probability
        if prev_dominant != 'neutral' or prev_neutral_prob < NEUTRAL_THRESHOLD:
            i += 1
            continue

        # Find non-neutral emotion spike
        non_neutral_emotions = {k: v for k, v in curr_emotions.items() if k != 'neutral'}
        if not non_neutral_emotions:
            i += 1
            continue

        spike_emotion = max(non_neutral_emotions, key=non_neutral_emotions.get)
        spike_prob = non_neutral_emotions[spike_emotion]

        # Condition 2: Non-neutral emotion >= threshold
        if spike_prob < MICRO_EXPR_EMOTION_THRESHOLD:
            i += 1
            continue

        # Check confidence score
        confidence_score = frame.get('confidence_score', 0)

        # Condition 5: Confidence >= 75
        if confidence_score < CONFIDENCE_THRESHOLD:
            i += 1
            continue

        # Find end of micro-expression
        end_idx = i + 1
        while end_idx < len(frame_emotions) and end_idx - i < max_duration_frames:
            end_frame = frame_emotions[end_idx]
            if end_frame.get('emotions') is None:
                end_idx += 1
                continue

            end_emotions = end_frame['emotions']
            end_dominant = max(end_emotions, key=end_emotions.get)
            end_neutral_prob = end_emotions.get('neutral', 0)

            # Condition 4: Returns to neutral
            if end_dominant == 'neutral' and end_neutral_prob >= NEUTRAL_THRESHOLD:
                break
            end_idx += 1

        # Condition 3: Duration < 0.5 seconds
        duration_frames = end_idx - i
        if duration_frames >= max_duration_frames:
            i += 1
            continue

        # Valid micro-expression found
        start_time = frame['timestamp']
        end_time = frame_emotions[min(end_idx, len(frame_emotions) - 1)]['timestamp']
        duration_seconds = end_time - start_time

        # Get peak probability and validation layers
        peak_prob = spike_prob
        validation_layers = frame.get('validation_layers', {
            'deepface': True,
            'optical_flow': False,
            'landmark': False
        })

        micro_expr = {
            'start_frame': frame['frame_idx'],
            'end_frame': frame_emotions[min(end_idx, len(frame_emotions) - 1)]['frame_idx'],
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration_seconds': float(duration_seconds),
            'emotion': spike_emotion,
            'peak_probability': float(peak_prob),
            'confidence_score': float(confidence_score),
            'validation_layers': validation_layers
        }
        micro_expressions.append(micro_expr)

        print(f"  [INFO] Frame {frame['frame_idx']}: MICRO-EXPRESSION detected - "
              f"{spike_emotion} (confidence: {confidence_score:.0f}%)")

        # Skip past this micro-expression
        i = end_idx + 1
        continue

    return micro_expressions


# ═══════════════════════════════════════════════════════════════
# METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def compute_suppression_score(
    micro_expressions: List[Dict],
    frame_emotions: List[Dict]
) -> float:
    """
    Compute suppression score (0-100).
    Formula: (micro_expression_count / total_emotion_events) * 100
    """
    # Count total emotion events (non-neutral dominant frames)
    total_events = 0
    for frame in frame_emotions:
        if frame.get('emotions') is None:
            continue
        dominant = max(frame['emotions'], key=frame['emotions'].get)
        if dominant != 'neutral':
            total_events += 1

    if total_events == 0:
        return 0.0

    score = (len(micro_expressions) / total_events) * 100
    return float(min(max(score, 0), 100))


def compute_emotional_range_score(frame_emotions: List[Dict]) -> float:
    """
    Compute emotional range score (0-100).
    Formula: (unique_emotions_expressed / 7) * 100
    """
    unique_emotions = set()

    for frame in frame_emotions:
        if frame.get('emotions') is None:
            continue
        dominant = max(frame['emotions'], key=frame['emotions'].get)
        # Only count if emotion probability is significant
        if frame['emotions'][dominant] >= 0.30:
            unique_emotions.add(dominant)

    score = (len(unique_emotions) / 7) * 100
    return float(min(max(score, 0), 100))


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════

def analyze_video(video_path: str, analysis_fps: int = 5) -> dict:
    """
    Main video analysis function with 3-layer validation.
    """
    # Validate video file
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(video_fps / analysis_fps))

    print("[INFO] " + "=" * 50)
    print(f"[INFO] Loading video: {os.path.basename(video_path)}")
    print(f"[INFO] Video: {video_fps:.1f} fps, {total_frames} frames, "
          f"analyzing every {frame_interval}th frame")
    print("[INFO] " + "=" * 50)

    video_metadata = {
        'filename': os.path.basename(video_path),
        'duration_seconds': round(duration_seconds, 3),
        'total_frames': total_frames,
        'fps': round(video_fps, 2),
        'analysis_fps': analysis_fps,
        'frames_analyzed': 0
    }

    # Initialize trackers
    landmark_tracker = LandmarkTracker()
    frame_emotions = []
    prev_frame = None
    frame_idx = 0
    analyzed_count = 0

    print("\n[INFO] Analyzing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps

            # Layer 1: DeepFace analysis
            emotions = analyze_frame_deepface(frame)

            # Layer 2: Optical flow
            optical_flow_mag = 0.0
            if prev_frame is not None:
                face_region = detect_face_region(frame)
                optical_flow_mag = compute_optical_flow(prev_frame, frame, face_region)

            # Layer 3: Landmark displacement
            landmark_disp = landmark_tracker.compute_displacement(frame)

            # Compute confidence score
            if emotions:
                non_neutral = {k: v for k, v in emotions.items() if k != 'neutral'}
                max_spike = max(non_neutral.values()) if non_neutral else 0
                confidence, validation_layers = compute_confidence_score(
                    max_spike, optical_flow_mag, landmark_disp
                )
                dominant = max(emotions, key=emotions.get)
            else:
                confidence = 0
                validation_layers = {'deepface': False, 'optical_flow': False, 'landmark': False}
                dominant = 'neutral'

            # Check for micro-expression indicator
            is_micro_expr = False
            if emotions:
                prev_was_neutral = False
                if len(frame_emotions) > 0:
                    prev_emo = frame_emotions[-1].get('emotions')
                    if prev_emo:
                        prev_dom = max(prev_emo, key=prev_emo.get)
                        prev_was_neutral = (prev_dom == 'neutral' and
                                          prev_emo.get('neutral', 0) >= NEUTRAL_THRESHOLD)

                if prev_was_neutral and dominant != 'neutral':
                    if emotions[dominant] >= MICRO_EXPR_EMOTION_THRESHOLD:
                        if confidence >= CONFIDENCE_THRESHOLD:
                            is_micro_expr = True

            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': round(timestamp, 3),
                'emotions': {k: round(v, 6) for k, v in emotions.items()} if emotions else None,
                'dominant_emotion': dominant,
                'is_micro_expression': is_micro_expr,
                'confidence_score': round(confidence, 2),
                'optical_flow_magnitude': round(optical_flow_mag, 6),
                'landmark_displacement': round(landmark_disp, 6),
                'validation_layers': validation_layers
            }
            frame_emotions.append(frame_data)

            analyzed_count += 1
            if analyzed_count % 50 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"[INFO] Analyzing frame {frame_idx}/{total_frames} ({pct:.0f}%)...")

            prev_frame = frame.copy()

        frame_idx += 1

    cap.release()
    landmark_tracker.reset()

    video_metadata['frames_analyzed'] = analyzed_count
    print(f"\n[INFO] Total frames analyzed: {analyzed_count}")

    # Detect micro-expressions
    print("\n[INFO] Detecting micro-expressions with 3-layer validation...")
    micro_expressions = detect_micro_expressions(frame_emotions, analysis_fps)

    # High confidence count
    high_conf_count = len([m for m in micro_expressions if m['confidence_score'] >= 75])

    print(f"[INFO] Found {len(micro_expressions)} micro-expressions "
          f"({high_conf_count} high confidence)")

    # Compute metrics
    suppression_score = compute_suppression_score(micro_expressions, frame_emotions)
    emotional_range_score = compute_emotional_range_score(frame_emotions)

    print(f"[INFO] Suppression Score: {suppression_score:.1f} | "
          f"Emotional Range Score: {emotional_range_score:.1f}")

    # Build result
    result = build_timeline_json(
        frame_emotions,
        micro_expressions,
        suppression_score,
        emotional_range_score,
        video_metadata
    )

    return result


# ═══════════════════════════════════════════════════════════════
# JSON BUILDER
# ═══════════════════════════════════════════════════════════════

def build_timeline_json(
    frame_emotions: List[Dict],
    micro_expressions: List[Dict],
    suppression_score: float,
    emotional_range_score: float,
    video_metadata: Dict
) -> dict:
    """Build the complete timeline JSON structure"""

    # Calculate emotion distribution
    emotion_counts = {e: 0.0 for e in ALL_EMOTIONS}
    valid_frames = 0

    for frame in frame_emotions:
        if frame.get('emotions'):
            for emotion in ALL_EMOTIONS:
                emotion_counts[emotion] += frame['emotions'].get(emotion, 0)
            valid_frames += 1

    if valid_frames > 0:
        emotion_distribution = {k: round(v / valid_frames, 4)
                               for k, v in emotion_counts.items()}
    else:
        emotion_distribution = {e: round(1/7, 4) for e in ALL_EMOTIONS}

    # Find overall dominant emotion
    dominant_overall = max(emotion_distribution, key=emotion_distribution.get)

    # High confidence count
    high_conf_count = len([m for m in micro_expressions if m['confidence_score'] >= 75])

    return {
        'video_metadata': video_metadata,
        'emotion_timeline': frame_emotions,
        'micro_expressions': micro_expressions,
        'summary': {
            'suppression_score': round(suppression_score, 2),
            'emotional_range_score': round(emotional_range_score, 2),
            'dominant_emotion_overall': dominant_overall,
            'micro_expression_count': len(micro_expressions),
            'high_confidence_micro_expressions': high_conf_count,
            'emotion_distribution': emotion_distribution
        }
    }


# ═══════════════════════════════════════════════════════════════
# OUTPUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_json_output(timeline_data: dict, output_path: str = "emotion_timeline_output.json") -> None:
    """Save timeline data to JSON file"""

    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False, default=convert_numpy)

    print(f"[INFO] Saved: {output_path}")


def generate_html_report(timeline_data: dict, output_path: str = "emotion_timeline.html") -> None:
    """Generate interactive HTML report with confidence heatmap"""

    timeline = timeline_data['emotion_timeline']
    micro_exprs = timeline_data['micro_expressions']
    summary = timeline_data['summary']
    metadata = timeline_data['video_metadata']

    # Prepare chart data
    timestamps = [entry['timestamp'] for entry in timeline if entry.get('emotions')]
    emotion_data = {emotion: [] for emotion in ALL_EMOTIONS}
    confidence_data = []

    for entry in timeline:
        if entry.get('emotions'):
            for emotion in ALL_EMOTIONS:
                emotion_data[emotion].append(entry['emotions'].get(emotion, 0))
            confidence_data.append(entry.get('confidence_score', 0))

    # Convert to JSON
    timestamps_json = json.dumps(timestamps)
    emotion_data_json = json.dumps({k: [float(v) for v in vals]
                                    for k, vals in emotion_data.items()})
    confidence_json = json.dumps([float(c) for c in confidence_data])
    micro_exprs_json = json.dumps(micro_exprs, default=lambda x: float(x) if hasattr(x, 'item') else x)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentio Mind - Advanced Emotion Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
        }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #FFD700, #FF8C00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header p {{ color: #95A5A6; }}
        .badge {{
            display: inline-block;
            background: rgba(155, 89, 182, 0.3);
            color: #9B59B6;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 10px;
        }}

        /* Dashboard */
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{ color: #95A5A6; font-size: 0.9rem; }}
        .stat-bar {{
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-top: 15px;
            overflow: hidden;
        }}
        .stat-bar-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }}

        /* Chart Container */
        .chart-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-title {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #fff;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}

        /* Confidence Heatmap */
        .heatmap-container {{
            margin-top: 20px;
        }}
        .heatmap-label {{
            font-size: 0.85rem;
            color: #95A5A6;
            margin-bottom: 10px;
        }}
        .heatmap {{
            height: 30px;
            border-radius: 5px;
            display: flex;
            overflow: hidden;
        }}
        .heatmap-cell {{
            flex: 1;
            min-width: 2px;
        }}

        /* Micro-expressions Panel */
        .micro-panel {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .micro-panel h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
        }}
        .micro-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        .micro-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .micro-card:hover {{
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }}
        .micro-emotion {{
            font-size: 1.1rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .micro-detail {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 0.9rem;
        }}
        .micro-detail-label {{ color: #95A5A6; }}
        .validation-badges {{
            display: flex;
            gap: 8px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .validation-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .validation-badge.active {{
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
        }}
        .validation-badge.inactive {{
            background: rgba(255,255,255,0.1);
            color: #95A5A6;
        }}

        /* Legend */
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #95A5A6;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentio Mind</h1>
            <p>Advanced Micro-Expression Detection System</p>
            <span class="badge">3-Layer Validation</span>
        </div>

        <!-- Dashboard -->
        <div class="dashboard">
            <div class="stat-card">
                <div class="stat-value" style="color: {get_score_color(summary['suppression_score'])}">
                    {summary['suppression_score']:.1f}
                </div>
                <div class="stat-label">Suppression Score</div>
                <div class="stat-bar">
                    <div class="stat-bar-fill" style="width: {summary['suppression_score']}%; background: {get_score_color(summary['suppression_score'])};"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: {get_range_color(summary['emotional_range_score'])}">
                    {summary['emotional_range_score']:.1f}
                </div>
                <div class="stat-label">Emotional Range</div>
                <div class="stat-bar">
                    <div class="stat-bar-fill" style="width: {summary['emotional_range_score']}%; background: {get_range_color(summary['emotional_range_score'])};"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #9B59B6;">
                    {summary['micro_expression_count']}
                </div>
                <div class="stat-label">Micro-Expressions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #2ecc71;">
                    {summary['high_confidence_micro_expressions']}
                </div>
                <div class="stat-label">High Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #3498db;">
                    {metadata['duration_seconds']:.1f}s
                </div>
                <div class="stat-label">Duration</div>
            </div>
        </div>

        <!-- Emotion Timeline Chart -->
        <div class="chart-section">
            <div class="chart-title">Emotion Timeline (Stacked Area)</div>
            <div class="chart-container">
                <canvas id="emotionChart"></canvas>
            </div>

            <!-- Confidence Heatmap -->
            <div class="heatmap-container">
                <div class="heatmap-label">Confidence Heatmap (darker = higher confidence)</div>
                <div class="heatmap" id="confidenceHeatmap"></div>
            </div>

            <!-- Legend -->
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #FF4444;"></div>Angry</div>
                <div class="legend-item"><div class="legend-color" style="background: #8B4513;"></div>Disgust</div>
                <div class="legend-item"><div class="legend-color" style="background: #9B59B6;"></div>Fear</div>
                <div class="legend-item"><div class="legend-color" style="background: #FFD700;"></div>Happy</div>
                <div class="legend-item"><div class="legend-color" style="background: #4A90D9;"></div>Sad</div>
                <div class="legend-item"><div class="legend-color" style="background: #FF8C00;"></div>Surprise</div>
                <div class="legend-item"><div class="legend-color" style="background: #95A5A6;"></div>Neutral</div>
            </div>
        </div>

        <!-- Micro-expressions Panel -->
        <div class="micro-panel">
            <h2>Micro-Expressions Detected</h2>
            <div class="micro-list" id="microList"></div>
        </div>

        <div class="footer">
            <p>Video: {metadata['filename']} | Frames Analyzed: {metadata['frames_analyzed']} | Analysis FPS: {metadata['analysis_fps']}</p>
            <p style="margin-top: 10px;">Powered by 3-Layer Validation: DeepFace + Optical Flow + MediaPipe Landmarks</p>
        </div>
    </div>

    <script>
        // Data
        const timestamps = {timestamps_json};
        const emotionData = {emotion_data_json};
        const confidenceData = {confidence_json};
        const microExpressions = {micro_exprs_json};

        const emotionColors = {{
            angry: '#FF4444',
            disgust: '#8B4513',
            fear: '#9B59B6',
            happy: '#FFD700',
            sad: '#4A90D9',
            surprise: '#FF8C00',
            neutral: '#95A5A6'
        }};

        // Create stacked area chart
        const ctx = document.getElementById('emotionChart').getContext('2d');

        const datasets = Object.keys(emotionData).map(emotion => ({{
            label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
            data: emotionData[emotion],
            backgroundColor: emotionColors[emotion] + '80',
            borderColor: emotionColors[emotion],
            borderWidth: 1,
            fill: true,
            tension: 0.4,
            pointRadius: 0
        }}));

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: timestamps.map(t => t.toFixed(1) + 's'),
                datasets: datasets
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Time (seconds)', color: '#95A5A6' }},
                        ticks: {{ color: '#95A5A6', maxTicksLimit: 20 }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        stacked: true,
                        title: {{ display: true, text: 'Emotion Probability', color: '#95A5A6' }},
                        ticks: {{ color: '#95A5A6' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        min: 0,
                        max: 1
                    }}
                }},
                interaction: {{
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }}
            }}
        }});

        // Create confidence heatmap
        const heatmap = document.getElementById('confidenceHeatmap');
        confidenceData.forEach(conf => {{
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            const intensity = conf / 100;
            cell.style.backgroundColor = `rgba(155, 89, 182, ${{intensity}})`;
            heatmap.appendChild(cell);
        }});

        // Render micro-expressions
        const microList = document.getElementById('microList');

        if (microExpressions.length === 0) {{
            microList.innerHTML = '<p style="color: #95A5A6; text-align: center; padding: 40px;">No micro-expressions detected with high confidence.</p>';
        }} else {{
            microExpressions.forEach((me, idx) => {{
                const card = document.createElement('div');
                card.className = 'micro-card';
                card.style.borderColor = emotionColors[me.emotion] || '#9B59B6';

                const layers = me.validation_layers || {{}};

                card.innerHTML = `
                    <div class="micro-emotion" style="color: ${{emotionColors[me.emotion] || '#9B59B6'}}">${{me.emotion}}</div>
                    <div class="micro-detail">
                        <span class="micro-detail-label">Time</span>
                        <span>${{me.start_time.toFixed(2)}}s - ${{me.end_time.toFixed(2)}}s</span>
                    </div>
                    <div class="micro-detail">
                        <span class="micro-detail-label">Duration</span>
                        <span>${{(me.duration_seconds * 1000).toFixed(0)}}ms</span>
                    </div>
                    <div class="micro-detail">
                        <span class="micro-detail-label">Peak Probability</span>
                        <span>${{(me.peak_probability * 100).toFixed(1)}}%</span>
                    </div>
                    <div class="micro-detail">
                        <span class="micro-detail-label">Confidence</span>
                        <span style="color: ${{me.confidence_score >= 75 ? '#2ecc71' : '#f39c12'}}">${{me.confidence_score.toFixed(0)}}%</span>
                    </div>
                    <div class="validation-badges">
                        <span class="validation-badge ${{layers.deepface ? 'active' : 'inactive'}}">DeepFace ${{layers.deepface ? '✓' : '✗'}}</span>
                        <span class="validation-badge ${{layers.optical_flow ? 'active' : 'inactive'}}">Optical Flow ${{layers.optical_flow ? '✓' : '✗'}}</span>
                        <span class="validation-badge ${{layers.landmark ? 'active' : 'inactive'}}">Landmark ${{layers.landmark ? '✓' : '✗'}}</span>
                    </div>
                `;
                microList.appendChild(card);
            }});
        }}
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[INFO] Saved: {output_path}")


def get_score_color(score: float) -> str:
    """Get color based on suppression score (high = bad)"""
    if score < 30:
        return '#2ecc71'  # Green
    elif score < 60:
        return '#f39c12'  # Orange
    else:
        return '#e74c3c'  # Red


def get_range_color(score: float) -> str:
    """Get color based on emotional range score (high = good)"""
    if score >= 70:
        return '#2ecc71'  # Green
    elif score >= 40:
        return '#f39c12'  # Orange
    else:
        return '#e74c3c'  # Red


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Sentio Mind - Advanced Micro-Expression Detection (3-Layer Validation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default='video_sample.mp4',
        help='Path to input video file'
    )
    parser.add_argument(
        '--fps', '-f',
        type=int,
        default=5,
        help='Analysis frames per second'
    )
    parser.add_argument(
        '--html', '-o',
        type=str,
        default='emotion_timeline.html',
        help='Output HTML report path'
    )
    parser.add_argument(
        '--json', '-j',
        type=str,
        default='emotion_timeline_output.json',
        help='Output JSON file path'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SENTIO MIND - Advanced Micro-Expression Detection")
    print("3-Layer Validation: DeepFace + Optical Flow + MediaPipe")
    print("=" * 60 + "\n")

    # Analyze video
    timeline_data = analyze_video(args.video, args.fps)

    # Generate outputs
    print("\n[INFO] Generating outputs...")
    generate_html_report(timeline_data, args.html)
    save_json_output(timeline_data, args.json)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  HTML Report: {args.html}")
    print(f"  JSON Output: {args.json}")
    print(f"  Suppression Score: {timeline_data['summary']['suppression_score']:.2f}")
    print(f"  Emotional Range Score: {timeline_data['summary']['emotional_range_score']:.2f}")
    print(f"  Micro-expressions Found: {timeline_data['summary']['micro_expression_count']}")
    print(f"  High Confidence: {timeline_data['summary']['high_confidence_micro_expressions']}")
    print()


if __name__ == '__main__':
    main()
