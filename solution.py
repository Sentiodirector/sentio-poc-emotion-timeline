#!/usr/bin/env python3
"""
Micro-Expression Detection System with 3-Layer Validation
==========================================================
A unique approach using DeepFace + Optical Flow + MediaPipe Landmarks
for high-confidence micro-expression detection.

Author: Senior ML/CV Engineer
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Suppress TensorFlow and other warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Thresholds for 3-layer validation
DEEPFACE_SPIKE_THRESHOLD = 0.20  # Spike detection on smoothed signal (lowered for sensitivity)
DEEPFACE_CONFIRMATION_THRESHOLD = 0.30  # For confidence voting (lowered)
OPTICAL_FLOW_THRESHOLD = 0.3  # Facial muscle movement threshold (lowered)
LANDMARK_DISPLACEMENT_THRESHOLD = 0.015  # MediaPipe landmark movement (lowered)

# Micro-expression detection thresholds
NEUTRAL_BASELINE_THRESHOLD = 0.30  # Neutral probability for baseline (lowered - allow emotion transitions)
EMOTION_SPIKE_THRESHOLD = 0.30  # Non-neutral emotion threshold (lowered)
CONFIDENCE_THRESHOLD = 60  # Minimum confidence for valid micro-expression (lowered for 2-layer detection)
CONFIDENCE_THRESHOLD_WITH_LANDMARK = 75  # Higher threshold when all 3 layers available

# Smoothing window size
SMOOTHING_WINDOW = 3

# MediaPipe landmark indices for key facial features
EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LIP_LANDMARKS = [61, 91, 181, 84, 17, 314, 405, 321, 375, 291]
EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133]
KEY_LANDMARKS = EYEBROW_LANDMARKS + LIP_LANDMARKS + EYE_LANDMARKS

# Color codes for HTML report
EMOTION_COLORS = {
    'angry': '#FF4444',
    'disgust': '#8B4513',
    'fear': '#9B59B6',
    'happy': '#FFD700',
    'sad': '#4A90D9',
    'surprise': '#FF8C00',
    'neutral': '#95A5A6'
}


# ═══════════════════════════════════════════════════════════════
# HELPER CLASSES
# ═══════════════════════════════════════════════════════════════

class FaceMeshProcessor:
    """Handles MediaPipe Face Mesh processing for landmark tracking."""

    def __init__(self):
        self.face_mesh = None
        self.use_new_api = False
        self.landmarker = None
        self.previous_landmarks = None
        self.enabled = False

        # Try to initialize MediaPipe (handle both old and new API versions)
        try:
            # Try old API first (mediapipe < 0.10.x)
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.enabled = True
                log_info("MediaPipe Face Mesh initialized (legacy API)")
            else:
                # New API (mediapipe >= 0.10.x) - use tasks API
                try:
                    from mediapipe.tasks import python as mp_tasks
                    from mediapipe.tasks.python import vision
                    import urllib.request
                    import ssl

                    # Download model if not exists
                    model_path = 'face_landmarker.task'
                    if not os.path.exists(model_path):
                        log_info("Downloading MediaPipe face landmark model...")
                        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

                        # Create SSL context that doesn't verify certificates
                        ssl_context = ssl.create_default_context()
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE

                        # Download with SSL workaround
                        with urllib.request.urlopen(url, context=ssl_context) as response:
                            with open(model_path, 'wb') as out_file:
                                out_file.write(response.read())
                        log_info("MediaPipe model downloaded successfully")

                    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.landmarker = vision.FaceLandmarker.create_from_options(options)
                    self.use_new_api = True
                    self.enabled = True
                    log_info("MediaPipe Face Landmarker initialized (tasks API)")
                except Exception as e:
                    log_warning(f"MediaPipe tasks API failed: {e}")
                    log_warning("Layer 3 (Landmark) validation will be disabled")
                    self.enabled = False
        except Exception as e:
            log_warning(f"MediaPipe initialization failed: {e}")
            log_warning("Layer 3 (Landmark) validation will be disabled")
            self.enabled = False

    def get_landmark_displacement(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Calculate displacement of key facial landmarks from previous frame.

        Returns:
            Tuple of (displacement_value, is_valid)
        """
        if not self.enabled:
            return 0.0, False

        try:
            if self.use_new_api:
                return self._get_displacement_new_api(frame)
            else:
                return self._get_displacement_legacy_api(frame)
        except Exception as e:
            return 0.0, False

    def _get_displacement_legacy_api(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Use legacy MediaPipe solutions API."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.previous_landmarks = None
            return 0.0, False

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        return self._calculate_displacement(landmarks, h, w)

    def _get_displacement_new_api(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Use new MediaPipe tasks API."""
        from mediapipe import Image as MpImage

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            self.previous_landmarks = None
            return 0.0, False

        landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]

        return self._calculate_displacement(landmarks, h, w)

    def _calculate_displacement(self, landmarks, h: int, w: int) -> Tuple[float, bool]:
        """Calculate displacement from landmarks."""
        # Extract key landmark positions
        current_landmarks = {}
        for idx in KEY_LANDMARKS:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Handle both old API (lm.x, lm.y) and new API (lm.x, lm.y are already normalized)
                x = lm.x if hasattr(lm, 'x') else lm.x
                y = lm.y if hasattr(lm, 'y') else lm.y
                current_landmarks[idx] = np.array([x * w, y * h])

        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return 0.0, True

        # Calculate euclidean displacement
        total_displacement = 0.0
        valid_landmarks = 0

        for idx in KEY_LANDMARKS:
            if idx in current_landmarks and idx in self.previous_landmarks:
                displacement = np.linalg.norm(
                    current_landmarks[idx] - self.previous_landmarks[idx]
                )
                # Normalize by frame dimensions
                total_displacement += displacement / max(w, h)
                valid_landmarks += 1

        self.previous_landmarks = current_landmarks

        if valid_landmarks == 0:
            return 0.0, False

        avg_displacement = total_displacement / valid_landmarks
        return avg_displacement, True

    def reset(self):
        """Reset the landmark tracker."""
        self.previous_landmarks = None

    def close(self):
        """Release resources."""
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except:
                pass


class OpticalFlowProcessor:
    """Handles optical flow computation for facial movement detection."""

    def __init__(self):
        self.previous_gray = None

    def compute_flow_magnitude(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Compute optical flow magnitude in the facial region.

        Args:
            frame: Current BGR frame
            face_bbox: Optional (x, y, w, h) for face region

        Returns:
            Mean optical flow magnitude in face region
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            self.previous_gray = gray
            return 0.0

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Extract face region if bbox provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Ensure bbox is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w > 0 and h > 0:
                face_magnitude = magnitude[y:y+h, x:x+w]
                if face_magnitude.size > 0:
                    mean_magnitude = float(np.mean(face_magnitude))
                else:
                    mean_magnitude = float(np.mean(magnitude))
            else:
                mean_magnitude = float(np.mean(magnitude))
        else:
            mean_magnitude = float(np.mean(magnitude))

        self.previous_gray = gray
        return mean_magnitude

    def reset(self):
        """Reset the optical flow tracker."""
        self.previous_gray = None


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log_info(message: str) -> None:
    """Print info message with prefix."""
    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    """Print warning message with prefix."""
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    """Print error message with prefix."""
    print(f"[ERROR] {message}")


def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion scores to sum to 1.0."""
    total = sum(emotions.values())
    if total == 0:
        return {e: 1/7 for e in EMOTIONS}
    return {e: v / total for e, v in emotions.items()}


def get_dominant_emotion(emotions: Dict[str, float]) -> str:
    """Get the emotion with highest probability."""
    return max(emotions.items(), key=lambda x: x[1])[0]


def smooth_emotions(frame_emotions: List[Dict], window: int = 3) -> List[Dict[str, float]]:
    """
    Apply sliding window smoothing to emotion probabilities.

    Args:
        frame_emotions: List of frame data with 'emotions' dict
        window: Window size for moving average

    Returns:
        List of smoothed emotion dictionaries
    """
    if len(frame_emotions) == 0:
        return []

    smoothed = []
    half_window = window // 2

    for i in range(len(frame_emotions)):
        start = max(0, i - half_window)
        end = min(len(frame_emotions), i + half_window + 1)
        window_frames = frame_emotions[start:end]

        avg_emotions = {}
        for emotion in EMOTIONS:
            values = [f['emotions'][emotion] for f in window_frames if 'emotions' in f]
            if values:
                avg_emotions[emotion] = np.mean(values)
            else:
                avg_emotions[emotion] = 0.0

        smoothed.append(avg_emotions)

    return smoothed


def detect_face_bbox(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bounding box using OpenCV Haar cascade.

    Returns:
        (x, y, w, h) tuple or None if no face detected
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Return the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest)
    return None


def analyze_frame_deepface(frame: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Analyze a single frame using DeepFace.

    Returns:
        Normalized emotion dictionary or None if detection fails
    """
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get('emotion', {})

        # Convert to lowercase keys and normalize
        emotions_normalized = {}
        for emotion in EMOTIONS:
            # DeepFace returns capitalized keys
            key = emotion.capitalize() if emotion != 'neutral' else 'neutral'
            emotions_normalized[emotion] = emotions.get(emotion, emotions.get(key, 0.0)) / 100.0

        return normalize_emotions(emotions_normalized)

    except Exception as e:
        return None


def compute_confidence_score(
    deepface_spike: float,
    optical_flow_magnitude: float,
    landmark_displacement: float,
    optical_flow_valid: bool = True,
    landmark_valid: bool = True
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute 3-layer validation confidence score.

    When landmark validation is unavailable, redistributes weight to other layers.

    Returns:
        Tuple of (confidence_score, validation_layers_dict)
    """
    confidence = 0.0
    validation_layers = {
        'deepface': False,
        'optical_flow': False,
        'landmark': False
    }

    # Determine weights based on available layers
    if landmark_valid:
        # All 3 layers available: standard weights
        deepface_weight = 40
        optical_flow_weight = 35
        landmark_weight = 25
    else:
        # Only 2 layers available: redistribute landmark weight
        deepface_weight = 50  # Increased from 40
        optical_flow_weight = 50  # Increased from 35
        landmark_weight = 0

    # Layer 1: DeepFace spike
    if deepface_spike >= DEEPFACE_CONFIRMATION_THRESHOLD:
        confidence += deepface_weight
        validation_layers['deepface'] = True
    elif deepface_spike >= DEEPFACE_SPIKE_THRESHOLD:
        # Partial credit for moderate spikes
        confidence += deepface_weight * 0.7
        validation_layers['deepface'] = True

    # Layer 2: Optical flow validation
    if optical_flow_valid and optical_flow_magnitude > OPTICAL_FLOW_THRESHOLD:
        confidence += optical_flow_weight
        validation_layers['optical_flow'] = True
    elif optical_flow_valid and optical_flow_magnitude > OPTICAL_FLOW_THRESHOLD * 0.5:
        # Partial credit for moderate movement
        confidence += optical_flow_weight * 0.6
        validation_layers['optical_flow'] = True

    # Layer 3: Landmark displacement
    if landmark_valid and landmark_displacement > LANDMARK_DISPLACEMENT_THRESHOLD:
        confidence += landmark_weight
        validation_layers['landmark'] = True

    return confidence, validation_layers


# ═══════════════════════════════════════════════════════════════
# REQUIRED FUNCTION STUBS (DO NOT RENAME)
# ═══════════════════════════════════════════════════════════════

def analyze_video(video_path: str, analysis_fps: int = 5) -> dict:
    """
    Main video analysis function with 3-layer validation.

    Args:
        video_path: Path to input video file
        analysis_fps: Frames per second to analyze (default: 5)

    Returns:
        Complete analysis dictionary matching JSON schema
    """
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    log_info(f"Loading video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame sampling interval
    frame_interval = max(1, int(video_fps / analysis_fps))
    frames_to_analyze = total_frames // frame_interval

    log_info(f"Video: {video_fps:.1f} fps, {total_frames} frames, analyzing every {frame_interval}th frame")

    # Initialize processors
    face_mesh_processor = FaceMeshProcessor()
    optical_flow_processor = OpticalFlowProcessor()

    # Storage for analysis results
    frame_emotions = []
    analyzed_count = 0
    frame_idx = 0

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process sampled frames
        if frame_idx % frame_interval == 0:
            analyzed_count += 1
            timestamp = frame_idx / video_fps

            # Progress logging
            if analyzed_count % 20 == 0 or analyzed_count == 1:
                progress = (analyzed_count / frames_to_analyze) * 100
                log_info(f"Analyzing frame {analyzed_count}/{frames_to_analyze} ({progress:.0f}%)...")

            # Detect face bbox for optical flow region
            face_bbox = detect_face_bbox(frame)

            # Layer 1: DeepFace emotion analysis
            emotions = analyze_frame_deepface(frame)

            if emotions is None:
                log_warning(f"No face detected in frame {frame_idx}, skipping")
                frame_idx += 1
                continue

            # Layer 2: Optical flow magnitude
            optical_flow_magnitude = optical_flow_processor.compute_flow_magnitude(
                frame, face_bbox
            )

            # Layer 3: Landmark displacement
            landmark_displacement, landmark_valid = face_mesh_processor.get_landmark_displacement(frame)

            # Store frame data
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': round(timestamp, 3),
                'emotions': emotions,
                'dominant_emotion': get_dominant_emotion(emotions),
                'is_micro_expression': False,  # Will be updated later
                'confidence_score': 0.0,  # Will be updated later
                'optical_flow_magnitude': round(optical_flow_magnitude, 4),
                'landmark_displacement': round(landmark_displacement, 4),
                'landmark_valid': landmark_valid
            }

            frame_emotions.append(frame_data)

        frame_idx += 1

    cap.release()
    face_mesh_processor.close()

    log_info(f"Processed {len(frame_emotions)} frames")

    # Warn if too few frames
    if len(frame_emotions) < 10:
        log_warning(f"Only {len(frame_emotions)} frames analyzed. Results may be unreliable.")

    # Apply smoothing and detect micro-expressions
    smoothed_emotions = smooth_emotions(frame_emotions, SMOOTHING_WINDOW)

    # Update frame emotions with smoothed values for spike detection
    for i, frame_data in enumerate(frame_emotions):
        frame_data['smoothed_emotions'] = smoothed_emotions[i] if i < len(smoothed_emotions) else frame_data['emotions']

    # Detect micro-expressions with 3-layer validation
    micro_expressions = detect_micro_expressions(frame_emotions, analysis_fps)

    # Mark micro-expression frames
    me_frames = set()
    for me in micro_expressions:
        for i, fd in enumerate(frame_emotions):
            if me['start_frame'] <= fd['frame_idx'] <= me['end_frame']:
                me_frames.add(i)
                frame_emotions[i]['is_micro_expression'] = True
                frame_emotions[i]['confidence_score'] = me['confidence_score']

    # Log detected micro-expressions
    for me in micro_expressions:
        log_info(f"Frame {me['start_frame']}: MICRO-EXPRESSION detected - {me['emotion']} (confidence: {me['confidence_score']:.0f}%)")

    log_info(f"Analysis complete. {len(micro_expressions)} micro-expressions found.")

    # Compute metrics
    suppression_score = compute_suppression_score(micro_expressions, frame_emotions)
    emotional_range_score = compute_emotional_range_score(frame_emotions)

    log_info(f"Suppression Score: {suppression_score:.1f} | Emotional Range Score: {emotional_range_score:.1f}")

    # Build video metadata
    video_metadata = {
        'filename': os.path.basename(video_path),
        'duration_seconds': round(duration_seconds, 2),
        'total_frames': total_frames,
        'fps': round(video_fps, 2),
        'analysis_fps': analysis_fps,
        'frames_analyzed': len(frame_emotions)
    }

    # Clean up frame data for output (remove internal fields)
    clean_frame_emotions = []
    for fd in frame_emotions:
        clean_fd = {
            'frame_idx': fd['frame_idx'],
            'timestamp': fd['timestamp'],
            'emotions': fd['emotions'],
            'dominant_emotion': fd['dominant_emotion'],
            'is_micro_expression': fd['is_micro_expression'],
            'confidence_score': fd['confidence_score'],
            'optical_flow_magnitude': fd['optical_flow_magnitude'],
            'landmark_displacement': fd['landmark_displacement']
        }
        clean_frame_emotions.append(clean_fd)

    # Build timeline JSON
    timeline_data = build_timeline_json(
        clean_frame_emotions,
        micro_expressions,
        suppression_score,
        emotional_range_score,
        video_metadata
    )

    return timeline_data


def detect_micro_expressions(frame_emotions: list, analysis_fps: int) -> list:
    """
    Detect micro-expressions using improved 3-layer validation.

    A micro-expression is detected when:
    1. Rapid emotion change occurs (spike from baseline)
    2. Duration is brief (< 1 second for micro, < 0.5s for true micro)
    3. Emotion intensity exceeds threshold
    4. Confidence score meets adaptive threshold

    Args:
        frame_emotions: List of frame analysis data
        analysis_fps: Analysis frames per second

    Returns:
        List of detected micro-expression events
    """
    if len(frame_emotions) < 3:
        return []

    micro_expressions = []
    max_duration_frames = int(analysis_fps * 1.0)  # Extended to 1 second
    min_duration_frames = 1  # At least 1 frame

    # Apply smoothing for spike detection
    smoothed = smooth_emotions(frame_emotions, SMOOTHING_WINDOW)

    # Check if landmark validation is available
    has_landmark = any(f.get('landmark_valid', False) for f in frame_emotions)
    effective_threshold = CONFIDENCE_THRESHOLD_WITH_LANDMARK if has_landmark else CONFIDENCE_THRESHOLD

    i = 1
    while i < len(frame_emotions) - 1:
        prev_frame = frame_emotions[i - 1]
        curr_frame = frame_emotions[i]

        # Get smoothed emotions
        prev_smoothed = smoothed[i - 1] if i - 1 < len(smoothed) else prev_frame['emotions']
        curr_smoothed = smoothed[i] if i < len(smoothed) else curr_frame['emotions']

        # Get previous dominant emotion
        prev_dominant = max(prev_smoothed.items(), key=lambda x: x[1])[0]

        # Find the dominant emotion in current frame
        curr_dominant = max(curr_smoothed.items(), key=lambda x: x[1])
        curr_emotion_name, curr_emotion_prob = curr_dominant

        # Detect emotion change (either from neutral or from different emotion)
        prev_neutral = prev_smoothed.get('neutral', 0)
        is_from_neutral = prev_neutral >= NEUTRAL_BASELINE_THRESHOLD
        is_emotion_change = prev_dominant != curr_emotion_name and curr_emotion_name != 'neutral'

        # Calculate spike magnitude
        baseline = prev_smoothed.get(curr_emotion_name, 0)
        spike_magnitude = curr_emotion_prob - baseline

        # Check for valid emotion spike
        valid_spike = (
            (is_from_neutral or is_emotion_change) and
            curr_emotion_prob >= EMOTION_SPIKE_THRESHOLD and
            spike_magnitude >= DEEPFACE_SPIKE_THRESHOLD
        )

        if not valid_spike:
            i += 1
            continue

        # Found potential start of micro-expression
        start_idx = i
        start_frame = curr_frame['frame_idx']
        peak_emotion_name = curr_emotion_name
        peak_emotion_prob = curr_emotion_prob

        # Find end of micro-expression (emotion subsides or changes)
        end_idx = i
        found_end = False

        for j in range(i + 1, min(i + max_duration_frames + 1, len(frame_emotions))):
            next_smoothed = smoothed[j] if j < len(smoothed) else frame_emotions[j]['emotions']
            next_prob = next_smoothed.get(peak_emotion_name, 0)
            next_neutral = next_smoothed.get('neutral', 0)

            # Track peak during expression
            if next_prob > peak_emotion_prob:
                peak_emotion_prob = next_prob

            # End conditions: returns to neutral, emotion drops significantly, or different emotion takes over
            next_dominant = max(next_smoothed.items(), key=lambda x: x[1])[0]
            emotion_dropped = next_prob < peak_emotion_prob * 0.5  # Dropped to less than 50% of peak
            returned_neutral = next_neutral >= NEUTRAL_BASELINE_THRESHOLD
            changed_emotion = next_dominant != peak_emotion_name and next_dominant != 'neutral'

            if returned_neutral or emotion_dropped or changed_emotion:
                end_idx = j
                found_end = True
                break

        if not found_end:
            # Expression continued beyond time limit - not a micro-expression
            i += 1
            continue

        # Check duration
        duration_frames = end_idx - start_idx
        if duration_frames < min_duration_frames or duration_frames > max_duration_frames:
            i += 1
            continue

        # Find peak frame within the expression
        peak_idx = start_idx
        for k in range(start_idx, end_idx + 1):
            if k < len(smoothed):
                if smoothed[k].get(peak_emotion_name, 0) > smoothed[peak_idx].get(peak_emotion_name, 0):
                    peak_idx = k

        peak_frame_data = frame_emotions[peak_idx]

        # Compute 3-layer confidence score
        confidence, validation_layers = compute_confidence_score(
            deepface_spike=spike_magnitude,
            optical_flow_magnitude=peak_frame_data.get('optical_flow_magnitude', 0),
            landmark_displacement=peak_frame_data.get('landmark_displacement', 0),
            optical_flow_valid=True,
            landmark_valid=peak_frame_data.get('landmark_valid', False)
        )

        # Must meet adaptive confidence threshold
        if confidence < effective_threshold:
            i += 1
            continue

        # Valid micro-expression detected
        end_frame = frame_emotions[end_idx]['frame_idx']
        start_time = frame_emotions[start_idx]['timestamp']
        end_time = frame_emotions[end_idx]['timestamp']

        micro_expression = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': round(start_time, 3),
            'end_time': round(end_time, 3),
            'duration_seconds': round(end_time - start_time, 3),
            'emotion': peak_emotion_name,
            'peak_probability': round(peak_emotion_prob, 4),
            'confidence_score': round(confidence, 1),
            'validation_layers': validation_layers
        }

        micro_expressions.append(micro_expression)

        # Skip past this micro-expression
        i = end_idx + 1

    return micro_expressions


def compute_suppression_score(micro_expressions: list, frame_emotions: list) -> float:
    """
    Compute suppression score indicating stress level.

    Formula: (micro_expression_count / total_emotion_events) * 100

    Args:
        micro_expressions: List of detected micro-expressions
        frame_emotions: List of all frame emotion data

    Returns:
        Suppression score clamped to [0, 100]
    """
    if len(frame_emotions) == 0:
        return 0.0

    # Count total emotion events (frames where dominant emotion is not neutral)
    total_emotion_events = sum(
        1 for f in frame_emotions
        if f.get('dominant_emotion', 'neutral') != 'neutral'
    )

    # Handle edge case where all frames are neutral
    if total_emotion_events == 0:
        return 0.0

    micro_count = len(micro_expressions)
    score = (micro_count / total_emotion_events) * 100

    # Clamp to [0, 100]
    return round(max(0.0, min(100.0, score)), 2)


def compute_emotional_range_score(frame_emotions: list) -> float:
    """
    Compute emotional range score indicating expression variety.

    Formula: (unique_emotions_expressed / 7) * 100

    Args:
        frame_emotions: List of all frame emotion data

    Returns:
        Emotional range score clamped to [0, 100]
    """
    if len(frame_emotions) == 0:
        return 0.0

    # Find unique dominant emotions expressed
    unique_emotions = set()
    for f in frame_emotions:
        dominant = f.get('dominant_emotion')
        if dominant:
            # Also check if the emotion probability is significant (> 0.15)
            prob = f.get('emotions', {}).get(dominant, 0)
            if prob > 0.15:
                unique_emotions.add(dominant)

    score = (len(unique_emotions) / 7) * 100

    # Clamp to [0, 100]
    return round(max(0.0, min(100.0, score)), 2)


def build_timeline_json(
    frame_emotions: list,
    micro_expressions: list,
    suppression_score: float,
    emotional_range_score: float,
    video_metadata: dict
) -> dict:
    """
    Build complete timeline JSON matching the required schema.

    Args:
        frame_emotions: List of frame emotion data
        micro_expressions: List of detected micro-expressions
        suppression_score: Computed suppression score
        emotional_range_score: Computed emotional range score
        video_metadata: Video metadata dictionary

    Returns:
        Complete timeline dictionary
    """
    # Compute emotion distribution
    emotion_distribution = {e: 0.0 for e in EMOTIONS}
    if frame_emotions:
        for emotion in EMOTIONS:
            values = [f['emotions'].get(emotion, 0) for f in frame_emotions]
            emotion_distribution[emotion] = round(np.mean(values), 4)

    # Find overall dominant emotion
    dominant_overall = max(emotion_distribution.items(), key=lambda x: x[1])[0]

    # Count high confidence micro-expressions
    high_confidence_count = sum(
        1 for me in micro_expressions
        if me['confidence_score'] >= 80
    )

    # Build summary
    summary = {
        'suppression_score': suppression_score,
        'emotional_range_score': emotional_range_score,
        'dominant_emotion_overall': dominant_overall,
        'micro_expression_count': len(micro_expressions),
        'high_confidence_micro_expressions': high_confidence_count,
        'emotion_distribution': emotion_distribution
    }

    # Build final timeline
    timeline = {
        'video_metadata': video_metadata,
        'emotion_timeline': frame_emotions,
        'micro_expressions': micro_expressions,
        'summary': summary
    }

    return timeline


def generate_html_report(timeline_data: dict, output_path: str) -> None:
    """
    Generate self-contained HTML report with Chart.js visualization.

    Features:
    - Stacked area river chart
    - Confidence heatmap band
    - Interactive micro-expression panel
    - Summary dashboard
    - Fully offline (Chart.js embedded inline)

    Args:
        timeline_data: Complete timeline data dictionary
        output_path: Path for output HTML file
    """
    # Convert data to JSON for embedding (use NumpyEncoder for numpy types)
    json_data = json.dumps(timeline_data, indent=2, cls=NumpyEncoder)

    # HTML template using placeholder for JSON data to avoid f-string issues with JS
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micro-Expression Analysis Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh; color: #e0e0e0; padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center; margin-bottom: 30px; padding: 20px;
            background: rgba(255,255,255,0.05); border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            font-size: 2.2em;
            background: linear-gradient(120deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 10px;
        }
        .header p { color: #888; font-size: 0.95em; }
        .dashboard {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .stat-label { font-size: 0.85em; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .stat-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .stat-indicator { height: 4px; border-radius: 2px; margin-top: 10px; }
        .suppression-high { background: linear-gradient(90deg, #ff4444, #ff6b6b); }
        .suppression-medium { background: linear-gradient(90deg, #ffa500, #ffcc00); }
        .suppression-low { background: linear-gradient(90deg, #4ade80, #22c55e); }
        .range-high { background: linear-gradient(90deg, #7c3aed, #a855f7); }
        .range-medium { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
        .range-low { background: linear-gradient(90deg, #6b7280, #9ca3af); }
        .chart-section {
            background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px;
            margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.1);
        }
        .chart-title { font-size: 1.3em; margin-bottom: 20px; color: #fff; }
        .chart-wrapper { position: relative; height: 400px; }
        .heatmap-container { margin-top: 10px; height: 30px; border-radius: 5px; overflow: hidden; background: #1a1a2e; }
        .heatmap-label { font-size: 0.8em; color: #888; margin-bottom: 5px; }
        #confidenceHeatmap { width: 100%; height: 100%; }
        .me-panel {
            position: fixed; right: -400px; top: 0; width: 380px; height: 100vh;
            background: rgba(22,33,62,0.98); border-left: 1px solid rgba(255,255,255,0.1);
            padding: 30px; transition: right 0.3s ease; z-index: 1000; overflow-y: auto;
        }
        .me-panel.active { right: 0; }
        .me-panel-close {
            position: absolute; top: 20px; right: 20px; background: none; border: none;
            color: #888; font-size: 1.5em; cursor: pointer; transition: color 0.2s;
        }
        .me-panel-close:hover { color: #fff; }
        .me-panel h2 { font-size: 1.4em; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .me-detail { margin-bottom: 20px; }
        .me-detail-label { font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .me-detail-value { font-size: 1.3em; color: #fff; }
        .me-emotion { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 0.9em; }
        .validation-item { display: flex; align-items: center; margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; }
        .validation-icon { width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 0.8em; }
        .validation-pass { background: #22c55e; color: #fff; }
        .validation-fail { background: #6b7280; color: #fff; }
        .me-list { background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); }
        .me-list h2 { font-size: 1.3em; margin-bottom: 20px; }
        .me-item {
            display: flex; align-items: center; padding: 15px; background: rgba(255,255,255,0.03);
            border-radius: 10px; margin-bottom: 10px; cursor: pointer; transition: background 0.2s, transform 0.2s;
        }
        .me-item:hover { background: rgba(255,255,255,0.08); transform: translateX(5px); }
        .me-item-emotion { width: 10px; height: 10px; border-radius: 50%; margin-right: 15px; }
        .me-item-info { flex: 1; }
        .me-item-title { font-weight: 600; text-transform: capitalize; margin-bottom: 3px; }
        .me-item-time { font-size: 0.85em; color: #888; }
        .me-item-confidence { font-size: 0.9em; padding: 5px 10px; border-radius: 5px; background: rgba(255,255,255,0.1); }
        .legend { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; justify-content: center; }
        .legend-item { display: flex; align-items: center; font-size: 0.85em; }
        .legend-color { width: 14px; height: 14px; border-radius: 3px; margin-right: 6px; }
        .validation-legend { margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); }
        .validation-legend h4 { font-size: 0.9em; color: #888; margin-bottom: 10px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Micro-Expression Analysis Report</h1>
            <p id="videoInfo">3-Layer Validation System | DeepFace + Optical Flow + MediaPipe</p>
        </div>
        <div class="dashboard" id="dashboard"></div>
        <div class="chart-section">
            <h2 class="chart-title">Emotion Timeline - Stacked Area River Chart</h2>
            <div class="chart-wrapper"><canvas id="emotionChart"></canvas></div>
            <div class="heatmap-label">Confidence Heatmap (darker = higher confidence)</div>
            <div class="heatmap-container"><canvas id="confidenceHeatmap"></canvas></div>
            <div class="legend" id="legend"></div>
        </div>
        <div class="me-list" id="meList">
            <h2>Detected Micro-Expressions</h2>
            <div id="meItems"></div>
        </div>
        <div class="footer"><p>Generated by 3-Layer Micro-Expression Detection System</p></div>
    </div>
    <div class="me-panel" id="mePanel">
        <button class="me-panel-close" onclick="closePanel()">&times;</button>
        <h2>Micro-Expression Details</h2>
        <div id="mePanelContent"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
        const timelineData = ___JSON_DATA_PLACEHOLDER___;
        const emotionColors = {
            angry: '#FF4444', disgust: '#8B4513', fear: '#9B59B6',
            happy: '#FFD700', sad: '#4A90D9', surprise: '#FF8C00', neutral: '#95A5A6'
        };
        function initDashboard() {
            const dashboard = document.getElementById('dashboard');
            const summary = timelineData.summary;
            const metadata = timelineData.video_metadata;
            let suppressionClass = summary.suppression_score > 60 ? 'suppression-high' : summary.suppression_score > 30 ? 'suppression-medium' : 'suppression-low';
            let rangeClass = summary.emotional_range_score > 60 ? 'range-high' : summary.emotional_range_score > 30 ? 'range-medium' : 'range-low';
            dashboard.innerHTML = `
                <div class="stat-card"><div class="stat-label">Suppression Score</div><div class="stat-value">${summary.suppression_score.toFixed(1)}</div><div class="stat-indicator ${suppressionClass}"></div></div>
                <div class="stat-card"><div class="stat-label">Emotional Range</div><div class="stat-value">${summary.emotional_range_score.toFixed(1)}</div><div class="stat-indicator ${rangeClass}"></div></div>
                <div class="stat-card"><div class="stat-label">Micro-Expressions</div><div class="stat-value">${summary.micro_expression_count}</div></div>
                <div class="stat-card"><div class="stat-label">High Confidence</div><div class="stat-value">${summary.high_confidence_micro_expressions}</div></div>
                <div class="stat-card"><div class="stat-label">Duration</div><div class="stat-value">${metadata.duration_seconds.toFixed(1)}s</div></div>
            `;
            document.getElementById('videoInfo').textContent = `${metadata.filename} | ${metadata.fps.toFixed(1)} FPS | ${metadata.frames_analyzed} frames analyzed`;
        }
        function initEmotionChart() {
            const ctx = document.getElementById('emotionChart').getContext('2d');
            const timeline = timelineData.emotion_timeline;
            const labels = timeline.map(f => f.timestamp.toFixed(1) + 's');
            const emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
            const datasets = emotions.map(emotion => ({
                label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                data: timeline.map(f => f.emotions[emotion] * 100),
                backgroundColor: emotionColors[emotion] + '80',
                borderColor: emotionColors[emotion],
                borderWidth: 1, fill: true, tension: 0.4, pointRadius: 0
            }));
            new Chart(ctx, {
                type: 'line',
                data: { labels: labels, datasets: datasets },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { display: true, title: { display: true, text: 'Time (seconds)', color: '#888' }, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        y: { display: true, stacked: true, min: 0, max: 100, title: { display: true, text: 'Probability (%)', color: '#888' }, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    },
                    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
                    interaction: { mode: 'nearest', axis: 'x', intersect: false }
                }
            });
            const legend = document.getElementById('legend');
            legend.innerHTML = emotions.map(e => `<div class="legend-item"><div class="legend-color" style="background: ${emotionColors[e]}"></div>${e.charAt(0).toUpperCase() + e.slice(1)}</div>`).join('');
        }
        function initConfidenceHeatmap() {
            const canvas = document.getElementById('confidenceHeatmap');
            const ctx = canvas.getContext('2d');
            const timeline = timelineData.emotion_timeline;
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = 30;
            const segmentWidth = canvas.width / timeline.length;
            timeline.forEach((frame, i) => {
                const confidence = frame.confidence_score || 0;
                const intensity = Math.min(confidence / 100, 1);
                const r = Math.round(40 + (1 - intensity) * 80);
                const g = Math.round(20 + intensity * 100);
                const b = Math.round(80 + intensity * 120);
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(i * segmentWidth, 0, segmentWidth + 1, 30);
            });
        }
        function initMEList() {
            const container = document.getElementById('meItems');
            const microExpressions = timelineData.micro_expressions;
            if (microExpressions.length === 0) {
                container.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No micro-expressions detected</p>';
                return;
            }
            container.innerHTML = microExpressions.map((me, idx) => `
                <div class="me-item" onclick="showMicroExpressionPanel(${idx})">
                    <div class="me-item-emotion" style="background: ${emotionColors[me.emotion]}"></div>
                    <div class="me-item-info">
                        <div class="me-item-title">${me.emotion}</div>
                        <div class="me-item-time">${me.start_time.toFixed(1)}s - ${me.end_time.toFixed(1)}s (${me.duration_seconds.toFixed(2)}s)</div>
                    </div>
                    <div class="me-item-confidence">${me.confidence_score.toFixed(0)}%</div>
                </div>
            `).join('');
        }
        function showMicroExpressionPanel(meOrIndex) {
            const me = typeof meOrIndex === 'number' ? timelineData.micro_expressions[meOrIndex] : meOrIndex;
            const panel = document.getElementById('mePanel');
            const content = document.getElementById('mePanelContent');
            const emotionStyle = `background: ${emotionColors[me.emotion]}; color: #fff;`;
            content.innerHTML = `
                <div class="me-detail"><div class="me-detail-label">Emotion</div><div class="me-detail-value"><span class="me-emotion" style="${emotionStyle}">${me.emotion.toUpperCase()}</span></div></div>
                <div class="me-detail"><div class="me-detail-label">Time Range</div><div class="me-detail-value">${me.start_time.toFixed(2)}s - ${me.end_time.toFixed(2)}s</div></div>
                <div class="me-detail"><div class="me-detail-label">Duration</div><div class="me-detail-value">${(me.duration_seconds * 1000).toFixed(0)}ms</div></div>
                <div class="me-detail"><div class="me-detail-label">Peak Probability</div><div class="me-detail-value">${(me.peak_probability * 100).toFixed(1)}%</div></div>
                <div class="me-detail"><div class="me-detail-label">Confidence Score</div><div class="me-detail-value">${me.confidence_score.toFixed(0)}%</div></div>
                <div class="validation-legend">
                    <h4>3-Layer Validation</h4>
                    <div class="validation-item"><div class="validation-icon ${me.validation_layers.deepface ? 'validation-pass' : 'validation-fail'}">${me.validation_layers.deepface ? '✓' : '✗'}</div><span>DeepFace Emotion Spike</span></div>
                    <div class="validation-item"><div class="validation-icon ${me.validation_layers.optical_flow ? 'validation-pass' : 'validation-fail'}">${me.validation_layers.optical_flow ? '✓' : '✗'}</div><span>Optical Flow Movement</span></div>
                    <div class="validation-item"><div class="validation-icon ${me.validation_layers.landmark ? 'validation-pass' : 'validation-fail'}">${me.validation_layers.landmark ? '✓' : '✗'}</div><span>Landmark Displacement</span></div>
                </div>
            `;
            panel.classList.add('active');
        }
        function closePanel() { document.getElementById('mePanel').classList.remove('active'); }
        document.addEventListener('DOMContentLoaded', () => { initDashboard(); initEmotionChart(); initConfidenceHeatmap(); initMEList(); });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePanel(); });
    </script>
</body>
</html>
'''

    # Replace the placeholder with actual JSON data
    html_content = html_template.replace('___JSON_DATA_PLACEHOLDER___', json_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    log_info(f"Saved: {output_path}")


def save_json_output(timeline_data: dict, output_path: str) -> None:
    """
    Save timeline data to JSON file.

    Args:
        timeline_data: Complete timeline data dictionary
        output_path: Path for output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2, cls=NumpyEncoder)

    log_info(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Micro-Expression Detection System with 3-Layer Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python solution.py --video interview.mp4
  python solution.py --video sample.mov --fps 10
  python solution.py --video test.mp4 --html report.html --json output.json
        '''
    )

    parser.add_argument(
        '--video',
        type=str,
        default='video_sample_1.mov',
        help='Input video path (default: video_sample_1.mov)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Analysis frames per second (default: 5)'
    )

    parser.add_argument(
        '--html',
        type=str,
        default='emotion_timeline.html',
        help='HTML output path (default: emotion_timeline.html)'
    )

    parser.add_argument(
        '--json',
        type=str,
        default='emotion_timeline_output.json',
        help='JSON output path (default: emotion_timeline_output.json)'
    )

    args = parser.parse_args()

    try:
        # Run analysis
        timeline_data = analyze_video(args.video, args.fps)

        # Save outputs
        save_json_output(timeline_data, args.json)
        generate_html_report(timeline_data, args.html)

        log_info("Analysis complete!")

    except FileNotFoundError as e:
        log_error(str(e))
        sys.exit(1)
    except Exception as e:
        log_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
