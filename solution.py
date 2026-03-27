"""
Micro-Expression & Emotion Transition Timeline Analyzer

Detects sub-second emotion flashes and tracks emotion transitions over video sessions.
Supports MULTIPLE PERSONS with persistent tracking and per-person analysis.
Generates river chart visualization and JSON output for Sentio Mind integration.
"""

import os
import gc
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from enum import Enum, auto
from deepface import DeepFace
import face_recognition
import warnings
import datetime
import urllib.request

warnings.filterwarnings('ignore')

# ============== CHART.JS BUNDLING FOR OFFLINE USE ==============
CHARTJS_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.chartjs_cache.js')
CHARTJS_CDN_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def get_chartjs_inline() -> str:
    """
    Get Chart.js code for inline embedding.
    Downloads once and caches locally for offline use.
    """
    # Check if cached
    if os.path.exists(CHARTJS_CACHE_PATH):
        with open(CHARTJS_CACHE_PATH, 'r', encoding='utf-8') as f:
            return f.read()

    # Download and cache
    print("Downloading Chart.js for offline bundling (one-time)...")
    try:
        with urllib.request.urlopen(CHARTJS_CDN_URL, timeout=30) as response:
            chartjs_code = response.read().decode('utf-8')
            # Cache for future use
            with open(CHARTJS_CACHE_PATH, 'w', encoding='utf-8') as f:
                f.write(chartjs_code)
            print("Chart.js cached successfully.")
            return chartjs_code
    except Exception as e:
        print(f"Warning: Could not download Chart.js: {e}")
        print("HTML report will use CDN fallback (requires internet).")
        return None

# ============== CONSTANTS ==============
ANALYSIS_FPS = 10  # Process at 10 FPS for efficiency
MICRO_EXPRESSION_MAX_FRAMES = int(ANALYSIS_FPS * 0.5)  # 5 frames = 0.5s
NEUTRAL_THRESHOLD = 0.50  # Min probability to be considered neutral
EMOTION_THRESHOLD = 0.40  # Min probability for non-neutral emotion
DETECTION_INTERVAL = 5  # Full face detection every N analyzed frames
BATCH_SIZE = 32  # Frames per batch for memory management
EMA_ALPHA = 0.3  # Smoothing factor (lower = more smoothing)
FACE_MATCH_THRESHOLD = 0.6  # Face embedding distance threshold for matching
MAX_MISSED_FRAMES = 30  # Remove person after N consecutive missed frames (increased)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NON_NEUTRAL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry': '#e74c3c',
    'disgust': '#9b59b6',
    'fear': '#3498db',
    'happy': '#f1c40f',
    'sad': '#1abc9c',
    'surprise': '#e67e22',
    'neutral': '#95a5a6'
}

PERSON_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']

# DeepFace configuration - balanced for speed and accuracy
DEEPFACE_CONFIG = {
    'detector_backend': 'opencv',  # Fast detection
    'enforce_detection': False,  # Don't fail on missed frames
    'align': True,  # Improves accuracy
}


# ============== DATA CLASSES ==============
@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_index: int
    timestamp: float
    emotions: Dict[str, float]
    dominant_emotion: str
    dominant_probability: float
    person_id: int = 0  # Added for multi-person support


@dataclass
class MicroExpression:
    """Detected micro-expression event."""
    start_time: float
    end_time: float
    emotion: str
    peak_probability: float
    duration_frames: int
    preceding_neutral_prob: float
    following_neutral_prob: float
    person_id: int = 0  # Added for multi-person support


@dataclass
class TrackedFace:
    """A tracked face with persistent ID and face embedding."""
    person_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    embedding: Optional[np.ndarray] = None  # Face embedding for re-identification
    missed_frames: int = 0
    tracker: Optional[cv2.Tracker] = None


class EmotionState(Enum):
    """State machine states for micro-expression detection."""
    NEUTRAL = auto()
    IN_EMOTION = auto()


# ============== MULTI-FACE TRACKER WITH FACE EMBEDDINGS ==============
class MultiFaceTracker:
    """
    Tracks multiple faces with persistent IDs using face embeddings.
    Uses face_recognition library for robust re-identification.
    """

    def __init__(self):
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.known_embeddings: Dict[int, np.ndarray] = {}  # pid -> embedding
        self.next_person_id = 1
        self.frames_since_detection = DETECTION_INTERVAL
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def update(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """Update tracking and return face crops for each person."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.frames_since_detection >= DETECTION_INTERVAL:
            # Full detection with face embeddings
            self._detect_and_match_faces(frame, rgb_frame)
            self.frames_since_detection = 0
        else:
            # Track existing faces
            self._track_faces(frame)
            self.frames_since_detection += 1

        self._cleanup_lost_faces()

        # Return face crops
        face_crops = {}
        for pid, tracked in self.tracked_faces.items():
            crop = self._crop_face(frame, tracked.bbox)
            if crop is not None and crop.size > 0:
                face_crops[pid] = crop

        return face_crops

    def _detect_and_match_faces(self, frame: np.ndarray, rgb_frame: np.ndarray) -> None:
        """Detect faces and match using embeddings."""
        # Detect face locations using face_recognition
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        if not face_locations:
            # No faces - increment missed frames
            for tracked in self.tracked_faces.values():
                tracked.missed_frames += 1
            return

        # Get embeddings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        matched_pids = set()
        matched_detections = set()

        # Match detected faces to known persons using embeddings
        for det_idx, (face_loc, face_enc) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = face_loc
            bbox = (left, top, right - left, bottom - top)

            # Find best matching known person
            best_pid = None
            best_distance = FACE_MATCH_THRESHOLD

            for pid, known_enc in self.known_embeddings.items():
                if pid in matched_pids:
                    continue
                distance = face_recognition.face_distance([known_enc], face_enc)[0]
                if distance < best_distance:
                    best_distance = distance
                    best_pid = pid

            if best_pid is not None:
                # Check if person still exists in tracked_faces (might have been removed)
                if best_pid in self.tracked_faces:
                    # Update existing person
                    self.tracked_faces[best_pid].bbox = bbox
                    self.tracked_faces[best_pid].embedding = face_enc
                    self.tracked_faces[best_pid].missed_frames = 0
                    self.tracked_faces[best_pid].tracker = self._create_tracker(frame, bbox)
                else:
                    # Re-create tracked face for re-identified person
                    self.tracked_faces[best_pid] = TrackedFace(
                        person_id=best_pid,
                        bbox=bbox,
                        embedding=face_enc,
                        tracker=self._create_tracker(frame, bbox)
                    )
                # Update embedding (running average)
                self.known_embeddings[best_pid] = 0.8 * self.known_embeddings[best_pid] + 0.2 * face_enc
                matched_pids.add(best_pid)
                matched_detections.add(det_idx)
            else:
                # New person
                new_pid = self.next_person_id
                self.next_person_id += 1
                self.tracked_faces[new_pid] = TrackedFace(
                    person_id=new_pid,
                    bbox=bbox,
                    embedding=face_enc,
                    tracker=self._create_tracker(frame, bbox)
                )
                self.known_embeddings[new_pid] = face_enc
                matched_pids.add(new_pid)
                matched_detections.add(det_idx)

        # Increment missed frames for unmatched tracked faces
        for pid in self.tracked_faces:
            if pid not in matched_pids:
                self.tracked_faces[pid].missed_frames += 1

    def _track_faces(self, frame: np.ndarray) -> None:
        """Track faces using OpenCV trackers."""
        for tracked in self.tracked_faces.values():
            if tracked.tracker is not None:
                success, box = tracked.tracker.update(frame)
                if success:
                    tracked.bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    tracked.missed_frames = 0
                else:
                    tracked.missed_frames += 1
            else:
                tracked.missed_frames += 1

    def _cleanup_lost_faces(self) -> None:
        """Remove faces that have been missing too long."""
        to_remove = [
            pid for pid, tracked in self.tracked_faces.items()
            if tracked.missed_frames > MAX_MISSED_FRAMES
        ]
        for pid in to_remove:
            del self.tracked_faces[pid]
            # Keep embedding for potential re-identification later
            # del self.known_embeddings[pid]

    def _create_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[cv2.Tracker]:
        """Create and initialize a tracker."""
        try:
            tracker = cv2.TrackerCSRT.create()
        except AttributeError:
            try:
                tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                return None
        tracker.init(frame, bbox)
        return tracker

    def _crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop face region from frame."""
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        x = max(0, x)
        y = max(0, y)
        return frame[y:y2, x:x2]

    def get_active_person_ids(self) -> List[int]:
        """Return list of currently tracked person IDs."""
        return list(self.tracked_faces.keys())


# ============== EMOTION SMOOTHER ==============
class EmotionSmoother:
    """EMA-based emotion smoothing to reduce noise while preserving rapid changes."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.ema_state: Dict[str, float] = {}
        self.initialized = False

    def smooth(self, raw_emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential moving average smoothing."""
        if not self.initialized:
            self.ema_state = {e: raw_emotions.get(e, 0.0) for e in EMOTIONS}
            self.initialized = True
            return raw_emotions

        smoothed = {}
        for emotion in EMOTIONS:
            raw = raw_emotions.get(emotion, 0.0)
            prev = self.ema_state.get(emotion, 0.0)
            smoothed[emotion] = self.alpha * raw + (1 - self.alpha) * prev
            self.ema_state[emotion] = smoothed[emotion]

        return smoothed


# ============== MICRO-EXPRESSION DETECTOR ==============
class MicroExpressionDetector:
    """State machine for detecting neutral -> emotion -> neutral transitions."""

    def __init__(self, person_id: int = 0):
        self.person_id = person_id
        self.state = EmotionState.NEUTRAL
        self.emotion_buffer: List[FrameAnalysis] = []
        self.last_neutral_frame: Optional[FrameAnalysis] = None
        self.detected: List[MicroExpression] = []

    def process_frame(self, frame: FrameAnalysis) -> Optional[MicroExpression]:
        """Process frame and detect micro-expressions."""
        neutral_prob = frame.emotions.get('neutral', 0)
        is_neutral = neutral_prob >= NEUTRAL_THRESHOLD

        non_neutral_probs = {e: frame.emotions.get(e, 0) for e in NON_NEUTRAL_EMOTIONS}
        strongest_emotion = max(non_neutral_probs, key=non_neutral_probs.get)
        strongest_prob = non_neutral_probs[strongest_emotion]
        has_emotion = strongest_prob >= EMOTION_THRESHOLD

        result = None

        if self.state == EmotionState.NEUTRAL:
            if is_neutral:
                self.last_neutral_frame = frame
            elif has_emotion and self.last_neutral_frame is not None:
                self.state = EmotionState.IN_EMOTION
                self.emotion_buffer = [frame]

        elif self.state == EmotionState.IN_EMOTION:
            if is_neutral:
                if (len(self.emotion_buffer) < MICRO_EXPRESSION_MAX_FRAMES and
                    len(self.emotion_buffer) > 0 and
                    self.last_neutral_frame is not None):

                    peak_frame = max(
                        self.emotion_buffer,
                        key=lambda f: max(f.emotions.get(e, 0) for e in NON_NEUTRAL_EMOTIONS)
                    )
                    peak_emotion = max(
                        NON_NEUTRAL_EMOTIONS,
                        key=lambda e: peak_frame.emotions.get(e, 0)
                    )
                    peak_prob = peak_frame.emotions.get(peak_emotion, 0)

                    if peak_prob >= EMOTION_THRESHOLD:
                        result = MicroExpression(
                            start_time=self.emotion_buffer[0].timestamp,
                            end_time=self.emotion_buffer[-1].timestamp,
                            emotion=peak_emotion,
                            peak_probability=peak_prob,
                            duration_frames=len(self.emotion_buffer),
                            preceding_neutral_prob=self.last_neutral_frame.emotions.get('neutral', 0),
                            following_neutral_prob=neutral_prob,
                            person_id=self.person_id
                        )
                        self.detected.append(result)

                self.state = EmotionState.NEUTRAL
                self.last_neutral_frame = frame
                self.emotion_buffer = []

            elif has_emotion:
                self.emotion_buffer.append(frame)
                if len(self.emotion_buffer) >= MICRO_EXPRESSION_MAX_FRAMES:
                    self.state = EmotionState.NEUTRAL
                    self.emotion_buffer = []

        return result

    def get_all_detected(self) -> List[MicroExpression]:
        return self.detected


# ============== PERSON STATE ==============
@dataclass
class PersonState:
    """Per-person analysis state."""
    person_id: int
    emotion_smoother: EmotionSmoother = field(default_factory=EmotionSmoother)
    micro_detector: MicroExpressionDetector = field(default=None)
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def __post_init__(self):
        if self.micro_detector is None:
            self.micro_detector = MicroExpressionDetector(self.person_id)


# ============== SCORE CALCULATORS ==============
def calculate_suppression_score(
    micro_expressions: List[MicroExpression],
    total_duration_seconds: float
) -> float:
    """Calculate Suppression Score (0-100)."""
    if not micro_expressions or total_duration_seconds <= 0:
        return 0.0

    freq_per_minute = (len(micro_expressions) / total_duration_seconds) * 60
    freq_component = min(freq_per_minute / 20.0, 1.0) * 40

    avg_following_neutral = np.mean([me.following_neutral_prob for me in micro_expressions])
    strength_component = avg_following_neutral * 30

    avg_duration_ratio = np.mean([
        1.0 - (me.duration_frames / MICRO_EXPRESSION_MAX_FRAMES)
        for me in micro_expressions
    ])
    duration_component = max(0, avg_duration_ratio) * 30

    score = freq_component + strength_component + duration_component
    return min(100.0, max(0.0, round(score, 2)))


def calculate_emotional_range_score(frame_analyses: List[FrameAnalysis]) -> float:
    """Calculate Emotional Range Score (0-100)."""
    if not frame_analyses:
        return 0.0

    emotion_appearances = {e: 0 for e in EMOTIONS}
    for frame in frame_analyses:
        for emotion in EMOTIONS:
            if frame.emotions.get(emotion, 0) >= 0.30:
                emotion_appearances[emotion] += 1

    min_appearances = max(1, len(frame_analyses) * 0.05)
    emotions_present = sum(1 for count in emotion_appearances.values() if count >= min_appearances)
    diversity_score = (emotions_present / 7.0) * 40

    dominant_probs = [max(frame.emotions.values()) for frame in frame_analyses]
    intensity_std = np.std(dominant_probs) if len(dominant_probs) > 1 else 0
    intensity_score = min(intensity_std / 0.2, 1.0) * 30

    transitions = sum(
        1 for i in range(1, len(frame_analyses))
        if frame_analyses[i].dominant_emotion != frame_analyses[i-1].dominant_emotion
    )

    total_seconds = frame_analyses[-1].timestamp - frame_analyses[0].timestamp if len(frame_analyses) > 1 else 1
    if total_seconds > 0:
        transitions_per_second = transitions / total_seconds
        transition_score = min(transitions_per_second / 1.0, 1.0) * 30
    else:
        transition_score = 0

    score = diversity_score + intensity_score + transition_score
    return min(100.0, max(0.0, round(score, 2)))


# ============== MAIN ANALYZER (MULTI-PERSON) ==============
class EmotionTimelineAnalyzer:
    """Main orchestrator for multi-person emotion timeline analysis."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.person_states: Dict[int, PersonState] = {}
        self.video_fps: float = 30.0
        self.video_duration: float = 0.0
        self.total_frames: int = 0

    def analyze(self) -> None:
        """Run full analysis pipeline for multiple persons."""
        print(f"Analyzing video: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0

        print(f"Video FPS: {self.video_fps:.2f}, Duration: {self.video_duration:.2f}s, Frames: {self.total_frames}")

        sample_interval = max(1, int(self.video_fps / ANALYSIS_FPS))
        print(f"Sampling every {sample_interval} frames (target {ANALYSIS_FPS} FPS)")

        face_tracker = MultiFaceTracker()
        frame_idx = 0
        batch_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / self.video_fps

                # Get all face crops
                face_crops = face_tracker.update(frame)

                for person_id, face_crop in face_crops.items():
                    # Initialize person state if new
                    if person_id not in self.person_states:
                        self.person_states[person_id] = PersonState(
                            person_id=person_id,
                            first_seen=timestamp
                        )
                        print(f"  New person detected: Person {person_id} at {timestamp:.1f}s")

                    person = self.person_states[person_id]
                    person.last_seen = timestamp

                    # Analyze emotions
                    raw_emotions = self._analyze_emotion(face_crop)

                    if raw_emotions:
                        smoothed = person.emotion_smoother.smooth(raw_emotions)
                        dominant = max(smoothed, key=smoothed.get)

                        frame_analysis = FrameAnalysis(
                            frame_index=frame_idx,
                            timestamp=timestamp,
                            emotions=smoothed,
                            dominant_emotion=dominant,
                            dominant_probability=smoothed[dominant],
                            person_id=person_id
                        )
                        person.frame_analyses.append(frame_analysis)
                        person.micro_detector.process_frame(frame_analysis)

                batch_count += 1
                if batch_count >= BATCH_SIZE:
                    gc.collect()
                    batch_count = 0
                    active_persons = len(face_tracker.get_active_person_ids())
                    print(f"Processed to {timestamp:.1f}s / {self.video_duration:.1f}s ({active_persons} active persons)")

            frame_idx += 1

        cap.release()

        # Print summary
        print(f"\nAnalysis complete:")
        print(f"  Total persons detected: {len(self.person_states)}")
        for pid, person in self.person_states.items():
            micros = person.micro_detector.get_all_detected()
            print(f"  Person {pid}: {len(person.frame_analyses)} frames, {len(micros)} micro-expressions")

    def _analyze_emotion(self, face_crop: np.ndarray) -> Optional[Dict[str, float]]:
        """Analyze emotion probabilities for a face crop."""
        try:
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                **DEEPFACE_CONFIG
            )
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            return {k.lower(): v / 100.0 for k, v in emotions.items()}
        except Exception:
            return None

    def calculate_person_scores(self, person_id: int) -> Dict[str, float]:
        """Calculate scores for a specific person."""
        if person_id not in self.person_states:
            return {'suppression_score': 0.0, 'emotional_range_score': 0.0}

        person = self.person_states[person_id]
        duration = person.last_seen - person.first_seen if person.last_seen > person.first_seen else self.video_duration
        micros = person.micro_detector.get_all_detected()

        return {
            'suppression_score': calculate_suppression_score(micros, duration),
            'emotional_range_score': calculate_emotional_range_score(person.frame_analyses)
        }

    def generate_html_report(self, output_path: str) -> None:
        """Generate HTML visualization with per-person tabs (works offline)."""
        if not self.person_states:
            print("No persons detected, creating empty report.")

        # Get Chart.js for inline embedding (offline support)
        chartjs_inline = get_chartjs_inline()
        if chartjs_inline:
            chartjs_tag = f"<script>{chartjs_inline}</script>"
        else:
            # Fallback to CDN if download failed
            chartjs_tag = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'

        # Build per-person data
        persons_data = []
        for pid in sorted(self.person_states.keys()):
            person = self.person_states[pid]
            scores = self.calculate_person_scores(pid)
            micros = person.micro_detector.get_all_detected()

            chart_data = [
                {
                    'timestamp': round(f.timestamp, 2),
                    **{e: round(f.emotions.get(e, 0) * 100, 2) for e in EMOTIONS}
                }
                for f in person.frame_analyses
            ]

            micro_data = [
                {
                    'start_time': round(me.start_time, 3),
                    'end_time': round(me.end_time, 3),
                    'emotion': me.emotion,
                    'peak_probability': round(me.peak_probability * 100, 1)
                }
                for me in micros
            ]

            persons_data.append({
                'person_id': pid,
                'scores': scores,
                'chart_data': chart_data,
                'micro_expressions': micro_data,
                'frame_count': len(person.frame_analyses),
                'first_seen': round(person.first_seen, 2),
                'last_seen': round(person.last_seen, 2)
            })

        # Generate person tabs HTML
        tabs_html = ""
        content_html = ""

        for i, pdata in enumerate(persons_data):
            pid = pdata['person_id']
            active = "active" if i == 0 else ""
            color = PERSON_COLORS[i % len(PERSON_COLORS)]

            tabs_html += f'''
                <button class="tab-btn {active}" onclick="showPerson({pid})" data-person="{pid}" style="border-bottom-color: {color}">
                    Person {pid}
                </button>
            '''

            micro_list = ""
            for j, me in enumerate(pdata['micro_expressions']):
                me_color = EMOTION_COLORS.get(me['emotion'], '#666')
                micro_list += f'''
                    <div class="micro-expression-item" style="border-color: {me_color}">
                        <strong>#{j+1}</strong>&nbsp;&nbsp;
                        <span style="color: {me_color}">{me['emotion'].upper()}</span>&nbsp;&nbsp;
                        at {me['start_time']:.2f}s - {me['end_time']:.2f}s&nbsp;&nbsp;
                        (peak: {me['peak_probability']:.1f}%)
                    </div>
                '''
            if not micro_list:
                micro_list = '<p>No micro-expressions detected.</p>'

            display = "block" if i == 0 else "none"
            content_html += f'''
                <div class="person-content" id="person-{pid}" style="display: {display}">
                    <div class="stats-grid">
                        <div class="stat-card" style="background: linear-gradient(135deg, {color} 0%, {color}99 100%)">
                            <div class="stat-value">{pdata['scores']['suppression_score']:.0f}</div>
                            <div class="stat-label">Suppression Score</div>
                        </div>
                        <div class="stat-card secondary">
                            <div class="stat-value">{pdata['scores']['emotional_range_score']:.0f}</div>
                            <div class="stat-label">Emotional Range</div>
                        </div>
                        <div class="stat-card tertiary">
                            <div class="stat-value">{len(pdata['micro_expressions'])}</div>
                            <div class="stat-label">Micro-Expressions</div>
                        </div>
                        <div class="stat-card quaternary">
                            <div class="stat-value">{pdata['frame_count']}</div>
                            <div class="stat-label">Frames Analyzed</div>
                        </div>
                    </div>
                    <p class="person-info">Visible from {pdata['first_seen']:.1f}s to {pdata['last_seen']:.1f}s</p>
                    <div class="chart-container">
                        <canvas id="chart-{pid}"></canvas>
                    </div>
                    <h3 class="section-title">Micro-Expressions</h3>
                    <div class="micro-list">{micro_list}</div>
                </div>
            '''

        # Create charts initialization JS with micro-expression markers
        charts_js = "const chartsData = " + json.dumps({p['person_id']: p['chart_data'] for p in persons_data}) + ";\n"
        charts_js += "const microExpressionsData = " + json.dumps({p['person_id']: p['micro_expressions'] for p in persons_data}) + ";\n"
        charts_js += """
        const emotionColors = {
            angry: '#e74c3c', disgust: '#9b59b6', fear: '#3498db',
            happy: '#f1c40f', sad: '#1abc9c', surprise: '#e67e22', neutral: '#95a5a6'
        };
        const emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'];
        const charts = {};

        // Custom plugin to draw vertical dashed lines for micro-expressions
        const microExpressionPlugin = {
            id: 'microExpressionLines',
            afterDraw: (chart) => {
                const personId = parseInt(chart.canvas.id.replace('chart-', ''));
                const microExpressions = microExpressionsData[personId] || [];
                if (microExpressions.length === 0) return;

                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                const chartData = chartsData[personId] || [];

                microExpressions.forEach(me => {
                    // Find the closest data point index for this timestamp
                    let closestIdx = 0;
                    let minDiff = Infinity;
                    chartData.forEach((d, idx) => {
                        const diff = Math.abs(d.timestamp - me.start_time);
                        if (diff < minDiff) {
                            minDiff = diff;
                            closestIdx = idx;
                        }
                    });

                    const xPos = xScale.getPixelForValue(closestIdx);
                    const color = emotionColors[me.emotion] || '#666';

                    // Draw vertical dashed line
                    ctx.save();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath();
                    ctx.moveTo(xPos, yScale.top);
                    ctx.lineTo(xPos, yScale.bottom);
                    ctx.stroke();

                    // Draw marker dot at top
                    ctx.fillStyle = color;
                    ctx.setLineDash([]);
                    ctx.beginPath();
                    ctx.arc(xPos, yScale.top + 8, 5, 0, Math.PI * 2);
                    ctx.fill();

                    // Draw tooltip label
                    ctx.fillStyle = '#333';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(me.emotion.toUpperCase(), xPos, yScale.top + 22);
                    ctx.restore();
                });
            }
        };

        function createChart(personId) {
            const data = chartsData[personId];
            if (!data || data.length === 0) return;

            const ctx = document.getElementById('chart-' + personId);
            if (!ctx) return;

            charts[personId] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.timestamp.toFixed(1) + 's'),
                    datasets: emotions.map(emotion => ({
                        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                        data: data.map(d => d[emotion]),
                        backgroundColor: emotionColors[emotion] + '60',
                        borderColor: emotionColors[emotion],
                        borderWidth: 1, fill: true, tension: 0.4, pointRadius: 0
                    }))
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: { legend: { display: false },
                        tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }}},
                    scales: {
                        x: { title: { display: true, text: 'Time (s)' }, ticks: { maxTicksLimit: 20 }},
                        y: { stacked: true, title: { display: true, text: 'Probability (%)' }, min: 0, max: 100 }
                    }
                },
                plugins: [microExpressionPlugin]
            });
        }

        function showPerson(personId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelector('[data-person="' + personId + '"]').classList.add('active');
            document.querySelectorAll('.person-content').forEach(c => c.style.display = 'none');
            document.getElementById('person-' + personId).style.display = 'block';
            if (!charts[personId]) createChart(personId);
        }

        // Initialize first person's chart
        document.addEventListener('DOMContentLoaded', () => {
            const firstPerson = Object.keys(chartsData)[0];
            if (firstPerson) createChart(parseInt(firstPerson));
        });
        """

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Person Emotion Timeline Analysis</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ margin-bottom: 5px; color: #2c3e50; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 20px; }}
        .summary {{ background: #f8f9fa; padding: 15px 20px; border-radius: 8px; margin-bottom: 25px; }}
        .tabs {{ display: flex; gap: 5px; border-bottom: 2px solid #eee; margin-bottom: 20px; }}
        .tab-btn {{ padding: 12px 24px; border: none; background: #f0f0f0; cursor: pointer; border-radius: 8px 8px 0 0; font-weight: 500; border-bottom: 3px solid transparent; }}
        .tab-btn.active {{ background: white; border-bottom-color: #3498db; }}
        .tab-btn:hover {{ background: #e8e8e8; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ padding: 20px; border-radius: 10px; text-align: center; color: white; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .stat-card.secondary {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card.tertiary {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-card.quaternary {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ margin-top: 5px; opacity: 0.9; font-size: 0.9em; }}
        .person-info {{ color: #7f8c8d; margin-bottom: 15px; font-size: 0.9em; }}
        .chart-container {{ position: relative; height: 350px; background: #fafafa; border-radius: 10px; padding: 15px; margin-bottom: 20px; }}
        .section-title {{ font-size: 1.1em; margin: 20px 0 10px; color: #2c3e50; }}
        .micro-expression-item {{ display: flex; align-items: center; padding: 10px 12px; border-left: 4px solid; margin-bottom: 8px; background: #f8f9fa; border-radius: 0 6px 6px 0; font-size: 0.9em; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.85em; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 3px; }}
    </style>
    {chartjs_tag}
</head>
<body>
    <div class="container">
        <h1>Multi-Person Emotion Timeline</h1>
        <p class="subtitle">Micro-Expression Detection & Emotional Pattern Analysis</p>

        <div class="summary">
            <strong>{len(self.person_states)}</strong> person(s) detected &nbsp;|&nbsp;
            <strong>{self.video_duration:.1f}s</strong> video duration &nbsp;|&nbsp;
            <strong>{sum(len(p.micro_detector.get_all_detected()) for p in self.person_states.values())}</strong> total micro-expressions
        </div>

        <div class="tabs">{tabs_html}</div>

        {content_html}

        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #e74c3c"></div>Angry</div>
            <div class="legend-item"><div class="legend-color" style="background: #9b59b6"></div>Disgust</div>
            <div class="legend-item"><div class="legend-color" style="background: #3498db"></div>Fear</div>
            <div class="legend-item"><div class="legend-color" style="background: #f1c40f"></div>Happy</div>
            <div class="legend-item"><div class="legend-color" style="background: #1abc9c"></div>Sad</div>
            <div class="legend-item"><div class="legend-color" style="background: #e67e22"></div>Surprise</div>
            <div class="legend-item"><div class="legend-color" style="background: #95a5a6"></div>Neutral</div>
        </div>
    </div>
    <script>{charts_js}</script>
</body>
</html>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report saved: {output_path}")

    def generate_json_output(self, output_path: str) -> None:
        """Generate JSON output with person_profiles structure for Sentio Mind."""
        person_profiles = {}

        for pid in sorted(self.person_states.keys()):
            person = self.person_states[pid]
            scores = self.calculate_person_scores(pid)
            micros = person.micro_detector.get_all_detected()

            # Calculate per-person emotion distribution
            emotion_totals = {e: 0.0 for e in EMOTIONS}
            for frame in person.frame_analyses:
                for emotion, prob in frame.emotions.items():
                    if emotion in emotion_totals:
                        emotion_totals[emotion] += prob

            total_prob = sum(emotion_totals.values())
            emotion_distribution = {
                e: round((v / total_prob * 100) if total_prob > 0 else 0, 2)
                for e, v in emotion_totals.items()
            }

            # Most suppressed emotion
            micro_emotion_counts: Dict[str, int] = {}
            for me in micros:
                micro_emotion_counts[me.emotion] = micro_emotion_counts.get(me.emotion, 0) + 1
            most_suppressed = max(micro_emotion_counts, key=micro_emotion_counts.get) if micro_emotion_counts else 'none'

            # Micro-expression timestamps
            micro_times = set()
            for me in micros:
                for frame in person.frame_analyses:
                    if me.start_time <= frame.timestamp <= me.end_time:
                        micro_times.add(frame.timestamp)

            duration = person.last_seen - person.first_seen if person.last_seen > person.first_seen else self.video_duration

            person_profiles[str(pid)] = {
                "person_id": pid,
                "first_seen_seconds": round(person.first_seen, 2),
                "last_seen_seconds": round(person.last_seen, 2),
                "duration_visible_seconds": round(duration, 2),
                "emotion_timeline": {
                    "scores": {
                        "suppression_score": scores['suppression_score'],
                        "emotional_range_score": scores['emotional_range_score']
                    },
                    "micro_expressions": [
                        {
                            "id": idx + 1,
                            "start_time": round(me.start_time, 3),
                            "end_time": round(me.end_time, 3),
                            "duration_ms": int((me.end_time - me.start_time) * 1000),
                            "emotion": me.emotion,
                            "peak_probability": round(me.peak_probability, 3),
                            "preceding_neutral_prob": round(me.preceding_neutral_prob, 3),
                            "following_neutral_prob": round(me.following_neutral_prob, 3)
                        }
                        for idx, me in enumerate(micros)
                    ],
                    "timeline": [
                        {
                            "timestamp": round(frame.timestamp, 3),
                            "emotions": {k: round(v, 3) for k, v in frame.emotions.items()},
                            "dominant_emotion": frame.dominant_emotion,
                            "is_micro_expression": frame.timestamp in micro_times
                        }
                        for frame in person.frame_analyses
                    ],
                    "summary_statistics": {
                        "dominant_emotion_overall": max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else 'neutral',
                        "emotion_distribution": emotion_distribution,
                        "micro_expression_count": len(micros),
                        "micro_expression_rate_per_minute": round(len(micros) / (duration / 60) if duration > 0 else 0, 2),
                        "average_micro_expression_duration_ms": round(
                            np.mean([(me.end_time - me.start_time) * 1000 for me in micros]) if micros else 0, 2
                        ),
                        "most_suppressed_emotion": most_suppressed
                    }
                }
            }

        output = {
            "schema_version": "1.0",
            "analysis_metadata": {
                "video_file": os.path.basename(self.video_path),
                "analysis_timestamp": datetime.datetime.now().isoformat(),
                "video_duration_seconds": round(self.video_duration, 2),
                "total_persons_detected": len(self.person_states),
                "analysis_fps": ANALYSIS_FPS
            },
            "person_profiles": person_profiles
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"JSON output saved: {output_path}")


# ============== MAIN ENTRY POINT ==============
def main():
    """Main entry point."""
    import sys

    # Use command line argument or default video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "Class_8_cctv_video_1.mov"  # Default video

    if not os.path.exists(video_path):
        print(f"Warning: Video file '{video_path}' not found.")
        print("Please provide video_sample_1.mov in the current directory.")
        print("Creating demo output...")

        analyzer = EmotionTimelineAnalyzer(video_path)
        analyzer.video_duration = 10.0
    else:
        analyzer = EmotionTimelineAnalyzer(video_path)
        analyzer.analyze()

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Total persons detected: {len(analyzer.person_states)}")

    for pid in sorted(analyzer.person_states.keys()):
        scores = analyzer.calculate_person_scores(pid)
        person = analyzer.person_states[pid]
        micros = person.micro_detector.get_all_detected()
        print(f"\nPerson {pid}:")
        print(f"  Suppression Score:     {scores['suppression_score']:.2f}")
        print(f"  Emotional Range Score: {scores['emotional_range_score']:.2f}")
        print(f"  Micro-expressions:     {len(micros)}")

    print(f"{'='*50}\n")

    analyzer.generate_html_report("emotion_timeline.html")
    analyzer.generate_json_output("emotion_timeline_output.json")

    print("\nDone! Output files generated:")
    print("  - emotion_timeline.html (with per-person tabs)")
    print("  - emotion_timeline_output.json (with person_profiles)")


if __name__ == "__main__":
    main()
