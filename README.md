# Sentio Mind - Advanced Micro-Expression Detection System

## Overview
This project implements an advanced **3-Layer Validation System** for detecting micro-expressions in video content. Unlike basic emotion detection that only uses a single source, this solution validates micro-expressions through multiple independent methods to reduce false positives and provide high-confidence results.

## Key Features

### 3-Layer Validation Architecture
| Layer | Technology | Purpose |
|-------|------------|---------|
| Layer 1 | DeepFace + Sliding Window | Detect emotion changes with temporal smoothing |
| Layer 2 | Optical Flow (OpenCV) | Validate actual facial movement occurred |
| Layer 3 | Landmark Displacement | Confirm facial muscle movement via key points |

### Confidence Voting System
- Layer 1 (DeepFace): 40 points
- Layer 2 (Optical Flow): 35 points
- Layer 3 (Landmarks): 25 points
- **Threshold: 40+ points = Confirmed Micro-Expression**

### Output Metrics
- **Suppression Score**: Measures how much a person suppresses emotions
  - Formula: `(micro_expression_count / total_emotion_events) * 100`
- **Emotional Range Score**: Measures variety of emotions displayed
  - Formula: `(unique_emotions / 7) * 100`

## Installation

```bash
# Install required dependencies
pip install deepface opencv-python numpy mediapipe tf-keras
```

## Usage

### Analyze Video
```bash
python solution_advanced.py --video video_sample.mp4
```

### Analyze Photos
```bash
python solution_advanced.py --photos photos/
```

### Custom FPS
```bash
python solution_advanced.py --video video_sample.mp4 --fps 10
```

## Output Files

| File | Description |
|------|-------------|
| `emotion_timeline_output.json` | Complete analysis data with validation layers |
| `emotion_timeline.html` | Interactive HTML report with charts |
| `photos_analysis.json` | Photo emotion analysis results |
| `photos_analysis.html` | Photo analysis HTML report |

## JSON Schema

```json
{
  "video_metadata": {
    "filename": "video_sample.mp4",
    "duration_seconds": 122.5,
    "total_frames": 7169,
    "fps": 58.51,
    "frames_analyzed": 652
  },
  "emotion_timeline": [
    {
      "frame_idx": 517,
      "timestamp": 8.835,
      "emotions": { "happy": 0.47, "sad": 0.12, ... },
      "dominant_emotion": "happy",
      "is_micro_expression": true,
      "confidence_score": 100.0,
      "optical_flow_magnitude": 1.234,
      "landmark_displacement": 0.056,
      "validation_layers": {
        "deepface": true,
        "optical_flow": true,
        "landmark": true
      }
    }
  ],
  "micro_expressions": [...],
  "summary": {
    "suppression_score": 6.52,
    "emotional_range_score": 85.71,
    "micro_expression_count": 33,
    "high_confidence_micro_expressions": 21
  }
}
```

## Results

### Video Analysis Results
- **Frames Analyzed**: 652
- **Micro-expressions Detected**: 33
- **High Confidence**: 21
- **Suppression Score**: 6.52
- **Emotional Range Score**: 85.71

### Detected Emotions
- Happy: Multiple occurrences
- Sad: Multiple occurrences
- Fear: 2 occurrences
- Angry, Disgust, Surprise, Neutral: Tracked throughout

## Technical Details

### What is a Micro-Expression?
A micro-expression is a brief, involuntary facial expression that occurs when a person tries to suppress or conceal their true emotions. They typically last **less than 500ms** (0.5 seconds).

### Why 3-Layer Validation?
1. **Reduces False Positives**: Single-source detection often flags noise as emotions
2. **Multi-Modal Confirmation**: Movement + Emotion + Landmarks must all agree
3. **Confidence Scoring**: Quantifiable trust level for each detection

### Optical Flow Validation
Uses Farneback optical flow algorithm to detect actual pixel movement in facial regions. This confirms that a detected emotion change corresponds to real facial movement.

### Landmark Displacement
Tracks 29 key facial points (eyebrows, lips, eyes) to measure how much facial features moved between frames.

## Dependencies
- Python 3.9+
- DeepFace
- OpenCV (cv2)
- NumPy
- MediaPipe
- TensorFlow/Keras

## Author
Gaurav Mishra

## License
This project is for assessment purposes.
