# Micro-Expression Detection System

A Python-based micro-expression detection system using **3-layer validation** for high-confidence emotion analysis in video files.

## Overview

This system analyzes video files to detect micro-expressions - brief, involuntary facial expressions that reveal true emotions. It uses a unique 3-layer validation approach combining:

1. **DeepFace** - Deep learning-based emotion recognition
2. **Optical Flow** - Facial muscle movement detection
3. **MediaPipe Landmarks** - Precise facial landmark displacement tracking

## Features

- Real-time emotion analysis across 7 emotion categories
- 3-layer validation for high-confidence micro-expression detection
- Interactive HTML report with visualization
- JSON output for further analysis
- Configurable analysis parameters
- Automatic MediaPipe model downloading

## Requirements

### Python Version
- Python 3.8+

### Dependencies
```bash
pip install opencv-python numpy deepface mediapipe
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >=4.5 | Video processing, optical flow |
| numpy | >=1.19 | Numerical computations |
| deepface | >=0.0.75 | Emotion recognition (Layer 1) |
| mediapipe | >=0.10 | Facial landmarks (Layer 3) |

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install opencv-python numpy deepface mediapipe
   ```
3. Run the analysis:
   ```bash
   python solution.py --video your_video.mp4
   ```

## Usage

### Basic Usage
```bash
python solution.py --video video_sample.mp4
```

### With Custom Parameters
```bash
python solution.py --video video_sample.mp4 --fps 10 --html report.html --json output.json
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | video_sample_1.mov | Input video file path |
| `--fps` | 5 | Analysis frames per second |
| `--html` | emotion_timeline.html | Output HTML report path |
| `--json` | emotion_timeline_output.json | Output JSON data path |

## Output Files

### 1. HTML Report (`emotion_timeline.html`)
Interactive visualization including:
- **Dashboard** - Suppression score, emotional range, micro-expression count
- **Emotion Timeline** - Stacked area chart showing emotion probabilities over time
- **Confidence Heatmap** - Visual representation of detection confidence
- **Micro-Expression List** - Clickable list with detailed information
- **3-Layer Validation Panel** - Shows which validation layers passed for each detection

### 2. JSON Output (`emotion_timeline_output.json`)
Complete analysis data including:
- Video metadata
- Frame-by-frame emotion probabilities
- Detected micro-expressions with timestamps
- Summary statistics

## 3-Layer Validation System

### Layer 1: DeepFace Emotion Analysis (40%)
- Uses deep learning to detect emotions
- Analyzes emotion probability spikes
- Detects transitions between emotional states

### Layer 2: Optical Flow (35%)
- Computes Farneback optical flow
- Measures facial muscle movement magnitude
- Validates physical facial movement

### Layer 3: MediaPipe Landmarks (25%)
- Tracks 478 facial landmarks
- Measures displacement of key facial features
- Focuses on eyebrows, lips, and eye regions

### Confidence Scoring
- **100%**: All 3 layers validate the micro-expression
- **85%**: 2 layers with strong signals
- **65-80%**: Partial validation with moderate signals
- **<60%**: Below threshold, not reported

## Detection Criteria

A micro-expression is detected when:
1. Rapid emotion change occurs (spike from baseline)
2. Duration is brief (< 1 second)
3. Emotion intensity exceeds threshold (30%)
4. Confidence score meets minimum threshold (60-75%)

## Emotion Categories

| Emotion | Color Code |
|---------|------------|
| Angry | #FF4444 |
| Disgust | #8B4513 |
| Fear | #9B59B6 |
| Happy | #FFD700 |
| Sad | #4A90D9 |
| Surprise | #FF8C00 |
| Neutral | #95A5A6 |

## Metrics

### Suppression Score
- Indicates emotional suppression/stress level
- Formula: `(micro_expression_count / total_emotion_events) * 100`
- Higher scores indicate more suppressed emotions

### Emotional Range Score
- Indicates variety of emotions expressed
- Formula: `(unique_emotions / 7) * 100`
- Higher scores indicate broader emotional expression

## Project Structure

```
Vani assignment/
├── solution.py                    # Main analysis script
├── README.md                      # This file
├── video_sample.mp4              # Sample video file
├── emotion_timeline.html         # Generated HTML report
├── emotion_timeline_output.json  # Generated JSON output
├── face_landmarker.task          # MediaPipe model (auto-downloaded)
└── photos/                       # Optional: extracted frames
```

## Example Output

```
[INFO] Loading video: video_sample.mp4
[INFO] Video: 58.5 fps, 7169 frames, analyzing every 11th frame
[INFO] MediaPipe Face Landmarker initialized (tasks API)
[INFO] Analyzing frame 1/651 (0%)...
...
[INFO] Frame 2541: MICRO-EXPRESSION detected - happy (confidence: 100%)
[INFO] Frame 5489: MICRO-EXPRESSION detected - happy (confidence: 100%)
...
[INFO] Analysis complete. 30 micro-expressions found.
[INFO] Suppression Score: 5.9 | Emotional Range Score: 85.7
[INFO] Saved: emotion_timeline_output.json
[INFO] Saved: emotion_timeline.html
[INFO] Analysis complete!
```

## Troubleshooting

### SSL Certificate Error
If MediaPipe model download fails due to SSL:
- The code automatically handles SSL certificate issues
- Model is downloaded once and cached locally

### No Micro-Expressions Detected
- Try lowering the `--fps` parameter for more detailed analysis
- Check if faces are clearly visible in the video
- Ensure adequate lighting in the video

### Memory Issues
- Reduce `--fps` for large videos
- Process shorter video segments

## Technical Details

### Thresholds (Configurable in code)
```python
DEEPFACE_SPIKE_THRESHOLD = 0.20
OPTICAL_FLOW_THRESHOLD = 0.3
LANDMARK_DISPLACEMENT_THRESHOLD = 0.015
CONFIDENCE_THRESHOLD = 60
```

### Supported Video Formats
- MP4, MOV, AVI, MKV
- Any format supported by OpenCV

## License

This project is for educational and research purposes.

## Author

Micro-Expression Detection System with 3-Layer Validation

---

**Note**: First run may take longer as the MediaPipe model (~4MB) is downloaded automatically.
