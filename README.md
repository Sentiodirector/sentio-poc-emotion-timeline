# Micro-Expression & Emotion Transition Timeline

Detect sub-second emotion flashes and track emotion transitions over video sessions.

## System Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Video File<br/>video_sample.mov] --> B[Frame Extraction]
        C[Profile Images<br/>Profiles_1/] --> D[Face Database]
    end

    subgraph Processing
        B --> E[Face Detection<br/>OpenCV Haar Cascade]
        E --> F[Emotion Analysis<br/>ONNX FERPlus Model]
        F --> G[Frame Emotions<br/>7 Emotion Probabilities]
        G --> H[Micro-Expression<br/>Detection Algorithm]
        H --> I[Metrics Calculation]
    end

    subgraph Output
        I --> J[emotion_timeline.html<br/>River Chart Visualization]
        I --> K[emotion_timeline_output.json<br/>Sentio Mind Integration]
    end
```

## High-Level System Flow

```mermaid
flowchart TB
    subgraph Input
        A[Video File<br/>video_sample.mov] --> B[Frame Extraction]
        C[Profile Images<br/>Profiles_1/] --> D[Face Database]
    end

    subgraph Processing
        B --> E[Face Detection<br/>OpenCV Haar Cascade]
        E --> F[Emotion Analysis<br/>ONNX FERPlus Model]
        F --> G[Frame Emotions<br/>7 Emotion Probabilities]
        G --> H[Micro-Expression<br/>Detection Algorithm]
        H --> I[Metrics Calculation]
    end

    subgraph Output
        I --> J[emotion_timeline.html<br/>River Chart Visualization]
        I --> K[emotion_timeline_output.json<br/>Sentio Mind Integration]
    end

    style A fill:#e74c3c,color:#fff
    style C fill:#3498db,color:#fff
    style J fill:#2ecc71,color:#fff
    style K fill:#2ecc71,color:#fff
```

## Emotion Detection Pipeline

```mermaid
flowchart LR
    A[Video Frame] --> B[Convert to<br/>Grayscale]
    B --> C[Face Detection<br/>Haar Cascade]
    C --> D{Face<br/>Found?}
    D -->|Yes| E[Crop & Resize<br/>Face to 64x64]
    D -->|No| F[Return Neutral]
    E --> G[Normalize<br/>Pixel Values]
    G --> H[ONNX Model<br/>Inference]
    H --> I[Softmax<br/>Probabilities]
    I --> J[7 Emotion<br/>Scores]

    style A fill:#3498db,color:#fff
    style J fill:#2ecc71,color:#fff
```

## Micro-Expression Detection Algorithm

```mermaid
flowchart TD
    A[Start Frame Analysis] --> B[Get Current Frame<br/>Emotion Probabilities]
    B --> C{Non-Neutral Emotion<br/>≥ 0.40?}
    C -->|No| D[Move to Next Frame]
    C -->|Yes| E{Previous Frame<br/>Neutral ≥ 0.50?}
    E -->|No| D
    E -->|Yes| F[Track Emotion<br/>Duration]
    F --> G{Duration<br/>< 0.5 seconds?}
    G -->|No| D
    G -->|Yes| H{Next Frame<br/>Neutral ≥ 0.50?}
    H -->|No| D
    H -->|Yes| I[✓ Micro-Expression<br/>Detected!]
    I --> J[Record: Emotion,<br/>Time, Duration, Probability]
    J --> D
    D --> K{More<br/>Frames?}
    K -->|Yes| A
    K -->|No| L[End]

    style I fill:#2ecc71,color:#fff
    style C fill:#f39c12,color:#fff
    style E fill:#f39c12,color:#fff
    style G fill:#f39c12,color:#fff
    style H fill:#f39c12,color:#fff
```

## Derived Metrics Calculation

```mermaid
flowchart LR
    subgraph Suppression Score
        A1[Count Micro-<br/>Expressions] --> B1[Count Quick<br/>Neutral Returns]
        B1 --> C1[Total Transitions /<br/>Max Possible × 100]
        C1 --> D1[Suppression<br/>Score 0-100]
    end

    subgraph Emotional Range Score
        A2[Count Unique<br/>Emotions Detected] --> B2[Diversity Score<br/>50% weight]
        C2[Max Probability<br/>Per Emotion] --> D2[Intensity Score<br/>50% weight]
        B2 --> E2[Combined<br/>Score 0-100]
        D2 --> E2
    end

    style D1 fill:#e74c3c,color:#fff
    style E2 fill:#9b59b6,color:#fff
```

## Output Generation Flow

```mermaid
flowchart TB
    subgraph Data Collection
        A[Frame Emotions Array] --> D[Generate Timeline Data]
        B[Micro-Expressions List] --> D
        C[Calculated Metrics] --> D
    end

    subgraph JSON Output
        D --> E[metadata:<br/>video info, fps, duration]
        D --> F[statistics:<br/>scores, counts]
        D --> G[timeline:<br/>timestamps, emotions]
        D --> H[micro_expressions:<br/>detected events]
        E --> I[emotion_timeline_output.json]
        F --> I
        G --> I
        H --> I
    end

    subgraph HTML Output
        I --> J[Inject Data<br/>into Template]
        J --> K[River Chart<br/>Chart.js]
        J --> L[Stats Cards]
        J --> M[Micro-Expression<br/>List]
        J --> N[Distribution<br/>Bars]
        K --> O[emotion_timeline.html]
        L --> O
        M --> O
        N --> O
    end

    style I fill:#3498db,color:#fff
    style O fill:#2ecc71,color:#fff
```

## Emotion Categories

```mermaid
pie title Emotion Detection Categories
    "Neutral" : 1
    "Happy" : 1
    "Sad" : 1
    "Angry" : 1
    "Surprise" : 1
    "Fear" : 1
    "Disgust" : 1
```

## Class Diagram

```mermaid
classDiagram
    class MicroExpressionDetector {
        -video_path: str
        -profile_dir: str
        -frame_emotions: list
        -micro_expressions: list
        -emotion_detector: EmotionDetector
        +load_profiles()
        +open_video()
        +process_video()
        +detect_micro_expressions()
        +calculate_suppression_score()
        +calculate_emotional_range_score()
        +generate_timeline_data()
        +save_json_output()
        +generate_html_visualization()
    }

    class EmotionDetector {
        <<abstract>>
        +name: str
        +detect(face_image): dict
    }

    class ONNXEmotionDetector {
        -session: InferenceSession
        +detect(face_image): dict
        -_download_model()
        -_softmax(x): array
    }

    class DeepFaceDetector {
        +detect(face_image): dict
    }

    class OpenCVEmotionDetector {
        -face_cascade: CascadeClassifier
        +detect(face_image): dict
    }

    EmotionDetector <|-- ONNXEmotionDetector
    EmotionDetector <|-- DeepFaceDetector
    EmotionDetector <|-- OpenCVEmotionDetector
    MicroExpressionDetector --> EmotionDetector
```

## Sequence Diagram - Full Analysis Flow

```mermaid
sequenceDiagram
    participant User
    participant Main as solution.py
    participant Detector as MicroExpressionDetector
    participant Video as VideoCapture
    participant ONNX as ONNXEmotionDetector
    participant Output as OutputGenerator

    User->>Main: python solution.py
    Main->>Detector: Initialize(video_path)
    Detector->>ONNX: Initialize Model
    ONNX-->>Detector: Model Ready

    Detector->>Video: Open Video
    Video-->>Detector: FPS, Frame Count

    loop For Each Frame
        Detector->>Video: Read Frame
        Video-->>Detector: Frame Data
        Detector->>Detector: Detect Face
        Detector->>ONNX: Analyze Emotions
        ONNX-->>Detector: 7 Emotion Probabilities
        Detector->>Detector: Store Frame Emotions
    end

    Detector->>Detector: detect_micro_expressions()
    Detector->>Detector: calculate_suppression_score()
    Detector->>Detector: calculate_emotional_range_score()

    Detector->>Output: generate_timeline_data()
    Output->>Output: Create JSON
    Output->>Output: Create HTML
    Output-->>User: emotion_timeline.html
    Output-->>User: emotion_timeline_output.json
```

## Problem Statement

While computing one dominant emotion per frame, micro-expressions — genuine emotional flashes lasting < 0.5s before being suppressed — are critical stress signals invisible to single-frame analysis.

## Features

- **Micro-Expression Detection**: Detects emotions lasting < 0.5 seconds
- **Emotion Timeline**: Tracks 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Derived Metrics**:
  - **Suppression Score (0-100)**: How often the person shows micro-expressions but immediately returns to neutral
  - **Emotional Range Score (0-100)**: How varied/expressive the person is across the session
- **River Chart Visualization**: Interactive HTML visualization with Chart.js
- **JSON Output**: Integration-ready data for Sentio Mind

## Micro-Expression Definition

A micro-expression is detected when:
- A non-neutral emotion appears suddenly with probability ≥ 0.40
- It lasts < 0.5 seconds (< ANALYSIS_FPS × 0.5 frames)
- It is preceded AND followed by neutral (probability ≥ 0.50)

### Detection Algorithm Flow

```mermaid
flowchart TD
    A[Start Frame Analysis] --> B[Get Current Frame<br/>Emotion Probabilities]
    B --> C{Non-Neutral Emotion<br/>≥ 0.40?}
    C -->|No| D[Move to Next Frame]
    C -->|Yes| E{Previous Frame<br/>Neutral ≥ 0.50?}
    E -->|No| D
    E -->|Yes| F[Track Emotion<br/>Duration]
    F --> G{Duration<br/>< 0.5 seconds?}
    G -->|No| D
    G -->|Yes| H{Next Frame<br/>Neutral ≥ 0.50?}
    H -->|No| D
    H -->|Yes| I[✓ Micro-Expression<br/>Detected!]
    I --> J[Record: Emotion,<br/>Time, Duration, Probability]
    J --> D
    D --> K{More<br/>Frames?}
    K -->|Yes| A
    K -->|No| L[End]
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage (uses default video)
python solution.py

# Specify video file
python solution.py path/to/video.mp4

# Custom options
python solution.py video.mp4 --fps 10 --output-json results.json --output-html results.html
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | `Class_8_cctv_video_1.mov` | Path to input video file |
| `--profiles` | `Profiles_1-.../Profiles_1` | Path to profile images directory |
| `--fps` | `10` | Analysis frames per second |
| `--output-json` | `emotion_timeline_output.json` | Output JSON file path |
| `--output-html` | `emotion_timeline.html` | Output HTML file path |

## Output Files

### 1. emotion_timeline.html
Interactive river chart visualization showing:
- Stacked area chart with 7 emotion bands (X = time in seconds)
- Micro-expression events marked with vertical dashed lines + tooltips
- Statistics cards (Suppression Score, Emotional Range Score, etc.)
- Emotion distribution breakdown
- Works offline (Chart.js loaded from CDN with fallback)

### 2. emotion_timeline_output.json
JSON structure for Sentio Mind integration:
```json
{
  "emotion_timeline": {
    "metadata": { ... },
    "statistics": {
      "suppression_score": 0-100,
      "emotional_range_score": 0-100,
      "micro_expression_count": N,
      "dominant_emotions": { ... }
    },
    "timeline": {
      "timestamps": [...],
      "emotions": { "angry": [...], "happy": [...], ... }
    },
    "micro_expressions": [...]
  },
  "person_profiles": {
    "primary": {
      "emotion_timeline": { ... }
    }
  }
}
```

## Project Structure

```
Expression/
├── solution.py                    # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── ARCHITECTURE.md                # Detailed Mermaid diagrams
├── .env                           # Environment configuration
├── emotion_timeline.html          # Generated visualization
├── emotion_timeline_output.json   # Generated JSON output
├── models/
│   └── emotion-ferplus-8.onnx     # Auto-downloaded emotion model
└── Profiles_1-.../                # Profile images for face recognition
```

## Technical Details

### Emotion Detection Pipeline
1. **Face Detection**: OpenCV Haar Cascade
2. **Emotion Analysis**: ONNX FERPlus model (auto-downloads on first run)
3. **Fallback Options**: DeepFace → ONNX → OpenCV basic

### Class Diagram

```mermaid
classDiagram
    class MicroExpressionDetector {
        -video_path: str
        -frame_emotions: list
        -micro_expressions: list
        +process_video()
        +detect_micro_expressions()
        +calculate_suppression_score()
        +calculate_emotional_range_score()
        +save_json_output()
        +generate_html_visualization()
    }

    class EmotionDetector {
        <<abstract>>
        +name: str
        +detect(face_image): dict
    }

    class ONNXEmotionDetector {
        -session: InferenceSession
        +detect(face_image): dict
    }

    class DeepFaceDetector {
        +detect(face_image): dict
    }

    EmotionDetector <|-- ONNXEmotionDetector
    EmotionDetector <|-- DeepFaceDetector
    MicroExpressionDetector --> EmotionDetector
```

### Supported Emotions
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Requirements

- Python 3.8+
- OpenCV
- ONNX Runtime
- NumPy

## Demo

After running the script, open `emotion_timeline.html` in a web browser to view the interactive visualization.
