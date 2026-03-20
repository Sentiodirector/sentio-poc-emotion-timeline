# Micro-Expression & Emotion Transition Timeline

---

## Why This Exists

Sentio Mind gives one dominant emotion per frame. That is not enough. A student might flash genuine fear for 0.3 seconds before composing their face back to neutral. That micro-expression is invisible to single-frame analysis but it is an important stress signal. This project builds the engine to detect and track those fleeting moments and turn them into a timeline that reveals what is really going on emotionally across a full session.

---

## What You Receive

```
p6_emotion_timeline/
├── video_sample_1.mov           ← close-up or medium shot of one person
├── emotion_timeline.py          ← your template — copy to solution.py
├── emotion_timeline.json        ← schema for emotion_timeline_output.json
└── README.md
```

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `emotion_timeline.html` — river chart with statistics and transition table, works offline
2. `emotion_timeline_output.json` — follows `emotion_timeline.json` schema exactly

### Micro-Expression Definition (implement exactly this)

A micro-expression is a frame sequence where:
1. A non-neutral emotion appears suddenly with probability ≥ 0.40
2. It lasts less than 0.5 seconds (fewer than `ANALYSIS_FPS × 0.5` frames)
3. It is directly preceded AND followed by neutral (probability ≥ 0.50)

### Two Derived Metrics

**Suppression Score (0–100)**
How often the person flashes a micro-expression and immediately returns to neutral — a stress indicator.
```
suppression_score = (micro_expression_count / total_expression_events) * 100
where total_expression_events = frames where any non-neutral emotion > 0.35
```

**Emotional Range Score (0–100)**
How varied/expressive the person is.
```
range_score = min(100, (distinct_emotions_with_prob_over_30 / 7) * 100 + std(all_probs_over_time) * 2)
```

### Chart Requirements

- Stacked area "river chart" — 7 emotion bands, X = time in seconds
- Micro-expression events marked with vertical dashed lines + tooltip showing emotion
- Works offline — bundle Chart.js or D3.js inline inside the HTML file (no CDN)

### Emotion Model

Try DeepFace first. If not installed, fall back to OpenCV DNN with the FER+ ONNX model:
```bash
# Download once
wget https://github.com/opencv/opencv_zoo/raw/main/models/emotion_ferplus/emotion_ferplus_2022jan.onnx -O models/emotion_ferplus.onnx
```

---

## Hard Rules

- Analyse at 8 fps (downsample from original video)
- Interpolate missing frames (face not visible) from the last known frame
- Do not rename functions in `emotion_timeline.py`
- Do not change key names in `emotion_timeline.json`
- HTML must work offline, no CDN
- Python 3.9+, no Jupyter notebooks

## Libraries

```
opencv-python==4.9.0   deepface==0.0.93   mediapipe==0.10.14   numpy==1.26.4
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `emotion_timeline.html` | River chart + stats |
| 3 | `emotion_timeline_output.json` | Output matching schema |
| 4 | `demo.mp4` | Screen recording under 2 min |

