# Micro-Expression & Emotion Transition Timeline
**Sentio Mind · Submission · Project 6**

## Overview
This project implements a detection and tracking engine for sub-second emotional flashes (micro-expressions) and long-term emotional transitions. It analyzes video at 8 FPS, detects the primary face using Haar Cascades, and evaluates emotions via an ONNX-based model.

## Deliverables
- `solution.py`: Main processing script.
- `emotion_timeline.html`: Premium offline river chart showing emotion distribution over time.
- `emotion_timeline_output.json`: Integration-ready JSON data following the required schema.

## Key Features
- **Offline River Chart:** A custom SVG-based stacked area chart that renders perfectly without any internet connection (no CDN dependencies).
- **Micro-Expression Logic:** Detects non-neutral emotions lasting <0.5s preceded/followed by neutral states.
- **Derived Metrics:** 
  - **Suppression Score:** Measures how frequently the user suppresses genuine emotional flashes.
  - **Emotional Range Score:** Measures the variance and richness of the user's emotional state.

## Installation & Running
1. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
2. Run the analysis:
   ```bash
   python solution.py
   ```
3. Open `emotion_timeline.html` to view the report.

## Submission details
- **Branch:** `[FirstName_LastName_RollNumber]`
- **Repository:** Private GitHub invited to `Sentiodirector`.
