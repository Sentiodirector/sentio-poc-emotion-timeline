#!/usr/bin/env python3
"""
Sentio Mind - Micro-Expression and Emotion Timeline Detection System
=====================================================================
Analyzes video for emotional states and micro-expressions using DeepFace.
Outputs structured JSON and interactive HTML visualization.

Author: Sentio Mind Platform
Python 3.9+ required
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from deepface import DeepFace

# Emotion colors for HTML visualization
EMOTION_COLORS = {
    'angry': '#FF4444',
    'disgust': '#8B4513',
    'fear': '#9B59B6',
    'happy': '#FFD700',
    'sad': '#4A90D9',
    'surprise': '#FF8C00',
    'neutral': '#95A5A6'
}

# All DeepFace emotions
ALL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion scores to probabilities summing to 1.0."""
    total = sum(emotions.values())
    if total == 0:
        return {e: 1.0/7 for e in ALL_EMOTIONS}
    return {k: v / total for k, v in emotions.items()}


def analyze_video(video_path: str, analysis_fps: int = 5) -> dict:
    """
    Main entry point. Reads video, runs DeepFace per sampled frame,
    returns the complete structured result dict.

    Args:
        video_path: Path to the input video file
        analysis_fps: Frames per second to analyze (default: 5)

    Returns:
        Complete structured result dictionary
    """
    # Validate video file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        sys.exit(1)

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Duration: {duration_seconds:.2f}s")
    print(f"  Total frames: {total_frames}")
    print(f"  Video FPS: {video_fps:.2f}")
    print(f"  Analysis FPS: {analysis_fps}")

    # Calculate frame sampling interval
    frame_interval = max(1, int(video_fps / analysis_fps))
    print(f"  Frame interval: every {frame_interval} frames")

    # Video metadata dict
    video_metadata = {
        'filename': os.path.basename(video_path),
        'duration_seconds': round(duration_seconds, 3),
        'total_frames': total_frames,
        'fps': round(video_fps, 2),
        'analysis_fps': analysis_fps,
        'frames_analyzed': 0
    }

    # Process frames
    frame_emotions = []
    frame_idx = 0
    analyzed_count = 0

    print("\nAnalyzing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only analyze at specified interval
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps if video_fps > 0 else 0

            try:
                # Run DeepFace analysis
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                # Handle result (can be list or dict)
                if isinstance(result, list):
                    result = result[0] if result else None

                if result and 'emotion' in result:
                    raw_emotions = result['emotion']
                    # Normalize to probabilities
                    emotions = normalize_emotions(raw_emotions)
                    dominant = result.get('dominant_emotion', max(emotions, key=emotions.get))

                    frame_data = {
                        'frame_idx': frame_idx,
                        'timestamp': round(timestamp, 3),
                        'emotions': {k: round(v, 4) for k, v in emotions.items()},
                        'dominant': dominant
                    }
                    frame_emotions.append(frame_data)
                    analyzed_count += 1

                    # Progress indicator
                    if analyzed_count % 10 == 0:
                        print(f"  Processed {analyzed_count} frames... (timestamp: {timestamp:.2f}s)")

            except Exception as e:
                # Skip frames where face detection fails
                pass

        frame_idx += 1

    cap.release()

    video_metadata['frames_analyzed'] = analyzed_count
    print(f"\nTotal frames analyzed: {analyzed_count}")

    # Warn if too few frames
    if analyzed_count < 10:
        print("WARNING: Fewer than 10 frames analyzed. Results may be unreliable.")

    # Detect micro-expressions
    print("\nDetecting micro-expressions...")
    micro_expressions = detect_micro_expressions(frame_emotions, analysis_fps)
    print(f"  Found {len(micro_expressions)} micro-expressions")

    # Compute metrics
    suppression = compute_suppression_score(micro_expressions, frame_emotions)
    emotional_range = compute_emotional_range_score(frame_emotions)

    print(f"\nMetrics:")
    print(f"  Suppression Score: {suppression:.2f}")
    print(f"  Emotional Range Score: {emotional_range:.2f}")

    # Build final result
    timeline_data = build_timeline_json(
        frame_emotions,
        micro_expressions,
        suppression,
        emotional_range,
        video_metadata
    )

    return timeline_data


def detect_micro_expressions(frame_emotions: list, analysis_fps: int) -> list:
    """
    Detect micro-expressions in the frame emotion sequence.

    A micro-expression is defined as:
    1. A non-neutral emotion has probability >= 0.40
    2. It lasts < 0.5 seconds (< analysis_fps * 0.5 frames)
    3. It is PRECEDED AND FOLLOWED by neutral frames (neutral >= 0.50)

    Args:
        frame_emotions: List of dicts with frame emotion data
        analysis_fps: Analysis frames per second

    Returns:
        List of micro-expression event dicts
    """
    if len(frame_emotions) < 3:
        return []

    micro_expressions = []
    max_micro_frames = int(analysis_fps * 0.5)  # Maximum frames for micro-expression

    i = 1  # Start from second frame (need preceding neutral)
    while i < len(frame_emotions) - 1:  # Stop before last (need following neutral)
        frame = frame_emotions[i]
        prev_frame = frame_emotions[i - 1]

        # Check if previous frame is neutral
        if prev_frame['emotions'].get('neutral', 0) < 0.50:
            i += 1
            continue

        # Check if current frame has non-neutral emotion >= 0.40
        non_neutral_emotions = {k: v for k, v in frame['emotions'].items() if k != 'neutral'}
        max_emotion = max(non_neutral_emotions.values()) if non_neutral_emotions else 0
        dominant_non_neutral = max(non_neutral_emotions, key=non_neutral_emotions.get) if non_neutral_emotions else None

        if max_emotion < 0.40:
            i += 1
            continue

        # Found potential micro-expression start, find the end
        start_idx = i
        peak_prob = max_emotion
        peak_emotion = dominant_non_neutral
        end_idx = i

        # Extend while non-neutral emotion continues
        j = i + 1
        while j < len(frame_emotions) - 1:
            curr = frame_emotions[j]
            curr_non_neutral = {k: v for k, v in curr['emotions'].items() if k != 'neutral'}
            curr_max = max(curr_non_neutral.values()) if curr_non_neutral else 0

            # If still showing significant non-neutral emotion, extend
            if curr_max >= 0.40 and curr['emotions'].get('neutral', 0) < 0.50:
                if curr_max > peak_prob:
                    peak_prob = curr_max
                    peak_emotion = max(curr_non_neutral, key=curr_non_neutral.get)
                end_idx = j
                j += 1
            else:
                break

        # Check duration constraint (< 0.5 seconds)
        duration_frames = end_idx - start_idx + 1
        if duration_frames >= max_micro_frames:
            i = end_idx + 1
            continue

        # Check if followed by neutral
        if end_idx + 1 < len(frame_emotions):
            next_frame = frame_emotions[end_idx + 1]
            if next_frame['emotions'].get('neutral', 0) < 0.50:
                i = end_idx + 1
                continue
        else:
            i = end_idx + 1
            continue

        # Valid micro-expression found
        start_frame = frame_emotions[start_idx]
        end_frame = frame_emotions[end_idx]

        micro_expr = {
            'start_frame': start_frame['frame_idx'],
            'end_frame': end_frame['frame_idx'],
            'start_time': round(start_frame['timestamp'], 3),
            'end_time': round(end_frame['timestamp'], 3),
            'duration_seconds': round(end_frame['timestamp'] - start_frame['timestamp'], 3),
            'emotion': peak_emotion,
            'peak_probability': round(peak_prob, 4)
        }
        micro_expressions.append(micro_expr)

        # Move past this micro-expression
        i = end_idx + 2

    return micro_expressions


def compute_suppression_score(micro_expressions: list, frame_emotions: list) -> float:
    """
    Compute suppression score (0-100).

    Formula: (micro_expression_count / total_emotion_events) * 100

    An "emotion event" is defined as a sequence of frames with the same dominant emotion.

    Args:
        micro_expressions: List of detected micro-expressions
        frame_emotions: List of frame emotion data

    Returns:
        Suppression score clamped to [0, 100]
    """
    if not frame_emotions:
        return 0.0

    # Count emotion events (transitions between dominant emotions)
    total_events = 0
    prev_dominant = None

    for frame in frame_emotions:
        curr_dominant = frame['dominant']
        if curr_dominant != prev_dominant:
            total_events += 1
            prev_dominant = curr_dominant

    if total_events == 0:
        return 0.0

    micro_count = len(micro_expressions)
    score = (micro_count / total_events) * 100

    return round(min(max(score, 0), 100), 2)


def compute_emotional_range_score(frame_emotions: list) -> float:
    """
    Compute emotional range score (0-100).

    Formula: (unique_emotions_expressed / 7) * 100

    An emotion is considered "expressed" if it was dominant in at least one frame.

    Args:
        frame_emotions: List of frame emotion data

    Returns:
        Emotional range score clamped to [0, 100]
    """
    if not frame_emotions:
        return 0.0

    unique_emotions = set()
    for frame in frame_emotions:
        unique_emotions.add(frame['dominant'])

    score = (len(unique_emotions) / 7) * 100

    return round(min(max(score, 0), 100), 2)


def build_timeline_json(frame_emotions: list, micro_expressions: list,
                        suppression_score: float, emotional_range_score: float,
                        video_metadata: dict) -> dict:
    """
    Assemble the final JSON-serializable result dictionary.

    Args:
        frame_emotions: List of frame emotion data
        micro_expressions: List of micro-expressions
        suppression_score: Computed suppression score
        emotional_range_score: Computed emotional range score
        video_metadata: Video metadata dict

    Returns:
        Complete timeline data dictionary
    """
    # Mark micro-expression frames
    micro_frames = set()
    for me in micro_expressions:
        for fe in frame_emotions:
            if me['start_frame'] <= fe['frame_idx'] <= me['end_frame']:
                micro_frames.add(fe['frame_idx'])

    # Build emotion timeline
    emotion_timeline = []
    for frame in frame_emotions:
        entry = {
            'frame_idx': frame['frame_idx'],
            'timestamp': frame['timestamp'],
            'emotions': frame['emotions'],
            'dominant_emotion': frame['dominant'],
            'is_micro_expression': frame['frame_idx'] in micro_frames
        }
        emotion_timeline.append(entry)

    # Compute emotion distribution
    emotion_distribution = {e: 0.0 for e in ALL_EMOTIONS}
    if frame_emotions:
        for frame in frame_emotions:
            for emotion, prob in frame['emotions'].items():
                emotion_distribution[emotion] += prob
        total = len(frame_emotions)
        emotion_distribution = {k: round(v / total, 4) for k, v in emotion_distribution.items()}

    # Find overall dominant emotion
    dominant_overall = max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else 'neutral'

    # Build summary
    summary = {
        'suppression_score': suppression_score,
        'emotional_range_score': emotional_range_score,
        'dominant_emotion_overall': dominant_overall,
        'micro_expression_count': len(micro_expressions),
        'emotion_distribution': emotion_distribution
    }

    return {
        'video_metadata': video_metadata,
        'emotion_timeline': emotion_timeline,
        'micro_expressions': micro_expressions,
        'summary': summary
    }


def generate_html_report(timeline_data: dict, output_path: str = "emotion_timeline.html") -> None:
    """
    Generate an offline HTML report with river chart visualization.

    Args:
        timeline_data: Complete timeline data dictionary
        output_path: Output HTML file path
    """
    # Extract data for visualization
    timeline = timeline_data['emotion_timeline']
    micro_exprs = timeline_data['micro_expressions']
    summary = timeline_data['summary']
    metadata = timeline_data['video_metadata']

    # Prepare chart data
    timestamps = [entry['timestamp'] for entry in timeline]
    emotion_data = {emotion: [] for emotion in ALL_EMOTIONS}
    for entry in timeline:
        for emotion in ALL_EMOTIONS:
            emotion_data[emotion].append(entry['emotions'].get(emotion, 0))

    # Convert numpy float32 to Python float for JSON serialization
    for emotion in emotion_data:
        emotion_data[emotion] = [float(v) for v in emotion_data[emotion]]

    # JSON encode for JavaScript
    timestamps_json = json.dumps(timestamps)
    emotion_data_json = json.dumps(emotion_data)
    micro_exprs_json = json.dumps(micro_exprs, default=lambda x: float(x) if hasattr(x, 'item') else x)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentio Mind - Emotion Timeline Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        header p {{
            color: #888;
            font-size: 1rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stat-label {{
            color: #888;
            margin-top: 5px;
            font-size: 0.9rem;
        }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .chart-title {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #fff;
        }}
        #emotionChart {{
            width: 100%;
            height: 400px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .micro-expr-list {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .micro-expr-list h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #fff;
        }}
        .micro-expr-item {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid;
        }}
        .micro-expr-emotion {{
            font-weight: bold;
            text-transform: capitalize;
        }}
        .micro-expr-details {{
            color: #888;
            font-size: 0.85rem;
        }}
        .micro-expr-prob {{
            font-size: 1.2rem;
            font-weight: bold;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }}
        .metadata {{
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sentio Mind</h1>
            <p>Emotion Timeline Analysis - {metadata['filename']}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{summary['suppression_score']:.1f}</div>
                <div class="stat-label">Suppression Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['emotional_range_score']:.1f}</div>
                <div class="stat-label">Emotional Range Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="font-size: 1.5rem; text-transform: capitalize;">{summary['dominant_emotion_overall']}</div>
                <div class="stat-label">Dominant Emotion</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['micro_expression_count']}</div>
                <div class="stat-label">Micro-Expressions</div>
            </div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Emotion River Chart</h2>
            <canvas id="emotionChart"></canvas>
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

        <div class="micro-expr-list">
            <h2>Micro-Expressions Detected</h2>
            <div id="microExprContainer"></div>
        </div>

        <div class="metadata">
            <p>Duration: {metadata['duration_seconds']:.2f}s | Frames Analyzed: {metadata['frames_analyzed']} | Analysis FPS: {metadata['analysis_fps']}</p>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

    <script>
    // Data from analysis
    const timestamps = {timestamps_json};
    const emotionData = {emotion_data_json};
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

    // Initialize chart
    document.addEventListener('DOMContentLoaded', function() {{
        const canvas = document.getElementById('emotionChart');
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = canvas.parentElement.clientWidth - 60;
        canvas.height = 400;

        const width = canvas.width;
        const height = canvas.height;
        const padding = {{ top: 20, right: 30, bottom: 40, left: 50 }};
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Scale functions
        const xScale = (val) => padding.left + (val / Math.max(...timestamps)) * chartWidth;
        const yScale = (val) => padding.top + chartHeight - val * chartHeight;

        // Draw stacked area chart
        const emotions = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry'];

        // Calculate cumulative values for stacking
        const stackedData = timestamps.map((t, i) => {{
            let cumulative = 0;
            const point = {{ time: t }};
            emotions.forEach(emotion => {{
                cumulative += emotionData[emotion][i] || 0;
                point[emotion] = cumulative;
            }});
            return point;
        }});

        // Draw areas from bottom to top
        emotions.forEach((emotion, emotionIdx) => {{
            ctx.beginPath();
            ctx.fillStyle = emotionColors[emotion] + '99';

            // Top line
            stackedData.forEach((point, i) => {{
                const x = xScale(point.time);
                const y = yScale(point[emotion]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});

            // Bottom line (previous emotion's top or baseline)
            const prevEmotion = emotions[emotionIdx - 1];
            for (let i = stackedData.length - 1; i >= 0; i--) {{
                const point = stackedData[i];
                const x = xScale(point.time);
                const y = prevEmotion ? yScale(point[prevEmotion]) : yScale(0);
                ctx.lineTo(x, y);
            }}

            ctx.closePath();
            ctx.fill();
        }});

        // Draw micro-expression markers
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;

        microExpressions.forEach(me => {{
            const x = xScale(me.start_time);
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
        }});

        ctx.setLineDash([]);

        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;

        // X axis
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();

        // Y axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();

        // X axis labels
        ctx.fillStyle = '#888';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        const maxTime = Math.max(...timestamps);
        for (let t = 0; t <= maxTime; t += Math.ceil(maxTime / 10)) {{
            const x = xScale(t);
            ctx.fillText(t.toFixed(1) + 's', x, height - padding.bottom + 20);
        }}

        // Y axis labels
        ctx.textAlign = 'right';
        for (let i = 0; i <= 1; i += 0.25) {{
            const y = yScale(i);
            ctx.fillText((i * 100).toFixed(0) + '%', padding.left - 10, y + 4);
        }}

        // Axis titles
        ctx.fillStyle = '#aaa';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (seconds)', width / 2, height - 5);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Emotion Probability', 0, 0);
        ctx.restore();

        // Populate micro-expressions list
        const container = document.getElementById('microExprContainer');
        if (microExpressions.length === 0) {{
            container.innerHTML = '<p style="color: #888; text-align: center;">No micro-expressions detected in this video.</p>';
        }} else {{
            microExpressions.forEach((me, i) => {{
                const div = document.createElement('div');
                div.className = 'micro-expr-item';
                div.style.borderColor = emotionColors[me.emotion];
                div.innerHTML = `
                    <div>
                        <div class="micro-expr-emotion" style="color: ${{emotionColors[me.emotion]}}">${{me.emotion}}</div>
                        <div class="micro-expr-details">
                            ${{me.start_time.toFixed(2)}}s - ${{me.end_time.toFixed(2)}}s (${{(me.duration_seconds * 1000).toFixed(0)}}ms)
                        </div>
                    </div>
                    <div class="micro-expr-prob" style="color: ${{emotionColors[me.emotion]}}">
                        ${{(me.peak_probability * 100).toFixed(1)}}%
                    </div>
                `;
                container.appendChild(div);
            }});
        }}

        // Tooltip handling
        const tooltip = document.getElementById('tooltip');
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Check if near a micro-expression line
            let found = false;
            microExpressions.forEach(me => {{
                const meX = xScale(me.start_time);
                if (Math.abs(x - meX) < 10) {{
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX + 10) + 'px';
                    tooltip.style.top = (e.clientY - 10) + 'px';
                    tooltip.innerHTML = `
                        <strong style="color: ${{emotionColors[me.emotion]}}">${{me.emotion.toUpperCase()}}</strong><br>
                        Time: ${{me.start_time.toFixed(2)}}s<br>
                        Duration: ${{(me.duration_seconds * 1000).toFixed(0)}}ms<br>
                        Peak: ${{(me.peak_probability * 100).toFixed(1)}}%
                    `;
                    found = true;
                }}
            }});

            if (!found) {{
                tooltip.style.display = 'none';
            }}
        }});

        canvas.addEventListener('mouseleave', () => {{
            tooltip.style.display = 'none';
        }});
    }});
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report saved to: {output_path}")


def save_json_output(timeline_data: dict, output_path: str = "emotion_timeline_output.json") -> None:
    """
    Save the timeline data to a JSON file.

    Args:
        timeline_data: Complete timeline data dictionary
        output_path: Output JSON file path
    """
    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False, default=convert_numpy)

    print(f"JSON output saved to: {output_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Sentio Mind - Micro-Expression and Emotion Timeline Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default='video_sample_1.mov',
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

    print("=" * 60)
    print("SENTIO MIND - Emotion Timeline Analysis")
    print("=" * 60)
    print()

    # Analyze video
    timeline_data = analyze_video(args.video, args.fps)

    # Generate outputs
    print("\nGenerating outputs...")
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
    print()


if __name__ == '__main__':
    main()
