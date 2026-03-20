#!/usr/bin/env python3
"""
solution.py — Micro-Expression & Emotion Timeline System
=========================================================
Final updated version handling image preprocessing, unified JSON/HTML, and exact constraints.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_ORDER: List[str] = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]
NON_NEUTRAL: List[str] = [
    "angry", "disgust", "fear", "happy", "sad", "surprise"
]

class Config:
    WORKSPACE_DIR: Path = Path(__file__).resolve().parent
    PROFILES_DIR:  str  = str(WORKSPACE_DIR / "Profiles_1")
    VIDEO_PATH:    str  = str(WORKSPACE_DIR / "Video_1" / "Class_8_cctv_video_1.mov")
    OUTPUT_JSON:   str  = str(WORKSPACE_DIR / "emotion_timeline_output.json")
    OUTPUT_HTML:   str  = str(WORKSPACE_DIR / "emotion_timeline.html")
    MODELS_DIR:    Path = WORKSPACE_DIR / "models"
    DEFAULT_VIDEO: str  = "video_sample_1.mov"
    DEFAULT_FPS:   int  = 8
    ANALYSIS_FPS:  int  = 8
    MICRO_MAX_DURATION_SEC: float = 0.5
    MICRO_MIN_PEAK_PCT:     float = 40.0
    NEUTRAL_SUPPRESS_PCT:   float = 50.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _r6(value: float) -> float:
    return round(float(value), 6)

def _dominant(emotions: Dict[str, Any]) -> str:
    return max(EMOTION_ORDER, key=lambda e: emotions.get(e, 0))

def _infer_fps(times: List[float], default: int = Config.DEFAULT_FPS) -> int:
    if len(times) < 2:
        return default
    deltas = [times[i] - times[i - 1] for i in range(1, len(times)) if times[i] > times[i - 1]]
    if not deltas:
        return default
    med = median(deltas)
    fps = int(round(1.0 / med)) if med > 0 else default
    return fps if fps > 0 else default

def _estimate_frame_interval(frame_series: List[Dict[str, Any]], default: float = 1.0 / Config.DEFAULT_FPS) -> float:
    if len(frame_series) < 2:
        return default
    deltas = [
        _safe_float(frame_series[i]["t"]) - _safe_float(frame_series[i - 1]["t"])
        for i in range(1, len(frame_series))
        if _safe_float(frame_series[i]["t"]) > _safe_float(frame_series[i - 1]["t"])
    ]
    return median(deltas) if deltas else default

# ---------------------------------------------------------------------------
# Fix 7 — Normalise emotions: EXACTLY 100
# ---------------------------------------------------------------------------

def _normalise_to_pct(emotions):
    EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    raw = {e: float(emotions.get(e, 0.0)) for e in EMOTIONS}
    total = sum(raw.values())
    if total <= 0:
        return {e: (100 if e == "neutral" else 0) for e in EMOTIONS}
    scaled = {e: raw[e] / total * 100 for e in EMOTIONS}
    floored = {e: int(scaled[e]) for e in EMOTIONS}
    remainder = 100 - sum(floored.values())
    fractions = sorted(EMOTIONS, key=lambda e: scaled[e] - floored[e], reverse=True)
    for i in range(remainder):
        floored[fractions[i]] += 1
    return floored

def _build_frame_series(raw_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    series = []
    for frame in raw_frames:
        t = _safe_float(frame.get("time", frame.get("t", 0.0)))
        emotions_raw = frame.get("emotions", {})
        if not isinstance(emotions_raw, dict):
            emotions_raw = {}
        series.append({"t": _r6(t), "emotions": _normalise_to_pct(emotions_raw)})
    series.sort(key=lambda f: f["t"])
    return series

def _compute_emotion_stats(frame_series: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float]]:
    totals = {e: 0.0 for e in EMOTION_ORDER}
    for frame in frame_series:
        for e in EMOTION_ORDER:
            totals[e] += _safe_float(frame["emotions"].get(e, 0.0))
    grand_total = sum(totals.values())
    if grand_total <= 0:
        pct = {e: 0.0 for e in EMOTION_ORDER}
    else:
        pct = {e: _r6((totals[e] / grand_total) * 100.0) for e in EMOTION_ORDER}
    dominant = max(EMOTION_ORDER, key=lambda e: totals[e]) if frame_series else "neutral"
    return dominant, pct

# ---------------------------------------------------------------------------
# Fix 6 — Transitions: 0.5-second hold
# ---------------------------------------------------------------------------

def _build_transitions(frame_series):
    HOLD_SEC = 0.5
    transitions = []
    if len(frame_series) < 2:
        return transitions

    prev_dom = _dominant(frame_series[0]["emotions"])
    candidate_dom = None
    candidate_start_t = None
    candidate_from = None

    for frame in frame_series[1:]:
        curr_t = frame["t"]
        curr_dom = _dominant(frame["emotions"])

        if curr_dom != prev_dom:
            if candidate_dom != curr_dom:
                candidate_dom = curr_dom
                candidate_start_t = curr_t
                candidate_from = prev_dom
        else:
            if candidate_dom == curr_dom:
                held = curr_t - candidate_start_t
                if held >= HOLD_SEC:
                    transitions.append({
                        "from_emotion": candidate_from,
                        "to_emotion": curr_dom,
                        "timestamp_sec": _r6(candidate_start_t),
                        "transition_duration_sec": _r6(held),
                    })
                    prev_dom = curr_dom
                    candidate_dom = None
                    candidate_start_t = None
                    candidate_from = None
    return transitions

# ---------------------------------------------------------------------------
# Fix 5 — Micro-expressions: neutral BEFORE AND after
# ---------------------------------------------------------------------------

def _build_micro_expressions(frame_series, max_duration_sec=0.5, min_peak_pct=40.0, neutral_suppress_pct=50.0):
    if not frame_series:
        return []
    frame_interval = _estimate_frame_interval(frame_series)
    dominants = [_dominant(f["emotions"]) for f in frame_series]

    runs = []
    i = 0
    while i < len(dominants):
        j = i + 1
        while j < len(dominants) and dominants[j] == dominants[i]:
            j += 1
        runs.append((dominants[i], i, j - 1))
        i = j

    micro_exprs = []
    m_id = 1

    for run_idx, (emotion, start_i, end_i) in enumerate(runs):
        if emotion not in NON_NEUTRAL:
            continue

        if run_idx == 0:
            continue  # no frame before, reject

        prev_run_end_i = runs[run_idx - 1][2]
        neutral_before = frame_series[prev_run_end_i]["emotions"].get("neutral", 0.0)
        if neutral_before < neutral_suppress_pct:
            continue  # not preceded by neutral, reject

        start_t  = _safe_float(frame_series[start_i]["t"])
        end_t    = _safe_float(frame_series[end_i]["t"])
        duration = max(0.0, (end_t - start_t) if end_i > start_i else frame_interval)

        if duration > max_duration_sec:
            continue

        peak = max(_safe_float(frame_series[k]["emotions"].get(emotion, 0.0)) for k in range(start_i, end_i + 1))
        if peak < min_peak_pct:
            continue

        if run_idx + 1 < len(runs):
            followed_by   = runs[run_idx + 1][0]
            next_start_i  = runs[run_idx + 1][1]
            neutral_after = _safe_float(frame_series[next_start_i]["emotions"].get("neutral", 0.0))
        else:
            followed_by   = emotion
            neutral_after = 0.0

        is_suppressed = bool(
            duration <= max_duration_sec
            and followed_by == "neutral"
            and neutral_after >= neutral_suppress_pct
        )

        micro_exprs.append({
            "id":               m_id,
            "timestamp_sec":    _r6(start_t),
            "duration_sec":     _r6(duration),
            "emotion":          emotion,
            "peak_probability": _r6(float(peak)),
            "followed_by":      followed_by,
            "is_suppressed":    is_suppressed,
        })
        m_id += 1

    return micro_exprs

# ---------------------------------------------------------------------------
# Fix 3 — Suppression score formula
# ---------------------------------------------------------------------------

def _suppression_score(micro_expressions, frame_series):
    total_expression_events = sum(
        1 for f in frame_series
        if any(
            f["emotions"].get(e, 0) > 35
            for e in ["angry","disgust","fear","happy","sad","surprise"]
        )
    )
    if total_expression_events == 0:
        return 0
    return min(100, int((len(micro_expressions) / total_expression_events) * 100))

# ---------------------------------------------------------------------------
# Fix 4 — Emotional range score formula
# ---------------------------------------------------------------------------

def _emotional_range_score(frame_series):
    EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    distinct = sum(
        1 for e in EMOTIONS
        if any(f["emotions"].get(e, 0) > 30 for f in frame_series)
    )
    all_probs = [
        f["emotions"].get(e, 0)
        for f in frame_series for e in EMOTIONS
    ]
    mean = sum(all_probs) / len(all_probs) if all_probs else 0
    std = (sum((x-mean)**2 for x in all_probs) / len(all_probs))**0.5 if all_probs else 0
    return min(100, int((distinct / 7) * 100 + std * 2))

# ---------------------------------------------------------------------------
# Build standard timeline struct
# ---------------------------------------------------------------------------

def build_emotion_timeline_output(raw_frames, video_name="video_sample_1.mov", fps_hint=None):
    frame_series = _build_frame_series(raw_frames)
    times = [_safe_float(f["t"]) for f in frame_series]
    dominant, pct = _compute_emotion_stats(frame_series)
    transitions = _build_transitions(frame_series)
    micro_exprs = _build_micro_expressions(frame_series)
    sup_score = _suppression_score(micro_exprs, frame_series)
    rng_score = _emotional_range_score(frame_series)
    duration_sec = _r6(times[-1]) if times else 0.0
    fps_analyzed = int(fps_hint) if fps_hint else _infer_fps(times)

    return {
        "source": "p6_emotion_timeline",
        "video": str(video_name),
        "duration_sec": duration_sec,
        "fps_analyzed": fps_analyzed,
        "dominant_emotion": dominant,
        "emotion_time_pct": {e: pct[e] for e in EMOTION_ORDER},
        "suppression_score": sup_score,
        "emotional_range_score": rng_score,
        "micro_expressions": micro_exprs,
        "transitions": transitions,
        "frame_series": frame_series,
    }

def _write_json(data, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Fix 8 — Chart.js Offline
# ---------------------------------------------------------------------------

def _get_chartjs_inline() -> str:
    cache_dir = Path("models")
    cache_dir.mkdir(exist_ok=True)
    chartjs_path = cache_dir / "chart.min.js"
    annotate_path = cache_dir / "chartjs-annotation.min.js"

    if not chartjs_path.exists():
        print("Downloading Chart.js for offline use...")
        urllib.request.urlretrieve(
            "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js",
            chartjs_path
        )
    if not annotate_path.exists():
        print("Downloading annotation plugin for offline use...")
        urllib.request.urlretrieve(
            "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/1.4.0/chartjs-plugin-annotation.min.js",
            annotate_path
        )

    return (
        f"<script>{chartjs_path.read_text(encoding='utf-8')}</script>\n"
        f"<script>{annotate_path.read_text(encoding='utf-8')}</script>"
    )

# ---------------------------------------------------------------------------
# Fix 9 — Combined HTML UI
# ---------------------------------------------------------------------------

_EMOTION_BG = { "angry": "rgba(214,39,40,0.7)", "disgust": "rgba(44,160,44,0.7)", "fear": "rgba(31,119,180,0.7)", "happy": "rgba(255,127,14,0.7)", "neutral": "rgba(127,127,127,0.7)", "sad": "rgba(148,103,189,0.7)", "surprise": "rgba(23,190,207,0.7)" }
_EMOTION_BD = { e: c.replace("0.7)", "1)") for e, c in _EMOTION_BG.items() }

def _write_html_multi(person_profiles: Dict[str, Any], output_path: str):
    students = list(person_profiles.keys())
    if not students:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<html><body>No data available.</body></html>")
        return

    scripts = _get_chartjs_inline()
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sentio Emotion Timeline - All Students</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px 20px 48px; margin: 0; }}
.header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }}
.header h1 {{ font-size: 1.5rem; font-weight: 700; margin: 0; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid #2d3148; padding-bottom: 10px; }}
.tab-btn {{ background: #1e2130; border: 1px solid #2d3148; color: #94a3b8; padding: 8px 16px; border-radius: 8px; cursor: pointer; transition: 0.2s; font-weight: 600; outline: none; }}
.tab-btn:hover {{ background: #2d3148; color: #fff; }}
.tab-btn.active {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; border-color: transparent; }}
.student-panel {{ display: none; }}
.student-panel.active {{ display: block; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 14px; margin-bottom: 28px; }}
.card {{ background: #1e2130; border: 1px solid #2d3148; border-radius: 14px; padding: 16px 18px; }}
.card .label {{ font-size: .7rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }}
.card .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 6px; color: #f1f5f9; }}
.card .value.accent {{ color: #818cf8; }}
.card .sub {{ font-size: .75rem; color: #475569; margin-top: 3px; }}
.panel {{ background: #1e2130; border: 1px solid #2d3148; border-radius: 16px; padding: 22px 24px; margin-bottom: 24px; }}
.panel h2 {{ margin-top: 0; font-size: 1rem; color: #cbd5e1; margin-bottom: 18px; display: flex; align-items: center; gap: 8px; }}
.panel h2::before {{ content: ''; width: 4px; height: 16px; background: linear-gradient(#6366f1, #8b5cf6); border-radius: 2px; }}
.chart-wrap {{ position: relative; height: 340px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
th {{ padding: 8px 12px; color: #64748b; text-align: left; border-bottom: 2px solid #2d3148; }}
td {{ padding: 9px 12px; border-bottom: 1px solid #1a1d2e; color: #cbd5e1; }}
tr:hover td {{ background: #232640; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: .78rem; font-weight: 600; }}
.badge-angry {{ background: rgba(214,39,40,.2); color: #f87171; }}
.badge-disgust {{ background: rgba(44,160,44,.2); color: #4ade80; }}
.badge-fear {{ background: rgba(31,119,180,.2); color: #60a5fa; }}
.badge-happy {{ background: rgba(255,127,14,.2); color: #fb923c; }}
.badge-neutral {{ background: rgba(127,127,127,.2); color: #94a3b8; }}
.badge-sad {{ background: rgba(148,103,189,.2); color: #c084fc; }}
.badge-surprise {{ background: rgba(23,190,207,.2); color: #22d3ee; }}
td.empty {{ text-align: center; color: #475569; padding: 20px; }}
</style>
{scripts}
</head>
<body>
<div class="header">
  <div style="font-size: 2rem;">🧠</div>
  <div><h1>Sentio Emotion Timeline</h1><p style="margin:2px 0 0;color:#94a3b8;font-size:0.85rem;">Combined Multi-Student Report</p></div>
</div>

<div class="tabs" id="tabContainer">
"""
    for idx, student in enumerate(students):
        active_cls = "active" if idx == 0 else ""
        html += f'<button class="tab-btn {active_cls}" onclick="switchTab(\'{student}\')">{student}</button>\n'

    html += "</div>\n\n<div id=\"contentContainer\">\n"
    
    js_datasets = {}

    for idx, student in enumerate(students):
        out = person_profiles[student]["emotion_timeline"]
        active_cls = "active" if idx == 0 else ""
        
        fs = out.get("frame_series", [])
        dur = out.get("duration_sec", 0)
        fps = out.get("fps_analyzed", 0)
        dom = out.get("dominant_emotion", "—")
        sup = out.get("suppression_score", 0)
        rng = out.get("emotional_range_score", 0)
        m_arr = out.get("micro_expressions", [])
        t_arr = out.get("transitions", [])
        
        micro_html = "".join(f"<tr><td>{m['id']}</td><td>{m['timestamp_sec']}s</td><td>{m['duration_sec']}s</td><td><span class='badge badge-{m['emotion']}'>{m['emotion']}</span></td><td>{m['peak_probability']}%</td><td>{m['followed_by']}</td><td>{'Yes' if m.get('is_suppressed') else '—'}</td></tr>" for m in m_arr) or "<tr><td colspan='7' class='empty'>No micro-expressions</td></tr>"
        trans_html = "".join(f"<tr><td>{t['timestamp_sec']}s</td><td><span class='badge badge-{t['from_emotion']}'>{t['from_emotion']}</span></td><td>→</td><td><span class='badge badge-{t['to_emotion']}'>{t['to_emotion']}</span></td><td>{t['transition_duration_sec']}s</td></tr>" for t in t_arr) or "<tr><td colspan='5' class='empty'>No transitions</td></tr>"
        
        html += f"""
<div id="panel_{student}" class="student-panel {active_cls}">
  <div class="cards">
    <div class="card"><div class="label">Duration</div><div class="value">{dur}s</div></div>
    <div class="card"><div class="label">FPS Analysed</div><div class="value">{fps}</div></div>
    <div class="card"><div class="label">Dominant Emotion</div><div class="value accent">{dom}</div></div>
    <div class="card"><div class="label">Suppression Score</div><div class="value">{sup}</div></div>
    <div class="card"><div class="label">Emotional Range</div><div class="value">{rng}</div></div>
    <div class="card"><div class="label">Micro-Expressions</div><div class="value">{len(m_arr)}</div></div>
    <div class="card"><div class="label">Transitions</div><div class="value">{len(t_arr)}</div></div>
  </div>
  
  <div class="panel">
    <h2>Timeline</h2>
    <div class="chart-wrap"><canvas id="chart_{student}"></canvas></div>
  </div>
  
  <div class="panel">
    <h2>Micro-Expressions</h2>
    <table><thead><tr><th>#</th><th>Timestamp</th><th>Duration</th><th>Emotion</th><th>Peak</th><th>Followed By</th><th>Suppressed</th></tr></thead><tbody>{micro_html}</tbody></table>
  </div>
  
  <div class="panel">
    <h2>Transitions</h2>
    <table><thead><tr><th>Timestamp</th><th>From</th><th></th><th>To</th><th>Duration</th></tr></thead><tbody>{trans_html}</tbody></table>
  </div>
</div>
"""
        
        labels = [_safe_float(f["t"]) for f in fs]
        datasets = []
        for e in EMOTION_ORDER:
            datasets.append({
                "label": e.capitalize(),
                "data": [_safe_float(f["emotions"].get(e,0)) for f in fs],
                "backgroundColor": _EMOTION_BG[e],
                "borderColor": _EMOTION_BD[e],
                "borderWidth": 1, "fill": True, "tension": 0.3, "pointRadius": 0
            })
            
        annotations = {}
        for m in m_arr:
            annotations[f"m_{m['id']}"] = {
                "type": "line", "xMin": m["timestamp_sec"], "xMax": m["timestamp_sec"],
                "borderColor": "rgba(255,50,50,0.8)", "borderWidth": 2, "borderDash": [4,4],
                "label": {"display": True, "content": f"{m['emotion']} {m['peak_probability']}%", "position": "start", "backgroundColor":"rgba(0,0,0,0.7)", "color":"#fff", "font":{"size":10}}
            }
            
        js_datasets[student] = {"labels": labels, "datasets": datasets, "annotations": annotations}

    html += """</div>

<script>
const chartData = """ + json.dumps(js_datasets) + """;
const renderedCharts = {};

function switchTab(student) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.student-panel').forEach(p => p.classList.remove('active'));
    
    document.querySelector(`.tab-btn[onclick="switchTab('${student}')"]`).classList.add('active');
    document.getElementById(`panel_${student}`).classList.add('active');
    
    renderChart(student);
}

function renderChart(student) {
    if(renderedCharts[student]) return;
    
    const ctx = document.getElementById(`chart_${student}`).getContext('2d');
    const data = chartData[student];
    
    renderedCharts[student] = new Chart(ctx, {
        type: 'line',
        data: { labels: data.labels, datasets: data.datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { type: 'linear', ticks:{color:'#64748b'}, grid:{color:'#1a1d2e'} },
                y: { stacked: true, min: 0, max: 100, ticks:{color:'#64748b'}, grid:{color:'#1a1d2e'} }
            },
            plugins: {
                legend: { position: 'bottom', labels: { color: '#94a3b8' } },
                annotation: { annotations: data.annotations }
            }
        }
    });
}
"""
    if students:
        html += f"window.onload = () => renderChart('{students[0]}');\n"

    html += "</script></body></html>"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

# ===========================================================================
# Fix 1 — load_profiles: PREPROCESSING
# ===========================================================================

def load_profiles(profiles_dir: str) -> Dict[str, object]:
    import face_recognition

    profiles = {}
    profiles_path = Path(profiles_dir)
    if not profiles_path.is_dir():
        print(f"WARNING: Profiles folder not found: {profiles_dir}")
        return profiles

    for img_file in sorted(profiles_path.iterdir()):
        if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        student_name = img_file.stem
        try:
            # Load and convert to RGB
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                print(f"  FAILED (cannot read): {student_name}")
                continue

            # Histogram equalization to fix dark/low contrast
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # Convert to RGB for face_recognition
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Try HOG model first
            encodings = face_recognition.face_encodings(
                img_rgb,
                face_recognition.face_locations(img_rgb, model="hog")
            )

            # Fallback: upsample=2 for small faces
            if not encodings:
                encodings = face_recognition.face_encodings(
                    img_rgb,
                    face_recognition.face_locations(
                        img_rgb, number_of_times_to_upsample=2, model="hog"
                    )
                )

            # Fallback: CNN model
            if not encodings:
                encodings = face_recognition.face_encodings(
                    img_rgb,
                    face_recognition.face_locations(img_rgb, model="cnn")
                )

            if encodings:
                profiles[student_name] = encodings[0]
                print(f"  LOADED: {student_name}")
            else:
                print(f"  FAILED (no face detected): {student_name}")

        except Exception as e:
            print(f"  FAILED (error): {student_name} — {e}")

    print(f"\nProfiles loaded: {len(profiles)}/10")
    print(f"Loaded: {list(profiles.keys())}")
    missing = [s for s in ["AARAV","Anamika","Harshita","Jahanvi","Manav",
                            "Mohini","Ranjhana","Rohini","Sparshita","Sushmita"]
               if s not in profiles]
    if missing:
        print(f"Missing: {missing}")
    return profiles

# ===========================================================================
# Fix 2 — process_video
# ===========================================================================

def process_video(video_path, profile_encodings, analysis_fps=8):
    import face_recognition
    from deepface import DeepFace

    TOLERANCE = 0.65  # increased from 0.6

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(native_fps / analysis_fps)))

    known_names = list(profile_encodings.keys())
    known_encodings = list(profile_encodings.values())

    per_person = {}
    unknown_debug_saved = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / native_fps
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, locations)

            for enc, loc in zip(encodings, locations):
                # Best match using face_distance (closest = best)
                name = f"Unknown_{frame_idx}"
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] < TOLERANCE:
                        name = known_names[best_idx]

                # Save debug crop for unknowns
                if name.startswith("Unknown") and name not in unknown_debug_saved:
                    top, right, bottom, left = [v * 4 for v in loc]
                    face_crop = frame[
                        max(0, top-10):min(frame.shape[0], bottom+10),
                        max(0, left-10):min(frame.shape[1], right+10)
                    ]
                    if face_crop.size > 0:
                        cv2.imwrite(f"debug_unknown_{name}.jpg", face_crop)
                        print(f"  Saved debug crop: debug_unknown_{name}.jpg")
                    unknown_debug_saved.add(name)

                # Emotion analysis
                top, right, bottom, left = [v * 4 for v in loc]
                pad = 20
                h, w = frame.shape[:2]
                face_img = frame[
                    max(0, top-pad):min(h, bottom+pad),
                    max(0, left-pad):min(w, right+pad)
                ]
                if face_img.size == 0:
                    continue

                try:
                    try:
                        result = DeepFace.analyze(
                            face_img,
                            actions=["emotion"],
                            enforce_detection=False,
                            silent=True,
                        )
                    except TypeError:
                        # Fallback for deepface v0.0.93 without 'silent'
                        result = DeepFace.analyze(
                            face_img,
                            actions=["emotion"],
                            enforce_detection=False,
                        )
                        
                    raw = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
                    total = sum(raw.values()) or 1.0
                    emotions = {k: v / total * 100 for k, v in raw.items()}
                except Exception as exc:
                    print(f"  [DeepFace Error] {exc}")
                    emotions = {e: (100.0 if e == "neutral" else 0.0)
                                for e in ["angry","disgust","fear","happy","neutral","sad","surprise"]}

                per_person.setdefault(name, []).append({
                    "time": round(timestamp, 3),
                    "emotions": emotions
                })

        frame_idx += 1

    cap.release()
    print(f"\nUnique people detected: {len(per_person)}")
    print(f"People: {list(per_person.keys())}")
    return {"per_person": per_person}

# ===========================================================================
# Fix 10 — main
# ===========================================================================

def main():
    print("STEP 1 — Loading student profiles...")
    profile_encodings = load_profiles(Config.PROFILES_DIR)

    print("\nSTEP 2 — Processing video...")
    raw_data = process_video(Config.VIDEO_PATH, profile_encodings, analysis_fps=8)
    per_person = raw_data.get("per_person", {})

    print("\nSTEP 3 — Building timelines per student...")
    person_profiles = {}
    for student_name, frames in per_person.items():
        if student_name.startswith("Unknown"):
            print(f"  Skipping {student_name} (unrecognized face)")
            continue
        print(f"  Processing {student_name} ({len(frames)} frames)...")
        output = build_emotion_timeline_output(
            frames,
            video_name="video_sample_1.mov",
            fps_hint=8
        )
        output["source"] = "p6_emotion_timeline"
        person_profiles[student_name] = {"emotion_timeline": output}
        print(f"    Dominant: {output['dominant_emotion']} | "
              f"Micro: {len(output['micro_expressions'])} | "
              f"Suppression: {output['suppression_score']} | "
              f"Range: {output['emotional_range_score']}")

    print("\nSTEP 4 — Writing combined output...")
    final_output = {"person_profiles": person_profiles}
    _write_json(final_output, Config.OUTPUT_JSON)
    _write_html_multi(person_profiles, Config.OUTPUT_HTML)

    print("=" * 50)
    print(f"Students processed: {list(person_profiles.keys())}")
    print(f"JSON  → {Path(Config.OUTPUT_JSON).name}")
    print(f"HTML  → {Path(Config.OUTPUT_HTML).name}")
    print("✅ Done!")

if __name__ == "__main__":
    main()
