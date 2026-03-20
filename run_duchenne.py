"""
Quick script: Add Duchenne smile classification to existing data.
Reads the existing JSON + video, runs Face Mesh ONLY on frames where happy >= 40%,
then regenerates the HTML.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import solution

# Load existing data
data = json.load(open("emotion_timeline_output.json"))
frame_series = data["frame_series"]

# Find frames where happy is dominant and >= 40
happy_indices = []
for i, fs in enumerate(frame_series):
    dom = max(fs["emotions"], key=fs["emotions"].get)
    if dom == "happy" and fs["emotions"]["happy"] >= 40:
        happy_indices.append(i)

print(f"Total frames: {len(frame_series)}")
print(f"Happy frames to check: {len(happy_indices)}")

# Init Face Mesh
face_mesh = solution._init_face_mesh()
if not face_mesh:
    print("ERROR: mediapipe not available")
    exit(1)

# Open video
cap = cv2.VideoCapture(str(solution.VIDEO_PATH))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

# For each happy frame, seek to that timestamp and classify
for idx in tqdm(happy_indices, desc="Duchenne analysis"):
    ts = frame_series[idx]["t"]
    frame_number = int(ts * fps_in)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        smile_type = solution.classify_smile_type(frame, face_mesh)
        frame_series[idx]["smile_type"] = smile_type
    else:
        frame_series[idx]["smile_type"] = "none"

# Set non-happy frames to "none"
for i, fs in enumerate(frame_series):
    if "smile_type" not in fs:
        fs["smile_type"] = "none"

cap.release()
# No need to close eye cascade detector

# Compute duchenne summary
duchenne_count = sum(1 for fs in frame_series if fs.get("smile_type") == "duchenne")
social_count   = sum(1 for fs in frame_series if fs.get("smile_type") == "social")
total_smiles   = duchenne_count + social_count
duchenne_pct   = round(duchenne_count / total_smiles * 100, 1) if total_smiles > 0 else 0

duchenne_events = []
in_duchenne = False
start_t = 0
for fs in frame_series:
    if fs.get("smile_type") == "duchenne" and not in_duchenne:
        in_duchenne = True
        start_t = fs["t"]
    elif fs.get("smile_type") != "duchenne" and in_duchenne:
        in_duchenne = False
        duchenne_events.append({"start_sec": start_t, "end_sec": fs["t"]})
if in_duchenne:
    duchenne_events.append({"start_sec": start_t, "end_sec": frame_series[-1]["t"]})

print(f"\nResults:")
print(f"  Genuine (Duchenne): {duchenne_count} frames")
print(f"  Social smiles:      {social_count} frames")
print(f"  Genuine %:          {duchenne_pct}%")
print(f"  Duchenne events:    {len(duchenne_events)}")

# Update data
data["frame_series"] = frame_series
data["duchenne_smiles"] = {
    "genuine_frames": duchenne_count,
    "social_frames":  social_count,
    "genuine_pct":    duchenne_pct,
    "events":         duchenne_events,
}

# Save updated JSON
with open("emotion_timeline_output.json", "w") as f:
    json.dump(data, f, indent=2, default=float)
print("  Updated emotion_timeline_output.json")

# Regenerate HTML
solution.generate_emotion_timeline_html(
    frame_series, data["micro_expressions"], data["transitions"], data, solution.REPORT_HTML_OUT
)
print("Done!")
