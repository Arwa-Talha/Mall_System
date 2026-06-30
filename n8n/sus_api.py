"""
suspicious_activity.py
───────────────────────
Runs the trained Video Swin Transformer violence-detection model on a video
feed, 8 frames at a time, and pushes an ALERT to the Unified API Gateway
when violence is detected for several consecutive windows in a row.

This is an alert-only pipeline: unlike crowd/parking, "Normal" results are
never sent to the gateway — only confirmed violence alerts are, to keep
this a high-signal security channel rather than a constant status feed.
"""

import cv2
import torch
import torch.nn as nn
import requests
from collections import deque
from torchvision import transforms
from torchvision.models.video import swin3d_t

# ── Paths (hardcoded, same folder as the rest of the pipeline) ──
VIDEO_PATH   = r"D:\NTI\GRAD\Parking_m\files\crowd\test4.mp4"
WEIGHTS_PATH = r"D:\NTI\GRAD\Parking_m\files\crowd\best_mall_violence_transformer.pth"
OUTPUT_PATH  = r"D:\NTI\GRAD\Parking_m\files\crowd\suspicious_result.mp4"

# Gateway URL — same FastAPI server used by crowd and parking.
API_URL = "http://127.0.0.1:8000/api/v1/events"

IMG_SIZE = 224
CLIP_LEN = 8  # frames per inference window, matches training

# How many consecutive violent windows must occur before the first alert
# fires, and how often (in windows) it re-alerts while violence continues.
# Both set to 8 per spec: ~8 windows of 8 frames each before the first
# alert, then again every 8 windows for as long as violence persists.
CONSECUTIVE_WINDOWS_TO_ALERT = 8
REALERT_EVERY_WINDOWS        = 8

# A single misclassified window shouldn't fully reset an ongoing violent
# streak. Allow up to this many consecutive Normal windows as "noise"
# before treating the streak as actually over.
NORMAL_TOLERANCE_WINDOWS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_violence_model(weights_path):
    """Loads the Swin3D-T backbone with a 2-class head, matching training."""
    model = swin3d_t()
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def send_violence_alert(label, confidence, consecutive_windows):
    """Push a suspicious-activity-shaped alert to the gateway. Matches the
    gateway's SuspiciousData model exactly: label, confidence,
    consecutive_windows. Only called for confirmed violence — there is no
    'normal' event sent to the gateway by design.
    """
    payload = {
        "source": "suspicious_activity_monitor",
        "event_type": "violence_alert",
        "data": {
            "kind": "suspicious",          # ← add this line
            "label": label,
            "confidence": confidence,
            "consecutive_windows": consecutive_windows,
        },
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"      [ALERT SENT] {label} confidence={confidence:.2f} "
                  f"consecutive_windows={consecutive_windows} -> gateway OK")
        else:
            print(f"      [ALERT REJECTED] status={response.status_code} body={response.text}")
    except Exception as e:
        print(f"      [ALERT SEND FAILED] {e}")


print("[1/4] Loading violence-detection model...")
model = load_violence_model(WEIGHTS_PATH)
print("      Model ready.")

print("[2/4] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open '{VIDEO_PATH}'.")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"      {width}x{height}  {fps}fps  {total} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_queue = deque(maxlen=CLIP_LEN)

# ── Consecutive-window tracking for the alert ──
consecutive_violent_windows = 0   # counts violent windows; tolerates brief Normal blips (see below)
normal_streak = 0                 # counts consecutive Normal windows; resets violent streak once it exceeds tolerance
window_idx = 0                    # counts completed 8-frame windows (not raw frames)
alert_active = False              # True once we've fired the first alert for this streak

print("[3/4] Processing frames & monitoring for sustained violence...")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    display_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_frame = inference_transforms(rgb_frame)
    frame_queue.append(transformed_frame)

    if len(frame_queue) == CLIP_LEN:
        window_idx += 1

        video_tensor = torch.stack(list(frame_queue))
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, 1)

        label_idx = pred.item()
        conf_val = confidence.item()

        if label_idx == 1:
            # ── Violent window ──
            consecutive_violent_windows += 1
            normal_streak = 0  # any violent window clears the "blip" counter
            text = f"VIOLENCE DETECTED ({conf_val*100:.1f}%)"
            color = (0, 0, 255)
            cv2.putText(display_frame, "SYSTEM ALERT: SECURITY DISPATCHED",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if consecutive_violent_windows >= CONSECUTIVE_WINDOWS_TO_ALERT:
                # how far past the initial trigger threshold are we, in windows?
                windows_past_trigger = consecutive_violent_windows - CONSECUTIVE_WINDOWS_TO_ALERT
                if not alert_active:
                    # first alert for this streak
                    send_violence_alert("VIOLENCE", conf_val, consecutive_violent_windows)
                    alert_active = True
                elif windows_past_trigger % REALERT_EVERY_WINDOWS == 0:
                    # re-alert at the configured cadence while violence continues
                    send_violence_alert("VIOLENCE", conf_val, consecutive_violent_windows)
        else:
            # ── Normal window ──
            # Tolerate a short run of Normal windows (likely a single
            # misclassified frame mid-incident) without fully resetting an
            # ongoing violent streak. Only reset for real once the Normal
            # run exceeds NORMAL_TOLERANCE_WINDOWS.
            normal_streak += 1
            text = f"Normal / Crowd ({conf_val*100:.1f}%)"
            color = (0, 255, 0)

            if normal_streak > NORMAL_TOLERANCE_WINDOWS:
                consecutive_violent_windows = 0
                alert_active = False

        cv2.rectangle(display_frame, (10, 10), (500, 50), (0, 0, 0), -1)
        cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # periodic progress print, every 30 windows (~ same cadence style as crowd)
        if window_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"      Window {window_idx}  Frame {frame_idx}/{total} ({pct:.0f}%) "
                  f"consecutive_violent={consecutive_violent_windows}")

    out.write(display_frame)

cap.release()
out.release()
print(f"[4/4] Done! Saved -> {OUTPUT_PATH}")
print(f"      Processed {frame_idx} frames, {window_idx} windows.")