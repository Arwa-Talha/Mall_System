import json
import cv2
import numpy as np

# ── Paths (edit these) ────────────────────────────────────────────────────────
VIDEO_PATH      = "/kaggle/input/your-dataset/video.mp4"
OUTPUT_PATH     = "/kaggle/working/output.mp4"
CONFIG_PATH     = "/kaggle/input/your-dataset/line_config.json"

# ── Tuning parameters ─────────────────────────────────────────────────────────
MIN_AREA        = 1000   # minimum blob area to count as a person
MAX_MATCH_DIST  = 100    # max pixel distance to match a blob to the previous frame
BAND            = 10     # dead-band half-width in pixels around the line
CONFIRM_FRAMES  = 3      # consecutive frames a side-flip must hold to count as a crossing

# ── Load line config ──────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = json.load(f)

line_a   = config["line"][0]
line_b   = config["line"][1]
frame_w  = config["frame_width"]
frame_h  = config["frame_height"]


# ── Helper functions ──────────────────────────────────────────────────────────
def signed_distance(point, a, b):
    """Signed perpendicular distance from point to the line a->b (in pixels)."""
    ax, ay = a;  bx, by = b;  px, py = point
    length = np.hypot(bx - ax, by - ay)
    if length == 0:
        return 0.0
    return ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / length


def match_blobs(current, previous, max_dist):
    """Match each current blob to the nearest previous blob (one frame back only)."""
    matches = []
    used = set()
    for c in current:
        best_idx, best_d = None, max_dist
        for i, p in enumerate(previous):
            if i in used:
                continue
            d = np.hypot(c[0] - p[0], c[1] - p[1])
            if d < best_d:
                best_d, best_idx = d, i
        if best_idx is not None:
            used.add(best_idx)
        matches.append((c, best_idx))
    return matches


# ── Main loop ─────────────────────────────────────────────────────────────────
cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

bg_sub   = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=250)
entered  = 0
left     = 0
tracked  = []   # list of dicts: {point, confirmed_side, pending_side, pending_streak}

# Pre-compute the dead-band edge lines for drawing
ax, ay = line_a;  bx, by = line_b
dx, dy = bx - ax, by - ay
length = np.hypot(dx, dy) or 1
nx, ny = -dy / length, dx / length   # unit normal

band_a1 = (int(ax + nx * BAND), int(ay + ny * BAND))
band_b1 = (int(bx + nx * BAND), int(by + ny * BAND))
band_a2 = (int(ax - nx * BAND), int(ay - ny * BAND))
band_b2 = (int(bx - nx * BAND), int(by - ny * BAND))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_w, frame_h))

    # Background subtraction → binary mask
    mask = bg_sub.apply(frame)
    _, mask = cv2.threshold(mask, 245, 255, cv2.THRESH_BINARY)

    # Find blobs and collect their center points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        current_points.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

    # Match blobs to previous frame
    prev_points = [t["point"] for t in tracked]
    matches = match_blobs(current_points, prev_points, MAX_MATCH_DIST)

    new_tracked = []
    for current_pt, prev_idx in matches:
        dist    = signed_distance(current_pt, line_a, line_b)
        reading = 1 if dist > BAND else (-1 if dist < -BAND else None)

        prev = tracked[prev_idx] if prev_idx is not None else None

        if prev is None:
            # Brand-new blob — start tracking with no crossing history
            new_tracked.append({"point": current_pt, "confirmed_side": reading,
                                 "pending_side": None, "pending_streak": 0})
            continue

        confirmed  = prev["confirmed_side"]
        pending    = prev["pending_side"]
        streak     = prev["pending_streak"]

        if reading is None:
            pass  # inside dead band — carry state forward unchanged
        elif confirmed is None:
            confirmed = reading;  pending = None;  streak = 0
        elif reading == confirmed:
            pending = None;  streak = 0  # still on the same side
        else:
            # Possible crossing — build up a streak before committing
            if reading == pending:
                streak += 1
            else:
                pending, streak = reading, 1

            if streak >= CONFIRM_FRAMES:
                if   confirmed == -1 and reading ==  1:  entered += 1
                elif confirmed ==  1 and reading == -1:  left    += 1
                confirmed = reading;  pending = None;  streak = 0

        new_tracked.append({"point": current_pt, "confirmed_side": confirmed,
                             "pending_side": pending, "pending_streak": streak})

    tracked = new_tracked

    # Overlay: dead-band edges, main line, counters
    cv2.line(frame, band_a1, band_b1, (0, 165, 255), 1)
    cv2.line(frame, band_a2, band_b2, (0, 165, 255), 1)
    cv2.line(frame, tuple(line_a), tuple(line_b), (0, 0, 255), 2)
    cv2.putText(frame, f"ENTERED: {entered}",        (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
    cv2.putText(frame, f"LEFT:    {left}",           (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),   2)
    cv2.putText(frame, f"INSIDE:  {entered - left}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    writer.write(frame)

cap.release()
writer.release()

print(f"Done — entered: {entered}, left: {left}, inside: {entered - left}")
print(f"Saved to: {OUTPUT_PATH}")
