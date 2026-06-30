"""
yolo_bridge.py
──────────────
Runs the YOLO parking model and pushes occupancy state to the Unified API
Gateway every N frames, shaped to match the gateway's ParkingData model.

Run:
  python yolo_bridge.py
"""

import json, time
import cv2, requests
from ultralytics import solutions

# ── Paths (hardcoded — same folder as the crowd pipeline's video) ──
VIDEO_PATH  = r"D:\NTI\GRAD\Parking_m\files\crowd\carPark.mp4"
MODEL_PATH  = r"D:\NTI\GRAD\Parking_m\files\crowd\bestYOLOv8n.pt"
JSON_PATH   = r"D:\NTI\GRAD\Parking_m\files\crowd\bounding_boxes.json"
OUTPUT_PATH = r"D:\NTI\GRAD\Parking_m\files\crowd\parking_output.mp4"

# ── Detection settings ──
CLASSES = [3, 4, 5]   # class IDs to track (e.g. car/van/truck — adjust to your model's labels)
CONF    = 0.15
IOU     = 0.45

# Gateway URL — same FastAPI server used by the crowd pipeline.
SERVER     = "http://127.0.0.1:8000"
EVENTS_URL = f"{SERVER}/api/v1/events"
PUSH_EVERY = 5


def load_spots(path):
    with open(path) as f:
        return json.load(f)


def compute_occupied(pm, spots):
    """
    pm.boxes is a torch.Tensor of shape (N, 6) in original resolution.
    Columns: x1, y1, x2, y2, conf, cls
    Compare each box against each spot polygon.
    """
    try:
        if pm.boxes is None or len(pm.boxes) == 0:
            return set()

        boxes = pm.boxes.cpu().numpy()
        occupied = set()

        for box in boxes:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            for idx, spot in enumerate(spots):
                pts = spot["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                sx1, sy1 = min(xs), min(ys)
                sx2, sy2 = max(xs), max(ys)

                ix1 = max(x1, sx1); iy1 = max(y1, sy1)
                ix2 = min(x2, sx2); iy2 = min(y2, sy2)

                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    spot_area = (sx2 - sx1) * (sy2 - sy1)
                    if spot_area > 0 and inter / spot_area >= 0.15:
                        occupied.add(idx)

        return occupied

    except Exception as e:
        print(f"  compute_occupied error: {e}")
        return set()


def push(occupied, total_spots, previous_occupied):
    """Builds a gateway-shaped ParkingData payload and POSTs it.

    `previous_occupied` is the occupied set from the last push, used to
    compute the newly_occupied / newly_vacated deltas the gateway expects.
    """
    newly_occupied = list(occupied - previous_occupied)
    newly_vacated  = list(previous_occupied - occupied)

    payload = {
        "source": "parking_lot_1",
        "event_type": "state_change",
        "data": {
            "occupied_spots": list(occupied),
            "occupied_count": len(occupied),
            "total_spots": total_spots,
            "free_count": total_spots - len(occupied),
            "newly_occupied": newly_occupied,
            "newly_vacated": newly_vacated,
        },
    }

    try:
        r = requests.post(EVENTS_URL, json=payload, timeout=2)
        if r.status_code == 200:
            print(f"  -> pushed {len(occupied)} occupied, status {r.status_code}")
        else:
            # surface validation errors (e.g. 422) instead of treating them
            # like a successful push
            print(f"  -> REJECTED status={r.status_code} body={r.text}")
    except Exception as e:
        print(f"  -> PUSH FAILED: {e}")


def run():
    spots = load_spots(JSON_PATH)
    total_spots = len(spots)
    print(f"Loaded {total_spots} spots from {JSON_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open: {VIDEO_PATH}"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    print(f"Video: {orig_w}x{orig_h} @ {fps}fps")

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    ) if OUTPUT_PATH else None

    pm = solutions.ParkingManagement(
        model=MODEL_PATH,
        json_file=JSON_PATH,
        classes=CLASSES if CLASSES else None,
        conf=CONF,
        iou=IOU,
    )

    n = 0
    t0 = time.time()
    previous_occupied = set()  # tracks state across pushes for delta computation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended — restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = pm(frame)

        if n % PUSH_EVERY == 0:
            occupied = compute_occupied(pm, spots)

            if n == 10:
                print(f"\n── DEBUG frame 10 ──")
                print(f"pm.boxes shape : {pm.boxes.shape if pm.boxes is not None else 'None'}")
                if pm.boxes is not None and len(pm.boxes) > 0:
                    b = pm.boxes[0].cpu().numpy()
                    print(f"box[0]         : x1={b[0]:.0f} y1={b[1]:.0f} x2={b[2]:.0f} y2={b[3]:.0f}")
                print(f"spot[0] pts    : {spots[0]['points']}")
                print(f"occupied IDs   : {occupied}")
                print(f"filled_slots   : {results.filled_slots}")
                print(f"────────────────\n")

            push(occupied, total_spots, previous_occupied)
            previous_occupied = occupied

        if writer and hasattr(results, 'plot_im'):
            writer.write(results.plot_im)

        n += 1
        if n % 100 == 0:
            elapsed = time.time() - t0
            print(f"Frame {n}  |  {n/elapsed:.1f} fps")

    cap.release()
    if writer:
        writer.release()
    print(f"Done — {n} frames")


if __name__ == "__main__":
    run()