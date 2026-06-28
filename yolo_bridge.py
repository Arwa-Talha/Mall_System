"""
yolo_bridge.py
──────────────
Runs your YOLO parking model and pushes occupied spot IDs to Flask every N frames.

Run:
  python yolo_bridge.py --video carPark.mp4 --model bestYOLOv8n.pt --classes 3 4 5 --conf 0.15
"""

import argparse, json, time
import cv2, requests
from ultralytics import solutions

SERVER     = "http://127.0.0.1:5000"
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

        boxes = pm.boxes.cpu().numpy()  # convert to numpy
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

def push(occupied):
    try:
        r = requests.post(f"{SERVER}/api/update",
                          json={"occupied": list(occupied)}, timeout=1)
        print(f"  → pushed {len(occupied)} occupied, status {r.status_code}")
    except Exception as e:
        print(f"  → PUSH FAILED: {e}")

def run(video, model_path, json_path, output, classes, conf, iou):
    spots = load_spots(json_path)
    print(f"Loaded {len(spots)} spots from {json_path}")

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"Cannot open: {video}"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    print(f"Video: {orig_w}x{orig_h} @ {fps}fps")

    writer = cv2.VideoWriter(
        output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    ) if output else None

    pm = solutions.ParkingManagement(
        model=model_path,
        json_file=json_path,
        classes=classes if classes else None,
        conf=conf,
        iou=iou,
    )

    n = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended — restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = pm(frame)

        if n % PUSH_EVERY == 0:
            occupied = compute_occupied(pm, spots)

            # ── one-time debug on frame 10 ──
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

            push(occupied)

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
    p = argparse.ArgumentParser()
    p.add_argument("--video",   default="carPark.mp4")
    p.add_argument("--model",   default="bestYOLOv8n.pt")
    p.add_argument("--json",    default="bounding_boxes.json")
    p.add_argument("--output",  default="parking_output.mp4")
    p.add_argument("--classes", nargs="+", type=int, default=None)
    p.add_argument("--conf",    type=float, default=0.15)
    p.add_argument("--iou",     type=float, default=0.45)
    a = p.parse_args()
    run(a.video, a.model, a.json, a.output, a.classes, a.conf, a.iou)
