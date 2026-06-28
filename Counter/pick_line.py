"""
pick_line.py
------------
Pulls the first frame from a video, lets you click 2 points to define
the entrance/exit line, and saves the coordinates + frame size to a
JSON file so the people-counter script can reuse them.

Usage:
    python pick_line.py path/to/video.mp4
    python pick_line.py path/to/video.mp4 --resize 1028 500
    python pick_line.py path/to/video.mp4 --out line_config.json

Controls:
    Left click  -> place a point (need exactly 2)
    r           -> reset points and start over
    s           -> save current 2 points to the config file
    Esc / q     -> quit without saving
"""

import argparse
import json

import cv2

points = []  # clicked points, in display-frame coordinates


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")


def draw_overlay(frame):
    display = frame.copy()
    for pt in points:
        cv2.circle(display, pt, 6, (0, 0, 255), -1)
    if len(points) == 2:
        cv2.line(display, points[0], points[1], (0, 255, 0), 2)
    cv2.putText(
        display,
        "Click 2 points for the line | r=reset s=save q=quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return display


def main():
    parser = argparse.ArgumentParser(description="Pick entrance/exit line coordinates from a video's first frame.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize the frame before picking points, e.g. --resize 1028 500. "
             "Use the SAME size in the counter script.",
    )
    parser.add_argument(
        "--out",
        default="line_config.json",
        help="Output JSON path (default: line_config.json)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read a frame from: {args.video}")
        return

    if args.resize:
        frame = cv2.resize(frame, (args.resize[0], args.resize[1]))

    height, width = frame.shape[:2]

    cv2.namedWindow("Pick Line")
    cv2.setMouseCallback("Pick Line", on_mouse)

    while True:
        cv2.imshow("Pick Line", draw_overlay(frame))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            print("Points reset.")

        elif key == ord("s"):
            if len(points) != 2:
                print(f"Need exactly 2 points, currently have {len(points)}.")
                continue
            config = {
                "frame_width": width,
                "frame_height": height,
                "line": [list(points[0]), list(points[1])],
            }
            with open(args.out, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved line config to: {args.out}")
            print(json.dumps(config, indent=2))
            break

        elif key == 27 or key == ord("q"):  # Esc or q
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
