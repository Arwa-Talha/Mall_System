"""
people_counter.py
------------------
Counts people entering and leaving by detecting when a moving blob
crosses a single line (picked beforehand with pick_line.py).

No persistent object IDs are used. Each frame, every detected blob is
matched to the closest blob from the PREVIOUS frame only (so we can tell
which side of the line it was a moment ago). That match is discarded
every frame -- there is no long-term identity, no growing dictionary,
no ID numbers shown on screen.

Three changes from a naive "box center crosses the line" approach, made
to avoid false counts from someone standing still or lingering near
the line:

1. We track the BOTTOM-CENTER of the bounding box (approx. feet
   position) instead of the box's full center. The box's height jitters
   frame to frame as background subtraction picks up more/less of a
   person's arms/head -- that jitter moves the box CENTER up and down
   even when the person hasn't taken a step, which can flip it across
   the line spuriously. The bottom edge is more stable, though not
   perfectly so (see point 3).

2. We use a dead band (a buffer distance on each side of the line, in
   pixels) instead of a single hard line. A crossing is only counted
   when a point goes from clearly beyond one edge of the band to
   clearly beyond the other edge. Hovering inside the band no longer
   triggers repeated counts.

3. A side flip must hold for several CONSECUTIVE frames (--confirm-frames)
   before it's counted as a real crossing. This matters because MOG2
   background subtraction can lose someone's legs for a single frame
   when they pause near the line (slow/static motion blends into the
   adapting background, or dark clothing blends into a dark floor) --
   the detected box shrinks, its bottom edge jumps far past the band in
   one frame, and then recovers the next frame. A one-frame dead-band
   jump like that is a detection glitch, not a real step through the
   line, so we require the new side to be confirmed for multiple frames
   in a row first. Genuine walk-throughs easily hold for several frames;
   single-frame glitches don't.

Direction convention:
    Take the line from point A (line[0]) to point B (line[1]).
    Walking along the screen, everything to the LEFT of A->B is "side 1",
    everything to the RIGHT is "side 2".
    side 1 -> side 2 crossing = ENTER
    side 2 -> side 1 crossing = EXIT
    (If it's backwards for your video, just swap the two points when you
    pick the line, or flip ENTER/EXIT below.)

Usage:
    python people_counter.py path/to/video.mp4 --config line_config.json
"""

import argparse
import json

import cv2
import numpy as np


def signed_distance(point, line_a, line_b):
    """
    Signed perpendicular distance (in pixels) from point to the infinite
    line through line_a -> line_b. Positive on one side, negative on the
    other. Magnitude is the actual pixel distance (not just a cross
    product), so it can be compared against a pixel-based dead band.
    """
    ax, ay = line_a
    bx, by = line_b
    px, py = point

    line_len = np.hypot(bx - ax, by - ay)
    if line_len == 0:
        return 0.0

    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return cross / line_len


def match_to_previous(current_points, previous_points, max_dist=50):
    """
    For each current point, find the index of the closest previous point
    within max_dist. Purely for one-frame-back comparison -- nothing
    persists beyond this single match.

    Returns a list of (current_point, matched_previous_index_or_None).
    Using indices (not point values) avoids ambiguity if two blobs ever
    land on the exact same coordinate.
    """
    matches = []
    used_prev = set()

    for c in current_points:
        best_idx = None
        best_dist = max_dist
        for i, p in enumerate(previous_points):
            if i in used_prev:
                continue
            dist = np.hypot(c[0] - p[0], c[1] - p[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_prev.add(best_idx)
            matches.append((c, best_idx))
        else:
            matches.append((c, None))

    return matches


def main():
    parser = argparse.ArgumentParser(description="Count people entering/leaving across a line.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--config", default="line_config.json", help="Path to line config JSON from pick_line.py")
    parser.add_argument("--min-area", type=int, default=1000, help="Minimum contour area to count as a person")
    parser.add_argument("--max-match-dist", type=int, default=100, help="Max pixel distance to match a blob to last frame")
    parser.add_argument("--band", type=int, default=10, help="Dead-band half-width in pixels around the line (reduces jitter false counts)")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Consecutive frames a side flip must hold before counting as a real crossing (filters one-frame detection glitches)")
    parser.add_argument("--save", default=None, help="Path to save the annotated video, e.g. output.mp4")
    parser.add_argument("--no-display", action="store_true", help="Don't open a preview window (useful when only saving)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    line_a, line_b = config["line"]
    frame_w, frame_h = config["frame_width"], config["frame_height"]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=250)
    video_capture = cv2.VideoCapture(args.video)

    writer = None
    if args.save:
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(f"Could not open writer for: {args.save}")
            writer = None

    entered = 0
    left = 0
    # Each tracked entry carries:
    #   point            - current bottom-center point
    #   confirmed_side    - the side (+1/-1) we've actually committed to, used as the
    #                       baseline for detecting a real crossing
    #   pending_side      - a side reading that differs from confirmed_side and is
    #                       being watched to see if it holds for multiple frames
    #   pending_streak    - how many consecutive frames pending_side has held
    tracked = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_w, frame_h))

        mask = bg_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 245, 255, cv2.THRESH_BINARY)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_points = []
        for cnt in contours:
            if cv2.contourArea(cnt) < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Bottom-center of the box (approx. feet position) -- much
            # more stable than the full box center, which jitters as
            # detected box height changes frame to frame.
            fx, fy = x + w // 2, y + h // 2
            current_points.append((fx, fy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.circle(frame, (fx, fy), 4, (255, 0, 255), -1)

        previous_points = [t["point"] for t in tracked]
        matches = match_to_previous(current_points, previous_points, args.max_match_dist)

        new_tracked = []
        for current_pt, previous_idx in matches:
            dist = signed_distance(current_pt, line_a, line_b)

            if dist > args.band:
                reading = 1
            elif dist < -args.band:
                reading = -1
            else:
                reading = None  # inside the dead band this frame

            prev_state = tracked[previous_idx] if previous_idx is not None else None

            if prev_state is None:
                # New blob, no history -- just start tracking it.
                new_tracked.append({
                    "point": current_pt,
                    "confirmed_side": reading,
                    "pending_side": None,
                    "pending_streak": 0,
                })
                continue

            confirmed_side = prev_state["confirmed_side"]
            pending_side = prev_state["pending_side"]
            pending_streak = prev_state["pending_streak"]

            if reading is None:
                # Inside the dead band this frame -- don't change anything,
                # just carry the existing state forward unchanged.
                pass
            elif confirmed_side is None:
                # We didn't have a confirmed side yet (e.g. first reading
                # was ambiguous) -- accept this reading immediately, no
                # crossing to count since there was nothing to cross from.
                confirmed_side = reading
                pending_side = None
                pending_streak = 0
            elif reading == confirmed_side:
                # Matches what we already believe -- reset any pending flip.
                pending_side = None
                pending_streak = 0
            else:
                # Reading disagrees with our confirmed side.
                if reading == pending_side:
                    pending_streak += 1
                else:
                    pending_side = reading
                    pending_streak = 1

                if pending_streak >= args.confirm_frames:
                    # Held for enough consecutive frames -- this is a real crossing.
                    if confirmed_side == -1 and reading == 1:
                        entered += 1
                    elif confirmed_side == 1 and reading == -1:
                        left += 1
                    confirmed_side = reading
                    pending_side = None
                    pending_streak = 0

            new_tracked.append({
                "point": current_pt,
                "confirmed_side": confirmed_side,
                "pending_side": pending_side,
                "pending_streak": pending_streak,
            })

        tracked = new_tracked

        inside = entered - left

        # Draw the line, its dead band, and the 3 counters
        ax, ay = line_a
        bx, by = line_b
        dx, dy = bx - ax, by - ay
        length = np.hypot(dx, dy) or 1
        nx, ny = -dy / length, dx / length  # unit normal to the line

        band_a1 = (int(ax + nx * args.band), int(ay + ny * args.band))
        band_b1 = (int(bx + nx * args.band), int(by + ny * args.band))
        band_a2 = (int(ax - nx * args.band), int(ay - ny * args.band))
        band_b2 = (int(bx - nx * args.band), int(by - ny * args.band))

        cv2.line(frame, band_a1, band_b1, (0, 165, 255), 1)
        cv2.line(frame, band_a2, band_b2, (0, 165, 255), 1)
        cv2.line(frame, tuple(line_a), tuple(line_b), (0, 0, 255), 2)
        cv2.putText(frame, f"ENTERED: {entered}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"LEFT: {left}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"INSIDE: {inside}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("People Counter", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # Esc or q
                break

    video_capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Final -> entered: {entered}, left: {left}, inside: {inside}")


if __name__ == "__main__":
    main()