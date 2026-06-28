import cv2, json

VIDEO_PATH = "carPark.mp4"   # ← your video

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    raise Exception("Cannot read video — check path")

clone = frame.copy()
spots, current_spot = [], []

def mouse_callback(event, x, y, flags, param):
    global current_spot, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        current_spot.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(current_spot) > 1:
            cv2.line(frame, tuple(current_spot[-2]), tuple(current_spot[-1]), (255, 0, 0), 2)
        if len(current_spot) == 4:
            cv2.line(frame, tuple(current_spot[-1]), tuple(current_spot[0]), (255, 0, 0), 2)
            spots.append({"points": current_spot.copy()})
            print(f"Spot #{len(spots)} saved")
            current_spot = []

cv2.namedWindow("Draw Parking Slots")
cv2.setMouseCallback("Draw Parking Slots", mouse_callback)
print("Click 4 corners per spot | S = save | R = reset | Q = quit")

while True:
    cv2.imshow("Draw Parking Slots", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open("bounding_boxes.json", "w") as f:
            json.dump(spots, f, indent=4)
        print(f"Saved {len(spots)} spots to bounding_boxes.json")
    elif key == ord("r"):
        frame = clone.copy()
        spots, current_spot = [], []
        print("Reset")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()