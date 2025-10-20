import cv2
from ultralytics import YOLO
from person_tracker import Tracker  # Your DeepSORT wrapper
from counter import Counter         # Updated occupancy counter
from utils import draw_tracks       # Updated draw_tracks with colors

# ---------------- Detector ----------------
class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # 'person' class
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, conf])
        return detections

# ---------------- Main ----------------
def main(source=0):
    detector = Detector(model_path="yolov8n.pt", conf_threshold=0.5)
    tracker = Tracker(max_age=30)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ Error opening video stream or file")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        return

    # Vertical line in the middle for left/right occupancy
    counter = Counter(line_position_x=frame.shape[1] // 2)
    print("✅ Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect humans
        raw_detections = detector.detect(frame)
        # 2. Convert to DeepSORT format: ([x1,y1,x2,y2], conf, class_id)
        detections = [([x1, y1, x2, y2], conf, "person")
                      for x1, y1, x2, y2, conf in raw_detections]

        # 3. Update tracker
        tracks = tracker.update(detections, frame)

        # 4. Update occupancy counter (left/right)
        counter.update(tracks)

        # 5. Draw tracks and counter line
        draw_tracks(frame, tracks)
        counter.draw(frame)

        # 6. Show frame
        cv2.imshow("YOLOv8 + DeepSORT Human Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)  # 0 for webcam, or replace with video path
