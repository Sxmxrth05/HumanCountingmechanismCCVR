from ultralytics import YOLO

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
