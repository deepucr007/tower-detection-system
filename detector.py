from ultralytics import YOLO
import numpy as np
import time

class TowerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def predict(self, image, confidence=0.25):
        t0 = time.time()
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        latency_ms = (time.time() - t0) * 1000.0
        result = results[0]
        detections = []

        if result.boxes is not None:
            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i].item())
                score = float(result.boxes.conf[i].item())
                bbox = result.boxes.xyxy[i].cpu().numpy().tolist()
                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": score,
                    "bbox": bbox,
                    "box_w": float(bbox[2] - bbox[0]),
                    "box_h": float(bbox[3] - bbox[1])
                })
        result.inference_time_ms = latency_ms
        return result, detections


def calculate_average_confidence(detections):
    """Calculates the average confidence score for each detected class."""
    grouped = {}
    for d in detections:
        grouped.setdefault(d["class_name"], []).append(d["confidence"])
    return {name: float(np.mean(scores)) for name, scores in grouped.items()}


def get_class_counts(detections):
    """Calculates the count of detected objects per class."""
    counts = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts
