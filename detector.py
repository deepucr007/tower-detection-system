from ultralytics import YOLO
import numpy as np

class TowerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image, confidence=0.25):
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        result = results[0]
        detections = []

        if result.boxes is None:
            return result, detections

        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i].item())
            score = float(result.boxes.conf[i].item())
            bbox = result.boxes.xyxy[i].cpu().numpy().tolist()
            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": score,
                "bbox": bbox
            })
        return result, detections

def calculate_average_confidence(detections):
    grouped = {}
    for d in detections:
        grouped.setdefault(d["class_name"], []).append(d["confidence"])
    return {name: float(np.mean(scores)) for name, scores in grouped.items()}
