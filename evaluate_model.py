from ultralytics import YOLO
from pathlib import Path
import json


# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/best.pt"
DATA_YAML = "dataset/data.yaml"

OUTPUT_DIR = Path("evaluation")


# ==============================
# LOAD MODEL
# ==============================

print("=" * 60)
print("YOLO TOWER DETECTION - MODEL EVALUATION")
print("=" * 60)

print("\nLoading model...")
model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:", model.names)


# ==============================
# RUN VALIDATION
# ==============================

print("\nRunning validation...")
print("Please wait...")

metrics = model.val(
    data=DATA_YAML,
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.50,
    plots=True,
    project="evaluation",
    name="tower_model"
)


# ==============================
# EXTRACT METRICS
# ==============================

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ==============================
# PRINT OVERALL RESULTS
# ==============================

print("\n")
print("=" * 60)
print("OVERALL MODEL EVALUATION")
print("=" * 60)

print(f"Precision       : {precision:.4f} ({precision:.2%})")
print(f"Recall          : {recall:.4f} ({recall:.2%})")
print(f"F1 Score        : {f1_score:.4f} ({f1_score:.2%})")
print(f"mAP@0.50        : {map50:.4f} ({map50:.2%})")
print(f"mAP@0.50:0.95   : {map50_95:.4f} ({map50_95:.2%})")


# ==============================
# PER-CLASS RESULTS
# ==============================

print("\n")
print("=" * 60)
print("PER-CLASS EVALUATION")
print("=" * 60)

class_names = model.names

# Per-class precision, recall and mAP
class_precision = metrics.box.p
class_recall = metrics.box.r
class_ap50 = metrics.box.ap50
class_ap = metrics.box.ap

for class_id, class_name in class_names.items():

    print(f"\nClass {class_id}: {class_name}")

    if class_id < len(class_precision):
        p = float(class_precision[class_id])
        r = float(class_recall[class_id])
        ap50 = float(class_ap50[class_id])
        ap = float(class_ap[class_id])

        print(f"  Precision     : {p:.4f} ({p:.2%})")
        print(f"  Recall        : {r:.4f} ({r:.2%})")
        print(f"  AP@0.50       : {ap50:.4f} ({ap50:.2%})")
        print(f"  AP@0.50:0.95  : {ap:.4f} ({ap:.2%})")


# ==============================
# SAVE REPORT
# ==============================

OUTPUT_DIR.mkdir(exist_ok=True)

report = {
    "model": MODEL_PATH,
    "dataset": DATA_YAML,
    "image_size": 640,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.50,
    "classes": class_names,
    "overall": {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP50": map50,
        "mAP50_95": map50_95
    },
    "per_class": {}
}


for class_id, class_name in class_names.items():

    if class_id < len(class_precision):

        p = float(class_precision[class_id])
        r = float(class_recall[class_id])
        ap50 = float(class_ap50[class_id])
        ap = float(class_ap[class_id])

        report["per_class"][class_name] = {
            "precision": p,
            "recall": r,
            "AP50": ap50,
            "AP50_95": ap
        }


report_path = OUTPUT_DIR / "evaluation_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)


# ==============================
# FINISHED
# ==============================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(f"\nReport saved to:")
print(report_path)

print("\nEvaluation plots saved inside:")
print("evaluation/tower_model/")