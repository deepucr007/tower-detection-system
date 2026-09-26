import sys
import cv2
from pathlib import Path

from quality import check_image_quality
from detector import TowerDetector, calculate_average_confidence, get_class_counts
from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# Determine image path from arguments or default
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    # Check default test images
    candidates = ["test.JPG", "test.jpg", "sample_images/good_supporting.jpg", "sample_images/good_monopole.jpg"]
    IMAGE_PATH = next((c for c in candidates if Path(c).exists()), "test.JPG")

print("=" * 60)
print("AI TOWER DETECTION - PIPELINE VERIFICATION")
print("=" * 60)
print(f"Target Image: {IMAGE_PATH}")

# -----------------------------
# 1. Load image
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"\n[ERROR]: Could not read image at '{IMAGE_PATH}'")
    sys.exit(1)

h, w, c = image.shape
print(f"Dimensions: {w}x{h} px | Channels: {c}")

# -----------------------------
# 2. Check image quality
# -----------------------------
print("\n" + "-" * 40)
print("STEP 1: PRE-FLIGHT IMAGE QUALITY ASSESSMENT")
print("-" * 40)

quality = check_image_quality(image)

print(f"Blur score (Laplacian Var): {quality['blur_score']:.2f} (Threshold: >={quality['blur_threshold']})")
print(f"Mean luminance brightness : {quality['mean_brightness']:.2f} (Safe range: {quality['underexposed_mean_thresh']}-{quality['overexposed_mean_thresh']})")
print(f"Underexposed pixel ratio  : {quality['underexposed_ratio']:.2%}")
print(f"Overexposed pixel ratio   : {quality['overexposed_ratio']:.2%}")

# -----------------------------
# 3. Reject bad image
# -----------------------------
if not quality["acceptable"]:
    print("\n" + "=" * 40)
    print("❌ IMAGE REJECTED: UNSUITABLE QUALITY")
    print("=" * 40)
    print("Failure Reasons:")
    for reason in quality["reasons"]:
        print(f"  • {reason}")
    print("\n[SECURITY GUARANTEE]: YOLO detection was NOT executed to prevent spurious inference.")
    sys.exit(0)

# -----------------------------
# 4. Run YOLO detector
# -----------------------------
print("\n" + "=" * 40)
print("✅ IMAGE QUALITY ACCEPTABLE")
print("=" * 40)
print("STEP 2: RUNNING YOLO TOWER LOCALIZATION...")

detector = TowerDetector(MODEL_PATH)
result, detections = detector.predict(image, confidence=CONFIDENCE_THRESHOLD)

# -----------------------------
# 5. Display detections
# -----------------------------
print("\n" + "-" * 40)
print(f"STEP 3: DETECTIONS SUMMARY ({len(detections)} Found)")
print("-" * 40)

if not detections:
    print("No telecommunication towers detected above confidence threshold.")
else:
    for i, detection in enumerate(detections, start=1):
        box = detection["bbox"]
        print(
            f"[{i}] {detection['class_name'].upper()}\n"
            f"    Confidence : {detection['confidence']:.2%}\n"
            f"    BoundingBox: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )

# -----------------------------
# 6. Average confidence & Class Counts
# (Inference Challenge Requirement)
# -----------------------------
average_confidence = calculate_average_confidence(detections)
class_counts = get_class_counts(detections)

print("\n" + "-" * 40)
print("STEP 4: AVERAGE CONFIDENCE PER DETECTED CLASS")
print("-" * 40)

if not average_confidence:
    print("N/A (No detections registered)")
else:
    for class_name, confidence in average_confidence.items():
        cnt = class_counts.get(class_name, 0)
        print(f"• {class_name.title():<18}: {confidence:.2%} (Count: {cnt})")

print("\n" + "=" * 60)
print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
print("=" * 60)