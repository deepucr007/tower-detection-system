import cv2
from pathlib import Path
from quality import check_image_quality
from detector import TowerDetector, calculate_average_confidence

detector = TowerDetector('models/best.pt')
test_dir = Path('sample_images/unseen_test')

for img_p in sorted(test_dir.glob('*.jpg')):
    img = cv2.imread(str(img_p))
    q = check_image_quality(img)
    print("=" * 60)
    print(f"FILE: {img_p.name} ({img.shape[1]}x{img.shape[0]})")
    print(f"Quality Acceptable : {q['acceptable']}")
    print(f"Blur Score         : {q['blur_score']:.2f} (Threshold: {q['blur_threshold']})")
    print(f"Mean Brightness    : {q['mean_brightness']:.2f}")
    print(f"Underexposed Ratio : {q['underexposed_ratio']:.2%}")
    print(f"Overexposed Ratio  : {q['overexposed_ratio']:.2%}")
    print(f"Rejection Reasons  : {q['reasons']}")
    
    # Also test detector directly on all images to see what the model predicts if passed!
    res, dets = detector.predict(img, confidence=0.20)
    print(f"Model Direct Detections (conf>=0.20): {len(dets)} detected")
    for d in dets:
        print(f"   -> {d['class_name']} ({d['confidence']:.2%}) bbox: {[round(x,1) for x in d['bbox']]}")
    print()
