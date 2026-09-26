import cv2
import numpy as np

BLUR_THRESHOLD = 80.0
UNDEREXPOSED_MEAN = 45.0
OVEREXPOSED_MEAN = 210.0
UNDEREXPOSED_RATIO = 0.25
OVEREXPOSED_RATIO = 0.25


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculates image sharpness using the variance of the Laplacian filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_exposure(image: np.ndarray) -> dict:
    """Calculates luminance statistics to detect underexposure and overexposure."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_pixels = max(1, gray.size)
    mean_val = float(np.mean(gray))
    under_ratio = float(np.sum(gray <= 5) / total_pixels)
    over_ratio = float(np.sum(gray >= 250) / total_pixels)

    return {
        "mean_brightness": mean_val,
        "underexposed_ratio": under_ratio,
        "overexposed_ratio": over_ratio
    }


def check_image_quality(image: np.ndarray) -> dict:
    """
    Automated pre-flight image quality filtering mechanism.
    Identifies unsuitable input images before object detection is performed:
      - Blurred images
      - Underexposed images
      - Overexposed images
    Only acceptable-quality images proceed to YOLO inference.
    """
    reasons = []
    blur_score = calculate_blur_score(image)
    exposure = calculate_exposure(image)

    is_blurry = blur_score < BLUR_THRESHOLD
    is_underexposed = (
        exposure["mean_brightness"] < UNDEREXPOSED_MEAN
        or exposure["underexposed_ratio"] > UNDEREXPOSED_RATIO
    )
    is_overexposed = (
        exposure["mean_brightness"] > OVEREXPOSED_MEAN
        or exposure["overexposed_ratio"] > OVEREXPOSED_RATIO
    )

    if is_blurry:
        reasons.append(f"Image is blurry (blur score: {blur_score:.2f} < threshold: {BLUR_THRESHOLD:.1f})")
    if is_underexposed:
        reasons.append(f"Image is underexposed (mean brightness: {exposure['mean_brightness']:.2f} < threshold: {UNDEREXPOSED_MEAN:.1f})")
    if is_overexposed:
        reasons.append(f"Image is overexposed (mean brightness: {exposure['mean_brightness']:.2f} > threshold: {OVEREXPOSED_MEAN:.1f})")

    acceptable = len(reasons) == 0

    return {
        "acceptable": acceptable,
        "blur_score": blur_score,
        "blur_threshold": BLUR_THRESHOLD,
        "is_blurry": is_blurry,
        "mean_brightness": exposure["mean_brightness"],
        "underexposed_mean_thresh": UNDEREXPOSED_MEAN,
        "overexposed_mean_thresh": OVEREXPOSED_MEAN,
        "underexposed_ratio": exposure["underexposed_ratio"],
        "overexposed_ratio": exposure["overexposed_ratio"],
        "is_underexposed": is_underexposed,
        "is_overexposed": is_overexposed,
        "reasons": reasons,
        "metrics": {
            "blur": {
                "name": "Sharpness (Laplacian Var)",
                "passed": not is_blurry,
                "value": blur_score,
                "threshold": BLUR_THRESHOLD,
                "unit": "variance"
            },
            "underexposure": {
                "name": "Shadow Floor (Brightness)",
                "passed": not is_underexposed,
                "value": exposure["mean_brightness"],
                "threshold": UNDEREXPOSED_MEAN,
                "unit": "0-255"
            },
            "overexposure": {
                "name": "Highlight Ceiling (Brightness)",
                "passed": not is_overexposed,
                "value": exposure["mean_brightness"],
                "threshold": OVEREXPOSED_MEAN,
                "unit": "0-255"
            }
        }
    }
