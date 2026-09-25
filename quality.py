import cv2
import numpy as np

BLUR_THRESHOLD = 80.0
UNDEREXPOSED_MEAN = 45
OVEREXPOSED_MEAN = 210
UNDEREXPOSED_RATIO = 0.25
OVEREXPOSED_RATIO = 0.25

def calculate_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    return {
        "mean_brightness": float(np.mean(gray)),
        "underexposed_ratio": float(np.sum(gray <= 5) / total_pixels),
        "overexposed_ratio": float(np.sum(gray >= 250) / total_pixels)
    }

def check_image_quality(image):
    reasons = []
    blur_score = calculate_blur_score(image)
    exposure = calculate_exposure(image)

    if blur_score < BLUR_THRESHOLD:
        reasons.append(f"Image is blurry (blur score: {blur_score:.2f})")
    if exposure["mean_brightness"] < UNDEREXPOSED_MEAN or exposure["underexposed_ratio"] > UNDEREXPOSED_RATIO:
        reasons.append(f"Image is underexposed (brightness: {exposure['mean_brightness']:.2f})")
    if exposure["mean_brightness"] > OVEREXPOSED_MEAN or exposure["overexposed_ratio"] > OVEREXPOSED_RATIO:
        reasons.append(f"Image is overexposed (brightness: {exposure['mean_brightness']:.2f})")

    return {
        "acceptable": len(reasons) == 0,
        "blur_score": blur_score,
        "mean_brightness": exposure["mean_brightness"],
        "underexposed_ratio": exposure["underexposed_ratio"],
        "overexposed_ratio": exposure["overexposed_ratio"],
        "reasons": reasons
    }
