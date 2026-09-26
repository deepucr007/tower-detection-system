import cv2
from quality import check_image_quality

IMAGE_PATH = "test.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load image.")
    print("Make sure test.jpg exists inside the project folder.")
    exit()

result = check_image_quality(image)

print()
print("================================")
print("IMAGE QUALITY ANALYSIS")
print("================================")

print(f"Blur score: {result['blur_score']:.2f}")
print(f"Mean brightness: {result['mean_brightness']:.2f}")
print(f"Underexposed pixels: {result['underexposed_ratio'] * 100:.2f}%")
print(f"Overexposed pixels: {result['overexposed_ratio'] * 100:.2f}%")
print()

if result["acceptable"]:
    print("RESULT: ACCEPTED")
else:
    print("RESULT: REJECTED")
    print()
    print("Reasons:")
    for reason in result["reasons"]:
        print(f"- {reason}")