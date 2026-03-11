import cv2
import numpy as np
from config import (
    AUTO_BRIGHTNESS_THRESHOLD,
    AUTO_NOISE_THRESHOLD,
    AUTO_BLUR_THRESHOLD
)

# ─── Load once at module level (not per call) ─────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def analyze_image(img: np.ndarray) -> list:
    """
    Analyzes an image and returns a list of recommended enhancement steps.
    Called automatically when user selects 'Auto Detect' option.
    """
    steps = []
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ─── Brightness Check ─────────────────────────────────────────────────────
    brightness = np.mean(gray)
    if brightness < AUTO_BRIGHTNESS_THRESHOLD:
        steps.append("lighting")
        print(f"  [analyzer] Dark image detected (brightness={brightness:.1f}) → lighting")

    # ─── Color Saturation Check ───────────────────────────────────────────────
    hsv            = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    if avg_saturation < 60:
        steps.append("color")
        print(f"  [analyzer] Low saturation detected (sat={avg_saturation:.1f}) → color")

    # ─── Noise Check ──────────────────────────────────────────────────────────
    blurred     = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
    if noise_score > AUTO_NOISE_THRESHOLD:
        steps.append("denoise")
        print(f"  [analyzer] Noise detected (score={noise_score:.1f}) → denoise")

    # ─── Blur / Sharpness Check ───────────────────────────────────────────────
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < AUTO_BLUR_THRESHOLD:
        steps.append("sharpen")
        print(f"  [analyzer] Blur detected (score={blur_score:.1f}) → sharpen")

    # ─── Face Detection ───────────────────────────────────────────────────────
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    if len(faces) > 0:
        steps.append("face")
        print(f"  [analyzer] {len(faces)} face(s) detected → face enhance")

    # ─── Summary ──────────────────────────────────────────────────────────────
    if not steps:
        print("  [analyzer] Image looks clean — no auto steps added")
    else:
        print(f"  [analyzer] Auto steps: {steps}")

    return steps


def get_image_stats(img: np.ndarray) -> dict:
    """
    Returns raw image stats — useful for debugging or displaying
    image info in the frontend later.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur_score  = cv2.Laplacian(gray, cv2.CV_64F).var()
    blurred     = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = np.std(gray.astype(np.float32) - blurred.astype(np.float32))

    faces = _face_cascade.detectMultiScale(
        gray, 1.1, 4, minSize=(30, 30)
    )

    return {
        "brightness":  round(float(np.mean(gray)), 2),
        "saturation":  round(float(np.mean(hsv[:, :, 1])), 2),
        "noise_score": round(float(noise_score), 2),
        "blur_score":  round(float(blur_score), 2),
        "faces_found": int(len(faces)),
        "resolution":  f"{img.shape[1]}x{img.shape[0]}"
    }
