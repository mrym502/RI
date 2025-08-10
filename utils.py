import cv2
import numpy as np

# --- Helper: decide threshold mode based on background brightness ---
def _is_light_background(gray: np.ndarray) -> bool:
    return float(np.mean(gray)) > 127.0

# --- Preprocess & segment grains ---
def segment_grains(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    light_bg = _is_light_background(gray)

    # Otsu threshold; invert if background is light so grains become white
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if light_bg:
        th = 255 - th

    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours (grains)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter tiny specks
    min_area = 30  # tune for your images
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return th, contours

# --- Measure a grain using a rotated rectangle ---
def measure_grain(contour: np.ndarray):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    length = max(w, h)  # pixels
    width = min(w, h)
    area = cv2.contourArea(contour)
    aspect = (length + 1e-6) / (width + 1e-6)
    return {
        "length_px": float(length),
        "width_px": float(width),
        "area_px": float(area),
        "aspect": float(aspect),
    }

# --- Make a simple ASCII bar chart ---
def ascii_bar_chart(breakdown: dict, total: int) -> str:
    if total == 0:
        return "(no grains detected)"
    lines = ["\nClass distribution:"]
    for k, v in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = (100.0 * v) / total
        bars = int(round(pct / 2))  # one block ≈ 2%
        lines.append(f"{k:>7} | {'█'*bars} {pct:5.1f}% ({v})")
    return "\n".join(lines)