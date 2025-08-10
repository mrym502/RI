from flask import Flask, request, jsonify
import numpy as np
import cv2

from utils import segment_grains, measure_grain, ascii_bar_chart
from classifier import classify_grain

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Send form-data with key 'image'."}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Invalid image."}), 400

    mask, contours = segment_grains(bgr)

    items = []
    breakdown = {"1121": 0, "1509": 0, "1847": 0, "Unknown": 0}

    # حدود الصورة لاستخدامها في فلترة الشواذ
    H, W = bgr.shape[:2]
    img_area = H * W

    for c in contours:
        # قياس خصائص كل حبّة
        feats = measure_grain(c)

        # فلترة الكائنات الشاذة (كتل/ظلال/اندماج)
        if (
            feats["length_px"] > 250 or
            feats["width_px"]  > 60  or
            feats["area_px"]   > 0.05 * img_area or
            feats["aspect"]    > 15
        ):
            continue

        # تصنيف الحبّة
        label = classify_grain(feats)
        breakdown[label] = breakdown.get(label, 0) + 1
        items.append({"features": feats, "label": label})

    total = sum(breakdown.values())
    percentages = {
        k: (0.0 if total == 0 else round(100.0 * v / total, 2))
        for k, v in breakdown.items()
    }

    majority = None
    mixture = "No"
    if total > 0:
        sorted_classes = sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
        majority = sorted_classes[0][0] if sorted_classes[0][1] > 0 else None
        non_major_counts = [c for k, c in breakdown.items() if k != majority and k != "Unknown"]
        mixture = "Yes" if any(v > 0 for v in non_major_counts) else "No"

    chart = ascii_bar_chart(breakdown, total)

    return jsonify({
        "total_grains": total,
        "majority": majority,
        "mixture": mixture,
        "breakdown": breakdown,
        "percentages": percentages,
        "per_grain": items,
        "chart": chart,
    })

if __name__ == "__main__":
    # تشغيل محلي
    app.run(host="0.0.0.0", port=8000, debug=True)
