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

    H, W = bgr.shape[:2]
    img_area = H * W

    for c in contours:
        # 1) قيسي خصائص الحبة
        feats = measure_grain(c)

        # 2) فلترة الكائنات الشاذة (كتل/ظلال/اندماج)
        if (
            feats["length_px"] > 250 or
            feats["width_px"]  > 60  or
            feats["area_px"]   > 0.05 * img_area or
            feats["aspect"]    > 15
        ):
            continue

        # 3) صنّفي الحبة
        label = classify_grain(feats)
        breakdown[label] = breakdown.get(label, 0) + 1
        items.append({"features": feats, "label": label})

    total = sum(breakdown.values())
    percentages = {k: (0.0 if total == 0 else round(100.0 * v / total, 2))
                   for k, v in breakdown.items()}

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