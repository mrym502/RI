# classifier.py
import os

# عتبات من Environment (تقدرين تغيّرينها من Render بدون نشر كود)
A1121_MIN_ASPECT = float(os.getenv("A1121_MIN_ASPECT", "5.2"))
A1121_MIN_LENGTH = float(os.getenv("A1121_MIN_LENGTH", "120"))
A1121_MAX_WIDTH  = float(os.getenv("A1121_MAX_WIDTH",  "23"))

A1847_MIN_WIDTH  = float(os.getenv("A1847_MIN_WIDTH",  "24.5"))
A1847_MAX_ASPECT = float(os.getenv("A1847_MAX_ASPECT", "4.9"))

A1509_MAX_ASPECT = float(os.getenv("A1509_MAX_ASPECT", "4.8"))
A1509_MAX_LENGTH = float(os.getenv("A1509_MAX_LENGTH", "110"))
A1509_MAX_WIDTH  = float(os.getenv("A1509_MAX_WIDTH",  "24.0"))

def classify_grain(feats: dict) -> str:
    a = float(feats["aspect"])
    L = float(feats["length_px"])
    W = float(feats["width_px"])

    # 1121: أطول وأنحف
    if a >= A1121_MIN_ASPECT and L >= A1121_MIN_LENGTH and W <= A1121_MAX_WIDTH:
        return "1121"

    # 1847: أعرض غالباً
    if W >= A1847_MIN_WIDTH and a <= A1847_MAX_ASPECT:
        return "1847"

    # 1509: أقصر وأعرض من 1121 لكن أضيق من 1847
    if a <= A1509_MAX_ASPECT and L <= A1509_MAX_LENGTH and W <= A1509_MAX_WIDTH:
        return "1509"

    # لو ما تطابقت العتبات
    return "Unknown"
