# classifier.py
import os

# عتبات قابلة للتعديل من Environment (قيم أولية منطقية كبداية)
A1121_MIN_ASPECT = float(os.getenv("A1121_MIN_ASPECT", "5.5"))
A1121_MIN_LENGTH = float(os.getenv("A1121_MIN_LENGTH", "120"))

A1509_MAX_ASPECT = float(os.getenv("A1509_MAX_ASPECT", "4.7"))
A1509_MAX_LENGTH = float(os.getenv("A1509_MAX_LENGTH", "110"))

A1847_ASPECT_MIN = float(os.getenv("A1847_ASPECT_MIN", "3.3"))
A1847_ASPECT_MAX = float(os.getenv("A1847_ASPECT_MAX", "5.5"))

def classify_grain(feats: dict) -> str:
    a = float(feats["aspect"])
    L = float(feats["length_px"])
    W = float(feats["width_px"])

    # 1121: أطول وأنحف (نسبة طول/عرض أعلى)
    if a >= A1121_MIN_ASPECT and L >= A1121_MIN_LENGTH:
        return "1121"

    # 1509: أقصر وأعرض نسبيًا (نسبة أصغر)
    if a <= A1509_MAX_ASPECT and L <= A1509_MAX_LENGTH:
        return "1509"

    # 1847: في الوسط
    if A1847_ASPECT_MIN <= a <= A1847_ASPECT_MAX:
        return "1847"

    return "Unknown"
