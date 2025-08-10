from dataclasses import dataclass
from typing import Dict

@dataclass
class GrainFeatures:
    length_px: float
    width_px: float
    area_px: float
    aspect: float

# --- Heuristic classifier (PLACEHOLDER THRESHOLDS – tune on your data!) ---
# Rough intuition (to calibrate):
# - 1121: long & slender → higher aspect
# - 1509: long but a bit broader → medium-high aspect
# - 1847: medium-long → medium aspect/area
# These are **not standards**; use a small labeled set to set real cutoffs.

def classify_heuristic(f: GrainFeatures) -> str:
    if f.length_px < 20 or f.width_px < 2:  # noise guard
        return "Unknown"

    if f.aspect >= 5.0 and f.length_px >= 60:
        return "1121"
    if 4.0 <= f.aspect < 5.0 and f.length_px >= 55:
        return "1509"
    if 3.2 <= f.aspect < 4.0 and f.length_px >= 48:
        return "1847"
    return "Unknown"

# --- (Optional) ML wrapper ---
# If you later train a model, expose a function with the same signature.
_model = None

def load_model_if_any(path: str = ""):
    # Implement if you train a model and pickle it.
    # Example:
    #   import joblib
    #   global _model
    #   _model = joblib.load(path)
    pass

def classify_ml_if_available(f: GrainFeatures) -> str:
    if _model is None:
        return None
    x = [[f.length_px, f.width_px, f.aspect, f.area_px]]
    pred = _model.predict(x)[0]
    return str(pred)

# Unified entry

def classify_grain(fdict: Dict[str, float]) -> str:
    f = GrainFeatures(**fdict)
    # Try ML first if available
    label = classify_ml_if_available(f)
    if label is not None:
        return label
    # Fall back to heuristic
    return classify_heuristic(f)