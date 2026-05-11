"""
predictor.py
------------
Eğitilmiş PyTorch modelini ve StandardScaler'ı uygulama başlarken
bir kez yükler; ardından API endpoint'i tarafından kullanılan
predict() fonksiyonunu dışa açar.
"""

from pathlib import Path

import joblib
import numpy as np
import torch

from app.model import ClassificationModel
from app.schemas import FetalHealthInput, PredictionResponse

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # proje kök dizini

MODEL_PATH = BASE_DIR / "models" / "fetal_health_classification.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

CLASS_NAMES = {
    0: "Normal",
    1: "Suspect",
    2: "Pathological",
}

FEATURE_ORDER = [
    "baseline_value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
]

# ---------------------------------------------------------------------------
# Modül düzeyinde tekil nesneler (import sırasında bir kez yüklenir)
# ---------------------------------------------------------------------------

def _load_model() -> ClassificationModel:
    """Modeli örnekler ve kaydedilmiş state_dict'i yükler."""
    m = ClassificationModel()
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m


def _load_scaler():
    """Diskten eğitilmiş StandardScaler'ı yükler."""
    return joblib.load(SCALER_PATH)


model: ClassificationModel = _load_model()
scaler = _load_scaler()

# ---------------------------------------------------------------------------
# Tahmin mantığı
# ---------------------------------------------------------------------------

def predict(input_data: FetalHealthInput) -> PredictionResponse:
    """
    Tek bir örnek üzerinde çıkarım (inference) yapar.

    Adımlar:
        1. FEATURE_ORDER sırasında numpy dizisi oluşturulur.
        2. Önceden eğitilmiş scaler ile ölçeklenir.
        3. float32 tensörüne dönüştürülür.
        4. Model üzerinden ileri geçiş yapılır (inference_mode).
        5. Softmax → argmax ile tahmin edilen sınıf bulunur.
        6. JSON serileştirilebilir PredictionResponse döndürülür.
    """
    sample = np.array(
        [[getattr(input_data, feat) for feat in FEATURE_ORDER]],
        dtype=np.float64,
    )

    sample_scaled = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    with torch.inference_mode():
        logits = model(sample_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()  # boyut: (3,)
        predicted_class: int = torch.argmax(probs).item()

    predicted_label = CLASS_NAMES[predicted_class]

    probabilities_percent = {
        CLASS_NAMES[i]: round(prob.item() * 100, 4)
        for i, prob in enumerate(probs)
    }

    return PredictionResponse(
        predicted_class_index=predicted_class,
        predicted_label=predicted_label,
        probabilities_percent=probabilities_percent,
    )
