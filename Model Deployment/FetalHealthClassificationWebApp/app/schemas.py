"""
schemas.py
----------
/predict endpoint'i için Pydantic istek / yanıt şemaları.
"""

from pydantic import BaseModel, Field


class FetalHealthInput(BaseModel):
    """İstemciden beklenen 11 KTG özelliği."""

    baseline_value: float = Field(..., description="Baz fetal kalp atış hızı (atım/dak)", example=120.0)
    accelerations: float = Field(..., description="Saniyedeki akselerasyon sayısı", example=0.003)
    fetal_movement: float = Field(..., description="Saniyedeki fetal hareket sayısı", example=0.0)
    uterine_contractions: float = Field(..., description="Saniyedeki rahim kasılması sayısı", example=0.004)
    light_decelerations: float = Field(..., description="Saniyedeki hafif deselerasyon sayısı", example=0.0)
    severe_decelerations: float = Field(..., description="Saniyedeki şiddetli deselerasyon sayısı", example=0.0)
    prolongued_decelerations: float = Field(..., description="Saniyedeki uzamış deselerasyon sayısı", example=0.0)
    abnormal_short_term_variability: float = Field(..., description="Anormal kısa dönem değişkenlik süresi (%)", example=25.0)
    mean_value_of_short_term_variability: float = Field(..., description="Kısa dönem değişkenlik ortalama değeri", example=1.5)
    percentage_of_time_with_abnormal_long_term_variability: float = Field(..., description="Anormal uzun dönem değişkenlik süresi (%)", example=5.0)
    mean_value_of_long_term_variability: float = Field(..., description="Uzun dönem değişkenlik ortalama değeri", example=10.0)

    class Config:
        json_schema_extra = {
            "example": {
                "baseline_value": 120.0,
                "accelerations": 0.003,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.004,
                "light_decelerations": 0.0,
                "severe_decelerations": 0.0,
                "prolongued_decelerations": 0.0,
                "abnormal_short_term_variability": 25.0,
                "mean_value_of_short_term_variability": 1.5,
                "percentage_of_time_with_abnormal_long_term_variability": 5.0,
                "mean_value_of_long_term_variability": 10.0,
            }
        }


class PredictionResponse(BaseModel):
    """İstemciye döndürülen tahmin sonucu."""

    predicted_class_index: int
    predicted_label: str
    probabilities_percent: dict[str, float]
