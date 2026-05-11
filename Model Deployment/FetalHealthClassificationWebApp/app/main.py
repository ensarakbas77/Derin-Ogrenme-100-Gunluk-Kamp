"""
main.py
-------
FastAPI uygulamasının giriş noktası.

Rotalar:
    GET  /          – HTML arayüzünü döndürür
    POST /predict   – 11 KTG özelliğini alır, tahmin JSON'ı döndürür
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import FetalHealthInput, PredictionResponse
from app.predictor import predict

# ---------------------------------------------------------------------------
# Uygulama kurulumu
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # proje kök dizini

app = FastAPI(
    title="Fetal Sağlık Sınıflandırıcı API",
    description="11 KTG özelliğinden ANN tabanlı fetal sağlık sınıflandırması.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ---------------------------------------------------------------------------
# Rotalar
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Arayüzü sun")
async def index(request: Request):
    """Ana HTML sayfasını döndürür."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Fetal sağlık sınıfını tahmin et",
)
async def predict_endpoint(input_data: FetalHealthInput) -> PredictionResponse:
    """
    11 KTG özellik değerini alır ve şunları döndürür:
    - predicted_class_index (0 / 1 / 2)
    - predicted_label (Normal / Suspect / Pathological)
    - probabilities_percent — her sınıf için yüzde olasılık
    """
    try:
        result = predict(input_data)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Tahmin başarısız: {exc}") from exc
    return result
