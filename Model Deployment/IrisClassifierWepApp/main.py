from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import numpy as np
import os

from model import IrisClassifier


# ─── Load Model ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_classification_model.pth")

model = IrisClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

CLASS_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
CLASS_DESCRIPTIONS = {
    "Iris Setosa": "Küçük çiçek yapraklarıyla tanınan, genellikle soğuk iklimlerde yetişen bir türdür.",
    "Iris Versicolor": "Orta büyüklükte, mavi-mor renkli çiçeklere sahip, Kuzey Amerika'ya özgü bir türdür.",
    "Iris Virginica": "Büyük ve canlı renkli çiçekleriyle dikkat çeken, güneydoğu ABD'ye özgü bir türdür.",
}
CLASS_COLORS = {
    "Iris Setosa": "#a78bfa",
    "Iris Versicolor": "#34d399",
    "Iris Virginica": "#f97316",
}


# ─── Schemas ─────────────────────────────────────────────────────────────────

class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=20, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=20, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=20, description="Petal width in cm")


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    description: str
    color: str


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Iris Classifier API",
    description="PyTorch tabanlı Iris çiçeği sınıflandırma API'si",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/predict", response_model=PredictionResult)
async def predict(data: IrisInput):
    features = torch.tensor(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        logits = model(features)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    probabilities = {CLASS_NAMES[i]: round(probs[i] * 100, 2) for i in range(3)}

    return PredictionResult(
        predicted_class=pred_class,
        confidence=round(probs[pred_idx] * 100, 2),
        probabilities=probabilities,
        description=CLASS_DESCRIPTIONS[pred_class],
        color=CLASS_COLORS[pred_class],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "IrisClassifier", "version": "1.0.0"}
