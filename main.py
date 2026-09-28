from pathlib import Path

import joblib
from fastapi import FastAPI

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR

app = FastAPI(title="RWA ANN-XAI API")

# Load the saved Joblib artifact at startup.
preprocessor = joblib.load(
    ARTIFACT_DIR / "ann_fusion_preprocessor.joblib"
)

@app.get("/")
def home():
    return {"message": "RWA ANN-XAI backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(payload: dict):
    # Temporary connectivity test only.
    # Actual ANN + Tiny ViT + fusion inference must be added here.
    return {
        "message": "API is connected, but model inference is not implemented yet."
    }
