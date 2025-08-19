from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a saved LinearRegression pipeline",
    version="1.0.0",
)

# Globals populated on startup
artifact = None
model = None
features_order = None
metrics = None

class PredictInput(BaseModel):
    area: float = Field(gt=0, description="Square feet")
    bedrooms: int = Field(ge=0, description="Number of bedrooms")
    bathrooms: int = Field(ge=0, description="Number of bathrooms")
    stories: int = Field(ge=1, description="Number of stories")

class PredictOutput(BaseModel):
    prediction: float

@app.on_event("startup")
def load_model():
    global artifact, model, features_order, metrics
    try:
        artifact = joblib.load("model.pkl")
        model = artifact["model"]
        features_order = artifact["features"]
        metrics = artifact.get("metrics", {})
    except Exception as e:
        # Fail fast if model can't be loaded
        raise RuntimeError(f"Could not load model.pkl: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    try:
        # Ensure order: [area, bedrooms, bathrooms, stories]
        row = np.array([[payload.area, payload.bedrooms, payload.bathrooms, payload.stories]])
        pred = float(model.predict(row)[0])
        return {"prediction": pred}
    except Exception as e:
        # Proper error handling with clear message
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/model-info")
def model_info():
    try:
        model_name = (
            type(model.named_steps["model"]).__name__
            if hasattr(model, "named_steps") else type(model).__name__
        )
        return {
            "model": model_name,
            "problem_type": "regression",
            "features": features_order,
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Info unavailable: {e}")
