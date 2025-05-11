import os
import pickle
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette_prometheus import PrometheusMiddleware, metrics
from pydantic import BaseModel, Field, validator
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the best model
MODEL_PATH = "results/best_model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Load feature details for validation
try:
    with open("results/evaluation_results_20250511_000301.json", "rb") as f:
        evaluation_data = json.load(f)
    best_model_name = evaluation_data.get("best_model", "random_forest_model")
    logger.info(f"Best model according to evaluation: {best_model_name}")
except Exception as e:
    logger.warning(f"Could not load evaluation data: {str(e)}")
    best_model_name = "random_forest_model"

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction API",
    description="This API predicts whether water is potable or not based on quality metrics.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add prometheus middleware for metrics
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# Directory setup for templates and static files
TEMPLATES_DIR = "web/templates"
STATIC_DIR = "web/static"

# Create directories if they don't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(f"{STATIC_DIR}/css", exist_ok=True)
os.makedirs(f"{STATIC_DIR}/js", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Define input data schema with validation


class WaterQualityData(BaseModel):
    ph: float = Field(..., description="pH value of water", ge=0, le=14)
    Hardness: float = Field(..., description="Hardness of water", ge=0)
    Solids: float = Field(..., description="Total dissolved solids", ge=0)
    Chloramines: float = Field(..., description="Chloramines level", ge=0)
    Sulfate: float = Field(..., description="Sulfate content", ge=0)
    Conductivity: float = Field(..., description="Conductivity of water", ge=0)
    Organic_carbon: float = Field(...,
                                  description="Organic carbon content", ge=0)
    Trihalomethanes: float = Field(...,
                                   description="Trihalomethanes level", ge=0)
    Turbidity: float = Field(..., description="Turbidity level", ge=0)

    # Add validators to check reasonable ranges
    @validator('ph')
    def ph_must_be_in_range(cls, v):
        if v < 0 or v > 14:
            raise ValueError('pH must be between 0 and 14')
        return v

    class Config:
        schema_extra = {
            "example": {
                "ph": 7.5,
                "Hardness": 204.89,
                "Solids": 20791.32,
                "Chloramines": 7.3,
                "Sulfate": 368.51,
                "Conductivity": 564.31,
                "Organic_carbon": 10.38,
                "Trihalomethanes": 86.99,
                "Turbidity": 2.96
            }
        }

# Model for prediction response


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_used: str
    timestamp: str
    input_data: Dict[str, float]
    feature_importance: Optional[List[Dict[str, Any]]] = None


# Store predictions for monitoring
predictions_log = []
MAX_LOG_SIZE = 100


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict", response_model=PredictionResponse)
def predict(data: WaterQualityData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Make prediction
        prediction = model.predict(input_data)[0]
        potability = "Potable" if prediction == 1 else "Non-Potable"

        # Get probability if available
        probability = 0.0
        try:
            probability = float(model.predict_proba(input_data)[0][1])
        except:
            # Some models don't support predict_proba
            probability = float(prediction)

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(input_data.columns, model.feature_importances_)
            ]
            # Sort by importance
            feature_importance.sort(
                key=lambda x: x["importance"], reverse=True)

        # Create response
        timestamp = datetime.now().isoformat()
        response = {
            "prediction": potability,
            "probability": probability,
            "model_used": best_model_name,
            "timestamp": timestamp,
            "input_data": data.dict(),
            "feature_importance": feature_importance
        }

        # Log prediction for monitoring
        predictions_log.append(response)
        if len(predictions_log) > MAX_LOG_SIZE:
            predictions_log.pop(0)

        logger.info(
            f"Prediction: {potability}, Probability: {probability:.4f}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": best_model_name}


@app.get("/api/metrics")
def get_metrics():
    """Get model metrics"""
    try:
        with open("results/evaluation_results_20250511_000301.json", "rb") as f:
            evaluation_data = json.load(f)
        return {
            "model": evaluation_data.get("best_model", "unknown"),
            "metrics": evaluation_data.get("test_metrics", {}).get(best_model_name, {})
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@app.get("/api/monitoring/recent_predictions")
def get_recent_predictions():
    """Get recent predictions for monitoring"""
    return {"predictions": predictions_log[-20:]}  # Return last 20 predictions


if __name__ == "__main__":
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8000, reload=True)
