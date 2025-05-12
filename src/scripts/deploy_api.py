import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from starlette_prometheus import PrometheusMiddleware, metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load models for A/B testing
HOME = os.getcwd()
MODELS_DIR = HOME + "/models"
RESULTS_DIR = HOME + "/results"

# Dictionary to store models
models = {}

# Load the best model (A) - primary model
MODEL_A_PATH = RESULTS_DIR + "/best_model.pkl"
try:
    with open(MODEL_A_PATH, "rb") as file:
        models["A"] = pickle.load(file)
    logger.info(f"Model A (primary) loaded from {MODEL_A_PATH}")
except Exception as e:
    logger.error(f"Failed to load Model A: {str(e)}")
    raise

# Load model B for A/B testing - using the second best model if available
try:
    # Try to load models from the models directory to use as Model B
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if model_files:
        # Use any model that's not the same as Model A for A/B testing
        MODEL_B_PATH = os.path.join(MODELS_DIR, model_files[0])
        with open(MODEL_B_PATH, "rb") as file:
            models["B"] = pickle.load(file)
        logger.info(f"Model B (secondary) loaded from {MODEL_B_PATH}")
except Exception as e:
    logger.warning(f"Failed to load Model B: {str(e)}. A/B testing will be disabled.")

# Default to model A
model = models["A"]

# Load feature details for validation
try:
    with open("results/evaluation_results.json", "rb") as f:
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
    version="1.0.0",
)

# A/B testing configuration
ab_testing_config = {
    "enabled": len(models) > 1,  # Enable only if we have multiple models
    "traffic_split": {"A": 0.90, "B": 0.10},  # 90% model A, 10% model B
    "metrics": {
        "A": {"requests": 0, "accuracy": 0, "latency_ms": []},
        "B": {"requests": 0, "accuracy": 0, "latency_ms": []},
    },
}

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
    Organic_carbon: float = Field(..., description="Organic carbon content", ge=0)
    Trihalomethanes: float = Field(..., description="Trihalomethanes level", ge=0)
    Turbidity: float = Field(..., description="Turbidity level", ge=0)

    # Add validators to check reasonable ranges
    @validator("ph")
    def ph_must_be_in_range(cls, v):
        if v < 0 or v > 14:
            raise ValueError("pH must be between 0 and 14")
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
                "Turbidity": 2.96,
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
        # Record start time for latency tracking
        start_time = datetime.now()

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # A/B testing - select model
        model_version = "A"  # Default to A
        if ab_testing_config["enabled"]:
            import random

            # Simple weighted random selection
            if random.random() > ab_testing_config["traffic_split"]["A"]:
                model_version = "B"
                selected_model = models["B"]
            else:
                selected_model = models["A"]
        else:
            selected_model = model  # Use default model

        # Track request
        ab_testing_config["metrics"][model_version]["requests"] += 1

        # Make prediction with selected model
        prediction = selected_model.predict(input_data)[0]
        potability = "Potable" if prediction == 1 else "Non-Potable"

        # Get probability if available
        probability = 0.0
        try:
            probability = float(selected_model.predict_proba(input_data)[0][1])
        except Exception:
            # Some models don't support predict_proba
            probability = float(prediction)

        # Get feature importance if available
        feature_importance = None
        if hasattr(selected_model, "feature_importances_"):
            feature_importance = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(
                    input_data.columns, selected_model.feature_importances_
                )
            ]
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        ab_testing_config["metrics"][model_version]["latency_ms"].append(latency_ms)

        # Keep only last 100 latency measurements
        if len(ab_testing_config["metrics"][model_version]["latency_ms"]) > 100:
            ab_testing_config["metrics"][model_version][
                "latency_ms"
            ] = ab_testing_config["metrics"][model_version]["latency_ms"][-100:]

        # Create response
        timestamp = datetime.now().isoformat()
        model_name = f"{best_model_name} (Version {model_version})"
        response = {
            "prediction": potability,
            "probability": probability,
            "model_used": model_name,
            "timestamp": timestamp,
            "input_data": data.dict(),
            "feature_importance": feature_importance,
        }

        # Log prediction for monitoring
        predictions_log.append(response)
        if len(predictions_log) > MAX_LOG_SIZE:
            predictions_log.pop(0)

        logger.info(
            f"Prediction: {potability}, Probability: {probability:.4f}, Model: {model_name}, Latency: {latency_ms:.2f}ms"
        )
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": best_model_name}


@app.get("/api/metrics")
def get_metrics():
    """Get model metrics"""
    try:
        with open("results/evaluation_results.json", "rb") as f:
            evaluation_data = json.load(f)
        return {
            "model": evaluation_data.get("best_model", "unknown"),
            "metrics": evaluation_data.get("test_metrics", {}).get(best_model_name, {}),
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving metrics: {str(e)}"
        )


@app.get("/api/monitoring/recent_predictions")
def get_recent_predictions():
    """Get recent predictions for monitoring"""
    return {"predictions": predictions_log[-20:]}  # Return last 20 predictions


@app.get("/api/ab-testing")
def get_ab_testing_stats():
    """Get A/B testing statistics"""
    stats = {
        "enabled": ab_testing_config["enabled"],
        "traffic_split": ab_testing_config["traffic_split"],
        "metrics": {},
    }

    # Calculate statistics for each model
    for version, metrics in ab_testing_config["metrics"].items():
        avg_latency = 0
        if metrics["latency_ms"]:
            avg_latency = sum(metrics["latency_ms"]) / len(metrics["latency_ms"])

        stats["metrics"][version] = {
            "requests": metrics["requests"],
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": metrics["accuracy"] / max(1, metrics["requests"]),
        }

    return stats


@app.post("/api/ab-testing/configure")
def configure_ab_testing(config: dict):
    """Configure A/B testing parameters"""
    # Validate the config
    if "enabled" in config:
        ab_testing_config["enabled"] = config["enabled"]

    if "traffic_split" in config:
        # Ensure traffic split sums to 1.0
        if abs(sum(config["traffic_split"].values()) - 1.0) > 0.001:
            raise HTTPException(status_code=400, detail="Traffic split must sum to 1.0")

        # Ensure both versions are included
        if not all(v in config["traffic_split"] for v in ["A", "B"]):
            raise HTTPException(
                status_code=400,
                detail="Traffic split must include both 'A' and 'B' versions",
            )

        ab_testing_config["traffic_split"] = config["traffic_split"]

    return {"status": "success", "config": ab_testing_config}


if __name__ == "__main__":
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8000, reload=True)
