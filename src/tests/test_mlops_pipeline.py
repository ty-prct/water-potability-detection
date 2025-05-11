import os
import json
import pickle
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import the API app
from src.scripts.deploy_api import app

# Create a test client
client = TestClient(app)

# Fixture for sample water quality data
@pytest.fixture
def sample_water_quality():
    return {
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

# Test if model file exists
def test_model_exists():
    model_path = "results/best_model.pkl"
    assert os.path.isfile(model_path), f"Model file not found at {model_path}"
    
    # Test if model can be loaded
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Failed to load model"

# Test API health endpoint
def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model" in data

# Test metrics endpoint
def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "metrics" in data
    metrics = data["metrics"]
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

# Test prediction endpoint
def test_prediction_endpoint(sample_water_quality):
    response = client.post("/api/predict", json=sample_water_quality)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Potable", "Non-Potable"]
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert "model_used" in data
    assert "timestamp" in data

# Test invalid input handling
def test_invalid_input():
    # Missing required field
    invalid_data = {
        "ph": 7.5,
        "Hardness": 204.89,
        # Missing Solids
        "Chloramines": 7.3,
        "Sulfate": 368.51,
        "Conductivity": 564.31,
        "Organic_carbon": 10.38,
        "Trihalomethanes": 86.99,
        "Turbidity": 2.96
    }
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

    # Value out of range
    invalid_data = {
        "ph": 20.0,  # pH should be 0-14
        "Hardness": 204.89,
        "Solids": 20791.32,
        "Chloramines": 7.3,
        "Sulfate": 368.51,
        "Conductivity": 564.31,
        "Organic_carbon": 10.38,
        "Trihalomethanes": 86.99,
        "Turbidity": 2.96
    }
    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

# Test if model gives consistent results
def test_prediction_consistency(sample_water_quality):
    # Make multiple predictions and check if they are consistent
    response1 = client.post("/api/predict", json=sample_water_quality)
    response2 = client.post("/api/predict", json=sample_water_quality)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    assert data1["prediction"] == data2["prediction"]
    assert abs(data1["probability"] - data2["probability"]) < 1e-10
