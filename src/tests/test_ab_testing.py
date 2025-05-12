"""
Tests for A/B testing functionality in the water potability detection API.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Set up path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Conditionally import API
try:
    from scripts.deploy_api import app

    client = TestClient(app)
except ImportError:
    # For CI environment where model might not be available
    pytestmark = pytest.mark.skip(
        "API module could not be loaded, skipping tests")

HOME = os.getcwd()
DATA_FOLDER = HOME + "/data/"
MODEL_FOLDER = HOME + "/models/"
RESULTS_FOLDER = HOME + "/results/"


@pytest.mark.skipif(
    not os.path.exists(RESULTS_FOLDER + "best_model.pkl"),
    reason="Model file not found, skipping test",
)
def test_ab_testing_endpoint_exists():
    """Test that the A/B testing endpoint exists"""
    response = client.get("/api/ab-testing")
    assert response.status_code == 200
    data = response.json()

    # Check that necessary fields are present
    assert "enabled" in data
    assert "traffic_split" in data
    assert "metrics" in data


@pytest.mark.skipif(
    not os.path.exists(RESULTS_FOLDER + "best_model.pkl")
    or not os.path.exists(MODEL_FOLDER + "random_forest_model.pkl"),
    reason="Model files not found, skipping test",
)
def test_ab_testing_configuration():
    """Test configuring A/B testing settings"""
    # Test enabling A/B testing
    response = client.post("/api/ab-testing/configure", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["config"]["enabled"] is True

    # Test setting traffic split
    response = client.post(
        "/api/ab-testing/configure", json={"traffic_split": {"A": 0.75, "B": 0.25}}
    )
    assert response.status_code == 200
    assert response.json()["config"]["traffic_split"]["A"] == 0.75
    assert response.json()["config"]["traffic_split"]["B"] == 0.25


@pytest.mark.skipif(
    not os.path.exists(RESULTS_FOLDER + "best_model.pkl"),
    reason="Model file not found, skipping test",
)
def test_prediction_with_ab_testing():
    """Test that prediction works with A/B testing enabled"""
    # Enable A/B testing
    client.post("/api/ab-testing/configure", json={"enabled": True})

    # Make a prediction
    sample_data = {
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

    response = client.post("/api/predict", json=sample_data)
    assert response.status_code == 200

    # Check that prediction contains model version info
    prediction = response.json()
    assert "model_used" in prediction
    assert "Version" in prediction["model_used"]

    # Check that metrics were updated
    ab_stats = client.get("/api/ab-testing").json()
    assert (
        ab_stats["metrics"]["A"]["requests"] > 0
        or ab_stats["metrics"]["B"]["requests"] > 0
    )


# def test_ab_testing_api_validation():
#     """Test validation of A/B testing configuration"""
#     # Test invalid traffic split (doesn't sum to 1.0)
#     response = client.post("/api/ab-testing/configure",
#                           json={"traffic_split": {"A": 0.6, "B": 0.6}})
#     assert response.status_code == 400

#     # Test missing required version
#     response = client.post("/api/ab-testing/configure",
#                           json={"traffic_split": {"A": 1.0}})
#     assert response.status_code == 400
