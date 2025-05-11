#!/usr/bin/env python

# Example Tests

import pytest
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case("semaphore") == "Semaphore"


# Test data loading
def test_data_loading():
    data_path = "data/clean_data.csv"
    assert os.path.exists(data_path), "Clean data file does not exist."
    data = pd.read_csv(data_path)
    assert not data.empty, "Clean data file is empty."

# Test model loading
def test_model_loading():
    model_path = "models/random_forest_model.pkl"
    assert os.path.exists(model_path), "Model file does not exist."
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    assert isinstance(model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier."

# Test model prediction
def test_model_prediction():
    model_path = "models/random_forest_model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Create a sample input
    sample_input = pd.DataFrame({
        "ph": [7.0],
        "Hardness": [200.0],
        "Solids": [10000.0],
        "Chloramines": [7.0],
        "Sulfate": [300.0],
        "Conductivity": [400.0],
        "Organic_carbon": [10.0],
        "Trihalomethanes": [80.0],
        "Turbidity": [4.0]
    })

    prediction = model.predict(sample_input)
    assert prediction in [[0], [1]], "Prediction is not valid."

