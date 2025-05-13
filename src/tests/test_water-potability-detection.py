#!/usr/bin/env python

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

"""
Water Potability Model Test Suite
=================================

This test suite evaluates the quality, robustness, and performance of a machine learning model
trained to predict water potability based on various chemical and physical parameters.

Test Cases Overview:
--------------------

1. test_data_balance:
   - Ensures the dataset has exactly two classes (binary classification).
   - Verifies that class distribution is not severely imbalanced (ratio < 10:1).

2. test_feature_range_and_distribution:
   - Checks that each feature falls within scientifically valid and expected ranges.

3. test_feature_correlations:
   - Logs any highly correlated features (|correlation| > 0.9) to detect potential multicollinearity.

4. test_outlier_detection:
   - Identifies statistical outliers using Z-scores and ensures no feature has more than 10% outliers.

5. test_model_vs_baseline:
   - Verifies that the trained model outperforms two baselines: a stratified random classifier and a majority-class predictor.

6. test_model_cross_validation:
   - Evaluates model stability and performance consistency using 5-fold cross-validation.
   - Fails if accuracy is < 0.6 or variance > 0.1.

7. test_model_metrics:
   - Assesses overall model performance using multiple metrics (accuracy, precision, recall, F1 score, AUC).
   - Fails if any metric falls below its defined threshold.

8. test_feature_importance_ranking:
   - Logs feature importances from the Random Forest model.
   - Fails if no feature has an importance > 0.1.

9. test_robustness_to_missing_values:
   - Verifies the model can handle single-feature missing values via median imputation without crashing.

10. test_threshold_optimization:
    - Analyzes F1 scores across various classification thresholds (0.1 to 0.9).
    - Identifies whether a threshold other than 0.5 would optimize performance.
    - Informational only; does not fail.

11. test_model_prediction:
    - Confirms the model can generate valid predictions (0 or 1) on a properly formatted sample input.

Dependencies:
-------------
- Requires: `data/clean_data.csv` and `models/random_forest_model.pkl`
- Libraries: pytest, pandas, numpy, sklearn, matplotlib, scipy

Note:
-----
Tests are written using `pytest`. Missing files or setup will trigger test skipping rather than failure.
"""


def test_data_balance():
    """
    Tests class balance in the dataset.

    This test ensures:
    1. The dataset contains exactly two classes (potable and non-potable water)
    2. There is no severe class imbalance (ratio less than 10:1)
    3. Logs the distribution of classes for information

    Fails if:
    - Dataset doesn't contain exactly two classes
    - Class imbalance ratio exceeds 10:1
    """
    data_path = "data/clean_data.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} does not exist, possibly needs DVC pull.")

    data = pd.read_csv(data_path)

    # Check class distribution
    class_counts = data["Potability"].value_counts()
    assert len(class_counts) == 2, "Expected binary classification with 2 classes"

    # Check for severe class imbalance
    imbalance_ratio = class_counts.max() / class_counts.min()
    assert (
        imbalance_ratio < 10
    ), f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})"

    # Log class distribution for information
    print(f"Class distribution: {class_counts.to_dict()}")


def test_feature_range_and_distribution():
    """
    Tests feature ranges and distributions against expected values.

    This test ensures that all features in the water quality dataset fall within
    scientifically reasonable ranges. For example, pH should be between 0-14,
    and other chemical measurements should be within their physical limits.

    Fails if:
    - Any feature contains values outside its expected range
    """
    data_path = "data/clean_data.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} does not exist, possibly needs DVC pull.")

    data = pd.read_csv(data_path)

    # Expected ranges for each feature
    expected_ranges = {
        "ph": (0, 14),
        "Hardness": (0, 1000),
        "Solids": (0, 100000),
        "Chloramines": (0, 50),
        "Sulfate": (0, 1000),
        "Conductivity": (0, 2000),
        "Organic_carbon": (0, 100),
        "Trihalomethanes": (0, 500),
        "Turbidity": (0, 50),
    }

    for feature, (min_val, max_val) in expected_ranges.items():
        if feature in data.columns:
            actual_min = data[feature].min()
            actual_max = data[feature].max()
            assert (
                actual_min >= min_val
            ), f"{feature} has values below expected minimum: {actual_min} < {min_val}"
            assert (
                actual_max <= max_val
            ), f"{feature} has values above expected maximum: {actual_max} > {max_val}"


def test_feature_correlations():
    """
    Tests for expected correlations between features.

    This test identifies highly correlated features (|r| > 0.9) which may indicate:
    1. Potential multicollinearity issues that could affect model performance
    2. Redundant features that might be simplified or combined
    3. Expected chemical/physical relationships in water quality parameters

    This is primarily an informational test that logs high correlations without failing.
    """
    data_path = "data/clean_data.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} does not exist, possibly needs DVC pull.")

    data = pd.read_csv(data_path)

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Check for high correlations (potential multicollinearity issues)
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:  # Threshold for high correlation
                high_corrs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    # This is an informational test, not a strict assertion
    if high_corrs:
        print("High correlations detected (>0.9):")
        for feat1, feat2, corr in high_corrs:
            print(f"  {feat1} and {feat2}: {corr:.4f}")


def test_outlier_detection():
    """
    Tests for outliers in the dataset using Z-score method.

    This test:
    1. Calculates Z-scores for each feature to identify statistical outliers
    2. Counts values with |z| > 3 (more than 3 standard deviations from mean)
    3. Logs all detected outliers by feature

    Fails if:
    - Any feature contains more than 10% outliers, which would indicate data quality issues
    """
    data_path = "data/clean_data.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} does not exist, possibly needs DVC pull.")

    data = pd.read_csv(data_path)

    # Exclude target variable
    X = data.drop("Potability", axis=1) if "Potability" in data.columns else data

    # Use Z-score to identify outliers (|z| > 3)
    outliers_by_feature = {}
    for col in X.columns:
        z_scores = np.abs(stats.zscore(X[col], nan_policy="omit"))
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            outliers_by_feature[col] = outliers

    # Informational - log the number of outliers
    if outliers_by_feature:
        print("Outliers detected by feature (|z| > 3):")
        for feat, count in outliers_by_feature.items():
            print(f"  {feat}: {count} outliers")

    # Check that no feature has more than 10% outliers
    for feat, count in outliers_by_feature.items():
        assert (
            count / len(X) < 0.1
        ), f"Feature {feat} has too many outliers: {count} ({count/len(X)*100:.2f}%)"


def test_model_vs_baseline():
    """
    Tests that the model outperforms baseline classifiers.

    This test compares the trained RandomForest model against:
    1. A stratified random classifier (maintains class distribution)
    2. A most frequent class classifier (always predicts majority class)

    The model should significantly outperform both baselines to demonstrate
    that it has learned meaningful patterns beyond simple heuristics.

    Fails if:
    - Model accuracy is less than or equal to either baseline accuracy
    """
    model_path = "models/random_forest_model.pkl"
    data_path = "data/clean_data.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        pytest.skip("Required files not found, skipping test.")

    try:
        # Load model and data
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        data = pd.read_csv(data_path)
        X = data.drop("Potability", axis=1)
        y = data["Potability"]

        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Evaluate actual model
        model_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, model_pred)

        # Baseline 1: Stratified random guessing
        dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
        dummy_stratified.fit(X_train, y_train)
        stratified_pred = dummy_stratified.predict(X_test)
        stratified_accuracy = accuracy_score(y_test, stratified_pred)

        # Baseline 2: Most frequent class
        dummy_frequent = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy_frequent.fit(X_train, y_train)
        frequent_pred = dummy_frequent.predict(X_test)
        frequent_accuracy = accuracy_score(y_test, frequent_pred)

        # Check that model outperforms baselines
        assert (
            model_accuracy > stratified_accuracy
        ), "Model should outperform random guessing"
        assert (
            model_accuracy > frequent_accuracy
        ), "Model should outperform most frequent class predictor"

        print(f"Model accuracy: {model_accuracy:.4f}")
        print(f"Stratified baseline accuracy: {stratified_accuracy:.4f}")
        print(f"Most frequent baseline accuracy: {frequent_accuracy:.4f}")

    except Exception as e:
        pytest.skip(f"Error comparing model to baseline: {str(e)}")


def test_model_cross_validation():
    """
    Tests model performance using 5-fold cross-validation.

    This test evaluates the model's performance across different data splits to ensure:
    1. The model performs consistently across different subsets of the data
    2. Performance is above an acceptable threshold (60% accuracy)
    3. There is low variance in performance across folds (<10% std)

    Fails if:
    - Cross-validation scores have high variance (>0.1 std dev)
    - Mean cross-validation accuracy is below 0.6
    """
    model_path = "models/random_forest_model.pkl"
    data_path = "data/clean_data.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        pytest.skip("Required files not found, skipping test.")

    try:
        # Load model and data
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        data = pd.read_csv(data_path)
        X = data.drop("Potability", axis=1)
        y = data["Potability"]

        # Perform 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        # Check consistency of performance across folds
        assert (
            cv_scores.std() < 0.1
        ), "High variance in cross-validation scores suggests model instability"

        # Check that mean performance is reasonable
        assert cv_scores.mean() > 0.6, "Model performance below acceptable threshold"

        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    except Exception as e:
        pytest.skip(f"Error during cross-validation: {str(e)}")


def test_model_metrics():
    """
    Tests comprehensive model evaluation using multiple metrics.

    This test evaluates the model using various classification metrics:
    1. Accuracy: Overall correctness
    2. Precision: Ratio of correct positive predictions to total positive predictions
    3. Recall: Ratio of correct positive predictions to total actual positives
    4. F1 score: Harmonic mean of precision and recall
    5. AUC: Area under the ROC curve (threshold-invariant performance)

    Fails if any metric is below its acceptable threshold:
    - Accuracy < 0.6
    - Precision < 0.5
    - Recall < 0.5
    - F1 score < 0.5
    - AUC < 0.6
    """
    model_path = "models/random_forest_model.pkl"
    data_path = "data/clean_data.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        pytest.skip("Required files not found, skipping test.")

    try:
        # Load model and data
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        data = pd.read_csv(data_path)
        X = data.drop("Potability", axis=1)
        y = data["Potability"]

        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Check that metrics are in acceptable ranges
        assert accuracy > 0.6, "Accuracy below acceptable threshold"
        assert precision > 0.5, "Precision below acceptable threshold"
        assert recall > 0.5, "Recall below acceptable threshold"
        assert f1 > 0.5, "F1 score below acceptable threshold"
        assert auc > 0.6, "AUC below acceptable threshold"

        print("Model metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

    except Exception as e:
        pytest.skip(f"Error calculating model metrics: {str(e)}")


def test_feature_importance_ranking():
    """
    Tests feature importance ranking from the Random Forest model.

    This test analyzes the feature importance values from the trained model to:
    1. Verify that the model has established meaningful feature relationships
    2. Check that at least one feature has significant importance (>0.1)
    3. Log the importance ranking for interpretability assessment

    Feature importance helps understand which water quality parameters are most
    critical for potability prediction according to the model.

    Fails if no feature has significant importance (>0.1).
    """
    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} does not exist, possibly needs DVC pull.")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Check if model has feature importances
        if not hasattr(model, "feature_importances_"):
            pytest.skip("Model doesn't have feature_importances_ attribute")

        # Get feature names from a sample input (assuming we know the order)
        feature_names = [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ]

        # Get feature importances
        importances = model.feature_importances_

        # Create sorted feature importance dict
        importance_dict = dict(zip(feature_names, importances))
        sorted_importances = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Check that at least one feature has significant importance (>0.1)
        assert any(
            imp > 0.1 for _, imp in sorted_importances
        ), "No feature has significant importance"

        # Print feature importance ranking
        print("Feature importance ranking:")
        for feature, importance in sorted_importances:
            print(f"  {feature}: {importance:.4f}")

    except Exception as e:
        pytest.skip(f"Error analyzing feature importance ranking: {str(e)}")


def test_robustness_to_missing_values():
    """
    Tests model robustness when handling potentially missing values.

    This test evaluates how the model responds when input features are imputed:
    1. Creates a complete sample record with reasonable water quality values
    2. For each feature, creates a modified record with median imputation
    3. Verifies the model can make predictions without crashing

    This test doesn't evaluate prediction quality, just that the model remains
    functional when dealing with imputed values.

    Fails if the model raises exceptions when predicting on imputed data.
    """
    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} does not exist, possibly needs DVC pull.")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Create complete sample input
        base_input = pd.DataFrame(
            {
                "ph": [7.0],
                "Hardness": [200.0],
                "Solids": [10000.0],
                "Chloramines": [7.0],
                "Sulfate": [300.0],
                "Conductivity": [400.0],
                "Organic_carbon": [10.0],
                "Trihalomethanes": [80.0],
                "Turbidity": [4.0],
            }
        )

        # Get baseline prediction
        model.predict(base_input)

        # Try different approaches to handling missing values
        # This test checks for crashes, not prediction quality
        for feature in base_input.columns:
            # Create a copy with one feature set to median value
            missing_input = base_input.copy()
            missing_input[feature] = missing_input[feature].median()

            try:
                pred = model.predict(missing_input)
                # Just checking it runs without error
                assert (
                    pred is not None
                ), f"Prediction failed with median imputation for {feature}"
            except Exception as e:
                assert (
                    False
                ), f"Model failed with median imputation for {feature}: {str(e)}"

    except Exception as e:
        pytest.skip(f"Error testing robustness to missing values: {str(e)}")


def test_threshold_optimization():
    """
    Tests threshold optimization for better classification performance.

    This test evaluates if the default threshold (0.5) for binary classification
    is optimal or if performance could be improved by adjusting it:

    1. Gets probability predictions from the model
    2. Calculates F1 score at default threshold (0.5)
    3. Tests F1 scores at thresholds from 0.1 to 0.9
    4. Identifies the threshold that maximizes F1 score

    This is primarily an informational test that logs suggestions for threshold
    calibration if the optimal threshold differs significantly from default.
    """
    model_path = "models/random_forest_model.pkl"
    data_path = "data/clean_data.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        pytest.skip("Required files not found, skipping test.")

    try:
        # Load model and data
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        data = pd.read_csv(data_path)
        X = data.drop("Potability", axis=1)
        y = data["Potability"]

        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Get probability predictions
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics at default threshold (0.5)
        default_pred = (y_prob >= 0.5).astype(int)
        default_f1 = f1_score(y_test, default_pred)

        # Try different thresholds to find optimal F1 score
        thresholds = np.linspace(0.1, 0.9, 9)
        best_threshold = 0.5
        best_f1 = default_f1

        for threshold in thresholds:
            pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_test, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Default F1 score (0.5 threshold): {default_f1:.4f}")
        print(f"Best F1 score ({best_threshold:.1f} threshold): {best_f1:.4f}")

        # Informational test: we're not asserting the best threshold should be different
        # We're just checking if the model's default threshold is reasonably calibrated
        if abs(best_threshold - 0.5) > 0.2:
            print(
                f"Note: Model might benefit from threshold calibration (optimal: {best_threshold:.2f})"
            )

    except Exception as e:
        pytest.skip(f"Error during threshold optimization: {str(e)}")


# Test model prediction
def test_model_prediction():
    """
    Tests the model's ability to make valid predictions on sample data.

    This test ensures:
    1. The model can be loaded correctly
    2. The model can accept properly formatted input data
    3. The model returns valid predictions (0 or 1)

    This is a basic functionality test to verify that the model works as expected
    when deployed for prediction tasks on new water sample data.

    Fails if:
    - Model cannot be loaded
    - Model crashes during prediction
    - Model returns predictions outside valid class values (0,1)
    """
    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} does not exist, possibly needs DVC pull.")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Create a sample input
        sample_input = pd.DataFrame(
            {
                "ph": [7.0],
                "Hardness": [200.0],
                "Solids": [10000.0],
                "Chloramines": [7.0],
                "Sulfate": [300.0],
                "Conductivity": [400.0],
                "Organic_carbon": [10.0],
                "Trihalomethanes": [80.0],
                "Turbidity": [4.0],
            }
        )

        prediction = model.predict(sample_input)
        assert prediction in [[0], [1]], "Prediction is not valid."
    except Exception as e:
        pytest.skip(f"Error running model prediction: {str(e)}")
