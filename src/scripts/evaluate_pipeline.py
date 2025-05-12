import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def evaluate_models():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    HOME = os.getcwd()
    DATA_FOLDER = HOME + "/data/"
    MODEL_FOLDER = HOME + "/models/"
    RESULTS_FOLDER = HOME + "/results/"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load test data
    logger.info("Loading test data...")
    data = pd.read_csv(os.path.join(DATA_FOLDER, "test_data.csv"))
    X_test = data.drop("Potability", axis=1)
    y_test = data["Potability"]

    # Load models
    model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    logger.info(f"Found {len(model_files)} models to evaluate")

    best_model = None
    best_model_name = None
    best_accuracy = 0
    results = {}

    for model_file in model_files:
        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(MODEL_FOLDER, model_file)

        logger.info(f"Evaluating {model_name}...")

        # Load model
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            # Some models might not have predict_proba
            y_pred_proba = y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(
            f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
        )

        # Store metrics
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    # Generate detailed report for best model
    logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    y_pred_best = best_model.predict(X_test)
    best_model_report = classification_report(y_test, y_pred_best, output_dict=True)

    # Save best model with timestamp
    best_model_timestamp_path = os.path.join(RESULTS_FOLDER, "best_model.pkl")
    with open(best_model_timestamp_path, "wb") as file:
        pickle.dump(best_model, file)

    # Save best model to standard path too (not using symlinks to ensure cross-platform compatibility)
    best_model_path = os.path.join(RESULTS_FOLDER, "best_model.pkl")
    with open(best_model_path, "wb") as file:
        pickle.dump(best_model, file)

    logger.info(
        f"Best model saved to {best_model_timestamp_path} and {best_model_path}"
    )

    # Save evaluation results
    evaluation_results = {
        "best_model": best_model_name,
        "timestamp": timestamp,
        "test_metrics": results,
        "best_model_detailed_report": best_model_report,
    }

    results_path = os.path.join(RESULTS_FOLDER, "evaluation_results.json")
    with open(results_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)

    logger.info(f"Evaluation results saved to {results_path}")

    # Extract feature importance if available
    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": X_test.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        importance_path = os.path.join(RESULTS_FOLDER, "feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")

    return best_model_name, best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ML models for water potability prediction"
    )
    args = parser.parse_args()

    evaluate_models()
