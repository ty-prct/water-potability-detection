import os
import pandas as pd
import pickle
import json
import argparse
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HOME = os.getcwd()
# HOME = HOME[0:HOME.find("src")]
DATA_FOLDER = HOME + "data/"

data = pd.read_csv(DATA_FOLDER + "data.csv")

clean_data = data.copy()

# Handle missing values - fill with median for each column
for column in clean_data.columns:
    if clean_data[column].isnull().sum() > 0:
        median_value = clean_data[column].median()
        print(
            f"Filling missing values in {column} with median: {median_value:.4f}")
        clean_data[column] = clean_data[column].fillna(median_value)

# Export preprocessed data to CSV
export_path = DATA_FOLDER + "clean_data.csv"
clean_data.to_csv(export_path, index=False)
print(f"Clean dataset exported to {export_path}")

print(f"Clean dataset exported to {export_path}")
X = clean_data.drop('Potability', axis=1)
y = clean_data['Potability']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Create DataFrames for train and test sets
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Export train and test sets
train_df.to_csv(DATA_FOLDER + "train_data.csv", index=False)
test_df.to_csv(DATA_FOLDER + "test_data.csv", index=False)

print(f"Training dataset shape: {train_df.shape}")
print(f"Testing dataset shape: {test_df.shape}")
print(f"Training and testing datasets exported to {DATA_FOLDER}")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    DATA_FOLDER = "data/"
    MODEL_FOLDER = "models/"
    RESULTS_FOLDER = "results/"
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    data = pd.read_csv(os.path.join(DATA_FOLDER, "train_data.csv"))
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define models to train
    models = {
        "random_forest_model": RandomForestClassifier(n_estimators=100, random_state=42),
        "knn_model": KNeighborsClassifier(n_neighbors=5),
        "logistic_regression_model": LogisticRegression(random_state=42, max_iter=1000),
        "svm_model": SVC(probability=True, random_state=42),
        "xgboost_model": XGBClassifier(random_state=42),
        "lightgbm_model": LGBMClassifier(random_state=42)
    }

    # Train and evaluate models
    results = {}
    best_model_name = None
    best_score = 0
    feature_importance = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_valid)
        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(X_valid)[:, 1]
        except AttributeError:
            # Some models might not have predict_proba
            y_pred_proba = y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred)
        recall = recall_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        roc_auc = roc_auc_score(y_valid, y_pred_proba)

        # Store results
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        logger.info(
            f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_model_name = name

            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

        # Save model
        model_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {model_path}")

    # Save best model separately
    best_model = models[best_model_name]
    best_model_path = os.path.join(
        RESULTS_FOLDER, f"best_model_{timestamp}.pkl")
    with open(best_model_path, "wb") as file:
        pickle.dump(best_model, file)

    # Create a symlink to the best model
    best_model_symlink = os.path.join(RESULTS_FOLDER, "best_model.pkl")
    if os.path.exists(best_model_symlink):
        os.remove(best_model_symlink)
    os.symlink(best_model_path, best_model_symlink)

    # Save evaluation results
    evaluation_results = {
        "best_model": best_model_name,
        "timestamp": timestamp,
        "validation_metrics": results
    }

    results_path = os.path.join(
        RESULTS_FOLDER, f"evaluation_results_{timestamp}.json")
    with open(results_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)

    # Save feature importance if available
    if feature_importance is not None:
        feature_importance.to_csv(
            os.path.join(RESULTS_FOLDER,
                         f"feature_importance_{timestamp}.csv"),
            index=False
        )

    logger.info(
        f"Best model: {best_model_name} with accuracy {best_score:.4f}")
    logger.info(f"Evaluation results saved to {results_path}")
    logger.info(f"Best model saved to {best_model_path}")

    return best_model_name, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for water potability prediction")
    args = parser.parse_args()

    train_model()
