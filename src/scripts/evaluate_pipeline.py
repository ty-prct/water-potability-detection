import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"
RESULTS_FOLDER = "results/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load test data
data = pd.read_csv(os.path.join(DATA_FOLDER, "test_data.csv"))
X_test = data.drop("Potability", axis=1)
y_test = data["Potability"]

# Load models
model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]

best_model = None
best_accuracy = 0

for model_file in model_files:
    model_path = os.path.join(MODEL_FOLDER, model_file)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {model_file}, Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_file

# Save evaluation results
results_path = os.path.join(RESULTS_FOLDER, "evaluation_results.txt")
with open(results_path, "w") as file:
    file.write(f"Best Model: {best_model_name}\n")
    file.write(f"Accuracy: {best_accuracy:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(y_test, best_model.predict(X_test)))

print(f"Evaluation results saved to {results_path}")