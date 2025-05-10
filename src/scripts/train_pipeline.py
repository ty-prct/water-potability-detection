import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Paths
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load data
data = pd.read_csv(os.path.join(DATA_FOLDER, "clean_data.csv"))
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model
model_path = os.path.join(MODEL_FOLDER, "random_forest_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)
print(f"Model saved to {model_path}")