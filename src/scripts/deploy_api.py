from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the best model
MODEL_PATH = "models/random_forest_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class WaterQualityData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.post("/predict")
def predict(data: WaterQualityData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)
    potability = "Potable" if prediction[0] == 1 else "Non-Potable"
    
    return {"prediction": potability}