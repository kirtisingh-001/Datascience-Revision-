from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Input format
class PredictionRequest(BaseModel):
    features: list[float]

# Home route
@app.get("/")
def home():
    return {"status": "running"}

# Prediction route
@app.post("/predict")
def predict(req: PredictionRequest):

    prediction = model.predict([req.features])

    return {
        "prediction": int(prediction[0])
    }