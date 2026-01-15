from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import csv
import os
from datetime import datetime

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

# Load model bundle
bundle = joblib.load("breast_cancer_pipeline.pkl")
model = bundle["model"]
ALL_FEATURES = bundle["features"]

LOG_FILE = "predictions.csv"

class Patient(BaseModel):
    radius_worst: float
    perimeter_worst: float
    concave_points_worst: float

@app.post("/predict")
def predict(data: Patient):

    # Create zero-filled row for all expected features
    row = {f: 0.0 for f in ALL_FEATURES}

    # Insert only the allowed 3
    row["radius_worst"] = data.radius_worst
    row["perimeter_worst"] = data.perimeter_worst
    row["concave_points_worst"] = data.concave_points_worst

    df = pd.DataFrame([row], columns=ALL_FEATURES)

    proba = model.predict_proba(df)[0]
    pred = int(model.predict(df)[0])

    prediction = "Malignant" if pred == 1 else "Benign"
    p_mal = round(float(proba[1]), 4)
    p_ben = round(float(proba[0]), 4)

    # ---------- LOG TO CSV ----------
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data.radius_worst,
            data.perimeter_worst,
            data.concave_points_worst,
            prediction,
            p_mal,
            datetime.utcnow().isoformat()
        ])

    return {
        "prediction": prediction,
        "probability_malignant": p_mal,
        "probability_benign": p_ben
    }

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return []

    df = pd.read_csv(LOG_FILE, header=None)
    df.columns = [
        "radius_worst",
        "perimeter_worst",
        "concave_points_worst",
        "prediction",
        "probability_malignant",
        "timestamp"
    ]
    return df.tail(50).to_dict(orient="records")
