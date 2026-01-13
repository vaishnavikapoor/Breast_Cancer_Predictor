from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("breast_cancer_pipeline.pkl")

# Extract real training schema
ALL_FEATURES = list(model.feature_names_in_)
SELECTED = ["radius_worst", "perimeter_worst", "concave_points_worst"]

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

    return {
        "prediction": "Malignant" if pred == 1 else "Benign",
        "probability_malignant": round(float(proba[1]), 4),
        "probability_benign": round(float(proba[0]), 4)
    }
