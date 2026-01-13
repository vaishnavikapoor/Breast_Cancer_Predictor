import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Risk Prediction System")
st.markdown("""
This app connects to a deployed **FastAPI service** for real-time cancer risk prediction  
and continuously logs model behavior for drift monitoring.
""")

st.sidebar.header("Patient Measurements")

def user_input():
    radius_worst = st.sidebar.slider('Radius (worst)', 7.0, 36.0, 16.0)
    perimeter_worst = st.sidebar.slider('Perimeter (worst)', 50.0, 250.0, 110.0)
    concave_points_worst = st.sidebar.slider('Concave Points (worst)', 0.0, 0.30, 0.10)

    return {
        "radius_worst": radius_worst,
        "perimeter_worst": perimeter_worst,
        "concave_points_worst": concave_points_worst
    }

data = user_input()
st.subheader("User Input")
st.write(pd.DataFrame([data]))

# ---------- API CALL ----------
if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        prob = response.json()["cancer_probability"]
        st.success(f"Malignancy Probability: {prob}")

    else:
        st.error("API Error – ensure FastAPI server is running.")

# ---------- STEP 4 : MODEL MONITORING ----------
st.markdown("---")
st.subheader("Model Monitoring Dashboard")

try:
    logs = pd.read_csv("predictions.csv", header=None)
    logs.columns = list(data.keys()) + ["prediction", "timestamp"]

    st.write("Recent Predictions")
    st.dataframe(logs.tail())

    st.write("Prediction Drift Over Time")
    st.line_chart(logs["prediction"])

except:
    st.info("No logs yet. Run predictions to generate monitoring data.")
