import streamlit as st
import pandas as pd
import requests

API_BASE = "https://breast-cancer-predictor-4vi0.onrender.com"

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Risk Prediction System")
st.markdown("""
This app connects to a deployed **FastAPI service** for real-time cancer risk prediction  
and continuously logs model behavior for drift monitoring.
""")

st.sidebar.header("Patient Measurements")

def user_input():
    return {
        "radius_worst": st.sidebar.slider(
            "Tumor Radius (largest size of tumor)", 7.0, 36.0, 16.0
        ),
        "perimeter_worst": st.sidebar.slider(
            "Tumor Perimeter (boundary length)", 50.0, 250.0, 110.0
        ),
        "concave_points_worst": st.sidebar.slider(
            "Tumor Irregularity (how indented the edges are)", 0.0, 0.30, 0.10
        )
    }


data = user_input()

st.subheader("User Input")
st.write(pd.DataFrame([data]))

# ---------- PREDICTION ----------
if st.button("Predict"):
    try:
        res = requests.post(f"{API_BASE}/predict", json=data, timeout=30)
        result = res.json()

        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Malignant Probability: {round(result['probability_malignant']*100, 2)}%")
        st.info(f"Benign Probability: {round(result['probability_benign']*100, 2)}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.subheader("Model Monitoring Dashboard")

try:
    logs = requests.get(f"{API_BASE}/logs", timeout=20).json()

    if len(logs) == 0:
        st.info("No predictions logged yet.")
    else:
        df = pd.DataFrame(logs)

        df.rename(columns={
            "radius_worst": "Tumor Radius",
            "perimeter_worst": "Tumor Perimeter",
            "concave_points_worst": "Tumor Irregularity",
            "probability_malignant": "Malignancy Probability"
        }, inplace=True)

        st.write("Recent Predictions")
        st.dataframe(df)

        st.write("Malignancy Risk Over Time")
        import altair as alt

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Time of Prediction"),
                y=alt.Y("Malignancy Probability:Q", title="Probability of Malignancy"),
                tooltip=["timestamp", "Malignancy Probability"]
            )
            .properties(title="Malignancy Risk Drift Over Time")
        )

        st.altair_chart(chart, use_container_width=True)


except:
    st.warning("Monitoring service unavailable.")
