# Turning Breast_Cancer_Classification into a streamlit app

# Importing Libraries
import streamlit as st
import pandas as pd 
from sklearn import datasets 
import joblib

# Load the model 
model = joblib.load('Breast_Cancer_Classification.pkl')

# Mapping class labels
class_names = ['Benign', 'Malignant']

# Writing title with a description 
st.write("""
# Breast Cancer Predictor 

This application uses a Logistic Regression model trained on the **Kaggle Breast Cancer Wisconsin dataset** to predict whether a tumor is **benign** or **malignant** based on key features.

Achieved an accuracy of **94%** on the test set.
""")

# Adding a sidebar for asthetics 
st.sidebar.header("User Input Parameters")

# Adding features 
# 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'   
def user_input_features():
    radius_mean = st.sidebar.slider('mean radius', 6.00, 28.50, 13.44) 
    texture_mean = st.sidebar.slider('texture mean', 9.50, 39.30, 23.37)
    perimeter_mean = st.sidebar.slider('perimeter mean', 43.5, 188.6, 105.51)
    area_mean = st.sidebar.slider('area mean', 143.0, 2501.0, 1256.77)
    smoothness_mean = st.sidebar.slider('smoothness_mean', 0.05263, 0.16340, 0.05263)
    data = {'radius_mean' : radius_mean,
            'texture_mean' : texture_mean,
            'perimeter_mean' : perimeter_mean,
            'area_mean' : area_mean,
            'smoothness_mean' : smoothness_mean}
    features = pd.DataFrame(data, index=[0])
    return features 

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Show Class Labels
st.subheader("Class labels and their corresponding index number")
st.write(pd.DataFrame(class_names, columns=["value"]))

# Predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Class labels
st.subheader("Class labels and their corresponding index number")
st.write(pd.DataFrame(class_names, columns=["value"]))

# Prediction
st.subheader("Prediction")
pred_df = pd.DataFrame({'value': [prediction[0]]})
st.write(pred_df)

# Prediction Probability
st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=class_names))