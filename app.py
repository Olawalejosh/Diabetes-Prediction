import streamlit as st
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please train and save your model first.")
    st.stop()

# Application title
st.title("Diabetes Prediction System")

st.write(
    "This application uses machine learning to predict "
    "whether an individual is likely to have diabetes."
)

st.divider()

# User input fields
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, value=79)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

# Prediction button
if st.button("Predict Diabetes", use_container_width=True):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Prediction: High likelihood of diabetes.")
    else:
        st.success("Prediction: Low likelihood of diabetes.")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.metric("Predicted Diabetes Probability", f"{probability * 100:.2f}%")

st.divider()

st.caption("Diabetes Prediction System | Machine Learning Project")
st.caption("This application is for educational purposes and is not a medical diagnosis.")