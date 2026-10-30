import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy import stats


# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# Load saved models
@st.cache_resource
def load_models():
    return joblib.load("diabetes_model.pkl")


try:
    models = load_models()

except FileNotFoundError:
    st.error("Model file not found. Run diabetes_prediction.py first.")
    st.stop()


# Application heading
st.title("Diabetes Prediction System")

st.write(
    "A machine learning application designed to assess "
    "the likelihood of diabetes using patient health information."
)

st.divider()


# Patient information
st.subheader("Patient Information")

col1, col2 = st.columns(2)


with col1:

    glucose = st.number_input(
        "Glucose Level",
        min_value=1.0,
        value=120.0
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.1,
        value=25.0
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )


with col2:

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        value=0
    )

    dbf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5
    )


st.divider()


# Model selection
st.subheader("Select Machine Learning Model")

model_choice = st.selectbox(
    "Choose a prediction model",
    ["Logistic Regression", "Random Forest"]
)


# Prediction button
if st.button("Predict Diabetes", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame([{
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Pregnancies": pregnancies,
        "DiabetesPedigreeFunction": dbf
    }])


    # Apply original preprocessing
    input_data["Age"] = stats.boxcox(
        input_data["Age"],
        lmbda=models["age_lambda"]
    )

    input_data["Pregnancies"] = stats.yeojohnson(
        input_data["Pregnancies"],
        lmbda=models["preg_lambda"]
    )

    input_data["skewed_dbf"] = np.log1p(
        input_data["DiabetesPedigreeFunction"]
    )

    input_data.drop(
        "DiabetesPedigreeFunction",
        axis=1,
        inplace=True
    )


    # Arrange features in training order
    input_data = input_data[
        ["Glucose", "BMI", "Age", "Pregnancies", "skewed_dbf"]
    ]


    # Select model
    if model_choice == "Logistic Regression":

        model = models["logistic_regression"]

        scaled_input = models["scaler"].transform(input_data)

        prediction = model.predict(scaled_input)

        probability = model.predict_proba(scaled_input)[0][1]


    else:

        model = models["random_forest"]

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]


    # Display results
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction[0] == 1:

            st.error("Higher likelihood of diabetes")

        else:

            st.success("Lower likelihood of diabetes")


    with col2:

        st.metric(
            "Predicted Diabetes Probability",
            f"{probability * 100:.2f}%"
        )


    st.progress(float(probability))

    st.caption(
        "This prediction is an estimate from a machine learning model, "
        "not a medical diagnosis."
    )


st.divider()

st.caption("Diabetes Prediction System | Machine Learning Project")