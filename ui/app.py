import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

st.title("Heart Disease Risk Predictor")
st.write("Enter patient data to predict the risk of heart disease using a trained ML model.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.model_loaded = False

def load_model():
    model_paths = [
        "models/final_model.pkl",
        "../models/final_model.pkl",
        "./final_model.pkl"
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                pipeline = joblib.load(path)
                st.session_state.pipeline = pipeline
                st.session_state.model_loaded = True
                return True, f"Model loaded from: {path}"
            except Exception as e:
                return False, f"Failed to load model at {path}: {e}"
    return False, "No model file found. Please train and save the model first."

if not st.session_state.model_loaded:
    success, message = load_model()
    if success:
        st.success(message)
    else:
        st.error(message)
        st.info("Make sure final_model.pkl exists in a models/ folder.")

st.header("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 55)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type", options=[
            (0, "Typical Angina"),
            (1, "Atypical Angina"),
            (2, "Non-anginal Pain"),
            (3, "Asymptomatic")
        ], format_func=lambda x: x[1])[0]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 246)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    with col2:
        restecg = st.selectbox("Resting ECG", options=[
            (0, "Normal"),
            (1, "ST-T Wave Abnormality"),
            (2, "Left Ventricular Hypertrophy")
        ], format_func=lambda x: x[1])[0]
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            (0, "Upsloping"),
            (1, "Flat"),
            (2, "Downsloping")
        ], format_func=lambda x: x[1])[0]
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[
            (1, "Normal"),
            (2, "Fixed Defect"),
            (3, "Reversible Defect")
        ], format_func=lambda x: x[1])[0]

    submitted = st.form_submit_button("Predict")

if submitted:
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please check the model file.")
    else:
        try:
            input_data = pd.DataFrame([{
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }])

            prediction = st.session_state.pipeline.predict(input_data)[0]
            probability = st.session_state.pipeline.predict_proba(input_data)[0][1]

            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("High Risk of Heart Disease")
                else:
                    st.success("Low Risk of Heart Disease")

            with col2:
                st.metric("Probability", f"{probability:.1%}")

            st.progress(float(probability))
            st.caption("Threshold = 50%. Above = High Risk, Below = Low Risk.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with st.expander("Debug Info"):
    st.write("Model loaded:", st.session_state.model_loaded)
    st.write("Working directory:", os.getcwd())
    st.write("Available model files:", [f for f in os.listdir('.') if f.endswith('.pkl') or f == 'models'])
