import streamlit as st
import pandas as pd
import joblib
import numpy as np

import sklearn.compose._column_transformer as ct
st.set_page_config(
    page_title= "Health Diagnosis",
    layout="wide",

)

disease_map = {
    0: "Influenza",
    1: "COVID-19",
    2: "Dengue",
    3: "Malaria",
    4: "Pneumonia",
    5: "Normal"
}

# Patch missing class so joblib can load older pipelines
class _RemainderColsList(list):
    pass

# Inject into sklearn module so pickle can find it
ct._RemainderColsList = _RemainderColsList

import joblib
model = joblib.load("XGB_model_pipeline.pkl")


# Load model and encoder
label_encoder = joblib.load("label_encoder.pkl")
st.title("🩺 AI Medical Diagnosis Predictor")
st.write("Enter patient information below to get an AI-powered diagnosis prediction.")


# ------------------------ TAB 1: PATIENT INFO ------------------------
st.title("👤 Patient Info")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 1, 90, 30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])


# ------------------------ TAB 2: SYMPTOMS ------------------------
st.title("🤧 Symptoms")
st.markdown("### Select symptom severity")
severity_options = {
    "None": 0,
    "Low": 1,
    "Moderate": 2,
    "Severe": 3
}

def severity_input(label):
    choice = st.selectbox(label, list(severity_options.keys()))
    return severity_options[choice]

c1, c2, c3 = st.columns(3)
with c1:
    fever = severity_input("Fever")
    fatigue = severity_input("Fatigue")
    nausea = severity_input("Nausea")
    skin_rash = severity_input("Skin Rash")
with c2:
    cough = severity_input("Cough")
    headache = severity_input("Headache")
    vomiting = severity_input("Vomiting")
    loss_smell = severity_input("Loss of Smell")
with c3:
    muscle_pain = severity_input("Muscle Pain")
    diarrhea = severity_input("Diarrhea")
    loss_taste = severity_input("Loss of Taste")


# ------------------------ TAB 3: VITALS ------------------------
st.title("❤️ Vitals")
c1, c2 = st.columns(2)
with c1:
    systolic_bp = st.slider("Systolic BP", 80, 180, 120)
    heart_rate = st.slider("Heart Rate", 40, 150, 88)
    oxygen_saturation = st.slider("Oxygen Saturation (%)", 80, 100, 95)
with c2:
    diastolic_bp = st.slider("Diastolic BP", 50, 120, 80)
    temperature_c = st.slider("Body Temperature (°C)", 35, 41, 37)


# ------------------------ TAB 4: LAB TESTS ------------------------
st.title("🧪 Lab Tests")
c1, c2 = st.columns(2)
with c1:
    wbc_count = st.slider("WBC Count (x10⁹/L)", 2.0, 15.0, 7.0)
    hemoglobin = st.slider("Hemoglobin (g/dL)", 5.0, 18.0, 13.0)
    crp_level = st.slider("CRP Level (mg/L)", 0.5, 50.0, 10.0)
with c2:
    platelet_count = st.slider("Platelet Count (x10⁹/L)", 50.0, 450.0, 250.0)
    glucose_level = st.slider("Glucose Level (mg/dL)", 70.0, 200.0, 110.0)


st.markdown("## 🔍 Predict Disease")

if st.button("Predict Diagnosis"):

    # Create input dictionary
    input_data = {
        "age": age,
        "gender": gender,
        "fever": fever,
        "cough": cough,
        "fatigue": fatigue,
        "headache": headache,
        "muscle_pain": muscle_pain,
        "nausea": nausea,
        "vomiting": vomiting,
        "diarrhea": diarrhea,
        "skin_rash": skin_rash,
        "loss_smell": loss_smell,
        "loss_taste": loss_taste,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "temperature_c": temperature_c,
        "oxygen_saturation": oxygen_saturation,
        "wbc_count": wbc_count,
        "hemoglobin": hemoglobin,
        "platelet_count": platelet_count,
        "crp_level": crp_level,
        "glucose_level": glucose_level
    }

    input_df = pd.DataFrame([input_data])

    raw_pred = model.predict(input_df)[0]

    # Ensure integer
    pred_class = int(raw_pred)

    # Dictionary-based decoding
    predicted_label = disease_map.get(pred_class, "Unknown")

    st.success(f"🧾 **Predicted Diagnosis:** {predicted_label}")

    # Confidence scores
    probs = model.predict_proba(input_df)[0]
    pred_confidence = float(probs[pred_class] * 100)

    st.info(f"📊 **Confidence:** {pred_confidence:.2f}%")

    # All confidence scores as table
    prob_df = pd.DataFrame({
        "Disease": list(disease_map.values()),
        "Confidence (%)": (probs * 100).round(2)
    }).sort_values(by="Confidence (%)", ascending=False)

    st.subheader("All Confidence Scores")
    st.table(prob_df)


