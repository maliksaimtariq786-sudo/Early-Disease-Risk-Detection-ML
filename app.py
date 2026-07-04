# ============================================================
# Early Disease Risk Detection Using Machine Learning
# Streamlit Web Application
# Author: Malik Sayim
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Early Disease Risk Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model and Scaler ─────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("disease_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Custom CSS Styling ────────────────────────────────────────
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .risk-high {
        background-color: #ffe0e0;
        border-left: 6px solid #e63946;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #e63946;
    }
    .risk-low {
        background-color: #e0f7ea;
        border-left: 6px solid #2a9d8f;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #2a9d8f;
    }
    .info-box {
        background-color: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Early Disease Risk Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Diabetes Risk Assessment System — Enter patient clinical data below</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar — About ───────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
    st.markdown("## 📋 About This App")
    st.info("""
    This application uses a **Machine Learning model** trained on the 
    Pima Indians Diabetes Dataset to predict a patient's risk of diabetes 
    based on clinical measurements.
    
    **Model:** Random Forest Classifier  
    **Dataset:** 768 patients  
    **Source:** UCI / Kaggle  
    """)

    st.markdown("## 🎯 How to Use")
    st.write("""
    1. Enter patient details in the form
    2. Click **Predict Risk** button
    3. View the risk assessment result
    """)

    st.markdown("## ⚠️ Disclaimer")
    st.warning("""
    This tool is for **educational purposes only** and should not replace 
    professional medical diagnosis.
    """)

    st.markdown("---")
    st.markdown("**Developed by:** Malik Sayim")
    st.markdown("**Program:** BS Artificial Intelligence")
    st.markdown("**Institution:** COMSATS University Islamabad")

# ── Input Form ────────────────────────────────────────────────
st.markdown("## 🩺 Patient Clinical Information")
st.markdown("Fill in all fields below with the patient's measurements:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Personal Info")
    age = st.number_input(
        "Age (years)",
        min_value=18, max_value=90, value=35,
        help="Patient age in years"
    )
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=20, value=1,
        help="Number of times pregnant"
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0, max_value=70.0, value=28.0, step=0.1,
        help="Weight in kg / (height in m)²"
    )

with col2:
    st.markdown("### 🩸 Blood Measurements")
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=50, max_value=300, value=100,
        help="Plasma glucose concentration (2hr oral glucose test)"
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=30, max_value=150, value=72,
        help="Diastolic blood pressure"
    )
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0, max_value=900, value=80,
        help="2-hour serum insulin"
    )

with col3:
    st.markdown("### 🧬 Other Measurements")
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=5, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05, max_value=3.0, value=0.47, step=0.01,
        help="Genetic diabetes likelihood score based on family history"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_btn = st.button("🔍 Predict Diabetes Risk", 
                             use_container_width=True,
                             type="primary")

# ── Prediction Logic ──────────────────────────────────────────
if predict_btn:
    # Prepare input features (match training order)
    glucose_risk = 0 if glucose < 140 else (1 if glucose < 200 else 2)
    bmi_cat      = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
    age_group    = 0 if age < 30 else (1 if age < 45 else (2 if age < 60 else 3))
    glucose_bmi  = glucose * bmi

    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age,
        glucose_risk, bmi_cat, age_group, glucose_bmi
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.markdown("## 📊 Risk Assessment Result")

    col_r1, col_r2 = st.columns([1.5, 1])

    with col_r1:
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
            ⚠️ HIGH RISK — Diabetes Detected<br>
            <span style="font-size:1rem; font-weight:normal;">
            This patient shows clinical indicators associated with diabetes. 
            Immediate medical consultation is strongly recommended.
            </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
            ✅ LOW RISK — No Diabetes Detected<br>
            <span style="font-size:1rem; font-weight:normal;">
            Current clinical indicators suggest low diabetes risk. 
            Continue regular health monitoring.
            </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("#### 📈 Prediction Confidence")
        st.progress(float(probability))
        st.markdown(f"**Diabetes Risk Probability: `{probability*100:.1f}%`**")

    with col_r2:
        st.markdown("#### 📋 Patient Summary")
        st.markdown(f"""
        <div class="info-box">
        <b>Age:</b> {age} years<br>
        <b>BMI:</b> {bmi}<br>
        <b>Glucose:</b> {glucose} mg/dL<br>
        <b>Blood Pressure:</b> {blood_pressure} mm Hg<br>
        <b>Insulin:</b> {insulin} mu U/ml<br>
        <b>Pregnancies:</b> {pregnancies}<br>
        <b>Skin Thickness:</b> {skin_thickness} mm<br>
        <b>Pedigree Function:</b> {dpf}
        </div>
        """, unsafe_allow_html=True)

    # Clinical interpretation
    st.markdown("---")
    st.markdown("#### 🔍 Clinical Interpretation")

    interp_col1, interp_col2, interp_col3 = st.columns(3)

    with interp_col1:
        glucose_status = "🔴 High" if glucose > 140 else ("🟡 Borderline" if glucose > 100 else "🟢 Normal")
        st.metric("Glucose Status", glucose_status, f"{glucose} mg/dL")

    with interp_col2:
        bmi_status = "🔴 Obese" if bmi >= 30 else ("🟡 Overweight" if bmi >= 25 else "🟢 Normal")
        st.metric("BMI Status", bmi_status, f"{bmi}")

    with interp_col3:
        bp_status = "🔴 High" if blood_pressure > 90 else ("🟡 Elevated" if blood_pressure > 80 else "🟢 Normal")
        st.metric("Blood Pressure", bp_status, f"{blood_pressure} mm Hg")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
Early Disease Risk Detection System | Developed by Malik Sayim | 
BS Artificial Intelligence, COMSATS University Islamabad | 
Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)