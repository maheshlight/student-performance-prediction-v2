import streamlit as st
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
        .main { background-color: #f0f4ff; }
        .stButton>button {
            background-color: #4f46e5;
            color: white;
            border-radius: 10px;
            padding: 0.6em 2em;
            font-size: 1rem;
            font-weight: 600;
            border: none;
        }
        .stButton>button:hover { background-color: #3730a3; }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .pass { background-color: #d1fae5; color: #065f46; border: 2px solid #10b981; }
        .fail { background-color: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🎓 Student Performance Predictor")
st.markdown("Enter student details below to predict whether the student will **Pass** or **Fail**.")
st.markdown("---")

# ─────────────────────────────────────────────
# Load or train model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("student_model.pkl"):
        model = joblib.load("student_model.pkl")
    else:
        # Fallback: train a simple model with sample data
        import pandas as pd
        data = {
            "study_hours":      [5, 2, 7, 3, 6, 1, 8, 4, 5, 2],
            "attendance":       [80, 50, 90, 60, 85, 40, 95, 70, 75, 55],
            "previous_scores":  [60, 40, 85, 55, 70, 35, 90, 65, 60, 45],
            "pass_fail":        [1,  0,  1,  0,  1,  0,  1,  1,  1,  0]
        }
        df = pd.DataFrame(data)
        X = df[["study_hours", "attendance", "previous_scores"]]
        y = df["pass_fail"]
        model = LogisticRegression()
        model.fit(X, y)
    return model

@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    else:
        # Use same sample data to fit scaler
        data = np.array([
            [5, 80, 60], [2, 50, 40], [7, 90, 85], [3, 60, 55],
            [6, 85, 70], [1, 40, 35], [8, 95, 90], [4, 70, 65],
            [5, 75, 60], [2, 55, 45]
        ])
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler

model = load_model()
scaler = load_scaler()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "📚 Study Hours (per day)",
        min_value=0.0, max_value=24.0,
        value=5.0, step=0.5
    )

with col2:
    attendance = st.number_input(
        "🏫 Attendance (%)",
        min_value=0.0, max_value=100.0,
        value=75.0, step=1.0
    )

with col3:
    previous_scores = st.number_input(
        "📝 Previous Score (%)",
        min_value=0.0, max_value=100.0,
        value=60.0, step=1.0
    )

st.markdown("")

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("🔍 Predict Result"):
    input_data = np.array([[study_hours, attendance, previous_scores]])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
    except Exception:
        # If scaler fails, predict without scaling
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.markdown(
            f'<div class="result-box pass">✅ PASS &nbsp;|&nbsp; Confidence: {probability[1]*100:.1f}%</div>',
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            f'<div class="result-box fail">❌ FAIL &nbsp;|&nbsp; Confidence: {probability[0]*100:.1f}%</div>',
            unsafe_allow_html=True
        )

    # Show input summary
    st.markdown("### 📊 Input Summary")
    st.markdown(f"- **Study Hours:** {study_hours} hrs/day")
    st.markdown(f"- **Attendance:** {attendance}%")
    st.markdown(f"- **Previous Score:** {previous_scores}%")

st.markdown("---")
st.markdown(
    "<center><small>Built by Mahesh | Student Performance Prediction v2</small></center>",
    unsafe_allow_html=True
)
