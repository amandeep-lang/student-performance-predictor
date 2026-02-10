# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

MODEL_PATH = "best_model.pkl"  # default location (in repo)

# ---------- helper functions ----------
def attendance_bin_label(value):
    # same bins used in training
    bins = [-0.01, 0.6, 0.8, 0.9, 1.01]
    labels = ['poor','fair','good','excellent']
    return pd.cut(pd.Series([value]), bins=bins, labels=labels)[0]

def past_grade_bin_label(value):
    bins = [-1, 50, 65, 80, 100]
    labels = ['low','average','good','very_good']
    return pd.cut(pd.Series([value]), bins=bins, labels=labels)[0]

# ---------- model loading (try repo file first, then uploader) ----------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Found {MODEL_PATH} but failed to load it: {e}")

if model is None:
    st.info("No `best_model.pkl` found in app folder. You can upload it here (from your Colab export).")
    uploaded_file = st.file_uploader("Upload best_model.pkl (joblib file)", type=["pkl","joblib"])
    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.success("Model uploaded and loaded.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

if model is None:
    st.error("No model available. Please upload `best_model.pkl` or add it to the app folder in your GitHub repo.")
    st.stop()

# ---------- UI ----------
st.title("🎓 Student Performance Predictor")
st.write("Fill the student details and click **Predict**. (This app expects the same preprocessing/features as the training code.)")

with st.form("input_form"):
    gender = st.selectbox("Gender", ["male","female"])
    age = st.slider("Age", 15, 19, 17)

    parent_education = st.selectbox("Parent education", ["none","high_school","bachelor","master"])
    study_time_hours = st.slider("Study time (hours/week)", 0, 40, 10)
    past_failures = st.selectbox("Past failures", [0,1,2,3])
    absences = st.slider("Absences (days)", 0, 200, 3)
    past_grade = st.slider("Past grade", 0, 100, 70)

    extracurricular = st.selectbox("Extracurricular", ["yes","no"])
    health = st.slider("Health (1-5)", 1, 5, 3)
    internet = st.selectbox("Internet access", ["yes","no"])
    family_support = st.selectbox("Family support", ["yes","no"])

    homework_completion_rate = st.slider("Homework completion rate (0.0 - 1.0)", 0.0, 1.0, 0.8, step=0.01)
    attendance_rate = st.slider("Attendance rate (0.0 - 1.0)", 0.0, 1.0, 0.9, step=0.01)
    test_prep_course = st.selectbox("Test prep course", ["completed","none"])

    submitted = st.form_submit_button("🔮 Predict Final Grade")

# ---------- Prediction logic ----------
if submitted:
    # Create input row identical to training features
    try:
        study_intensity = None
        # compute engineered features exactly as in training code
        study_intensity = study_time_hours * homework_completion_rate
        attendance_bin = attendance_bin_label(attendance_rate)
        past_grade_bin = past_grade_bin_label(past_grade)

        # Build DataFrame with the same column names used in training
        input_dict = {
            'gender': gender,
            'age': age,
            'parent_education': parent_education,
            'study_time_hours': study_time_hours,
            'past_failures': past_failures,
            'absences': absences,
            'past_grade': past_grade,
            'extracurricular': extracurricular,
            'health': health,
            'internet': internet,
            'family_support': family_support,
            'homework_completion_rate': homework_completion_rate,
            'attendance_rate': attendance_rate,
            'test_prep_course': test_prep_course,
            # engineered features
            'study_intensity': study_intensity,
            'attendance_bin': attendance_bin,
            'past_grade_bin': past_grade_bin
        }

        input_df = pd.DataFrame([input_dict])

        # If the loaded model is a pipeline which already includes preprocessing
        # the pipeline will accept the raw input_df. Otherwise, the model
        # must accept the engineered/encoded features. We assume the training
        # pipeline included preprocessing (recommended).
        prediction = model.predict(input_df)
        if isinstance(prediction, (list, np.ndarray, pd.Series)):
            pred_value = float(prediction[0])
        else:
            pred_value = float(prediction)

        st.success(f"📊 Predicted final grade: **{pred_value:.2f} / 100**")

        if pred_value >= 85:
            st.info("🌟 Excellent performance expected!")
        elif pred_value >= 60:
            st.info("👍 Good performance expected.")
        else:
            st.warning("⚠️ Student may need additional support or intervention.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Common causes:")
        st.write("- Your `best_model.pkl` does not contain the preprocessing pipeline (so it expects different columns).")
        st.write("- The model was saved without the engineered columns. In that case re-save the full pipeline in Colab (instructions below).")
