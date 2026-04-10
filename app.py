import streamlit as st
import numpy as np
import pickle

# ================== CONFIG ==================
st.set_page_config(page_title="AI Health Risk Predictor", layout="wide")

# ================== LOAD MODEL ==================
model = pickle.load(open("models/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ================== FUNCTIONS ==================

def yn(x):
    return 1 if x == "Yes" else 0


def risk_label(risk):
    if risk < 25:
        return "Low"
    elif risk < 60:
        return "Moderate"
    else:
        return "High"


def risk_color(label):
    return {"Low": "green", "Moderate": "orange", "High": "red"}[label]


def map_lifestyle(smoking, alcohol, junk, exercise):

    smoking_map = {
        "Never": 0,
        "Occasionally (1-2 days/week)": 1,
        "Regular (3-5 days/week)": 2,
        "Daily": 3
    }

    alcohol_map = {
        "Never": 0,
        "Social (1-2 times/week)": 1,
        "Moderate (3-5 times/week)": 2,
        "Heavy (Daily)": 3
    }

    junk_map = {
        "Rare (1-2 times/week)": 0,
        "Moderate (3-4 times/week)": 1,
        "Frequent (5+ times/week)": 2
    }

    exercise_map = {
        "None": 0,
        "1-2 days/week": 1,
        "3-5 days/week": 2,
        "Daily": 3
    }

    return (
        smoking_map[smoking],
        alcohol_map[alcohol],
        junk_map[junk],
        exercise_map[exercise]
    )


# ================== UI ==================

st.title("🧠 Lifestyle Disease Risk Predictor")

col1, col2 = st.columns(2)

# ================== INPUT ==================
with col1:
    st.subheader("📋 Enter Details")

    age = st.number_input("Age", 10, 100)
    bmi = st.number_input("BMI", 10.0, 50.0)
    
    # ================== BMI HELPER CALCULATOR ==================
st.markdown("#### 🧮 Calculate your BMI (Optional)")

col_bmi1, col_bmi2 = st.columns(2)

with col_bmi1:
    height = st.number_input("Height (cm)", 100.0, 220.0, key="height")

with col_bmi2:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, key="weight")

if height > 0:
    calc_bmi = weight / ((height / 100) ** 2)
    calc_bmi = round(calc_bmi, 2)

    st.info(f"👉 Your calculated BMI is: {calc_bmi}")

    # Category
    if calc_bmi < 18.5:
        st.warning("Underweight")
    elif calc_bmi < 25:
        st.success("Normal Weight")
    elif calc_bmi < 30:
        st.warning("Overweight")
    else:
        st.error("Obese")

    st.caption("⬆️ You can copy this BMI and enter it above")

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender == "Male" else 0

    smoking = st.selectbox(
        "Smoking Habit",
        ["Never", "Occasionally (1-2 days/week)", "Regular (3-5 days/week)", "Daily"]
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["Never", "Social (1-2 times/week)", "Moderate (3-5 times/week)", "Heavy (Daily)"]
    )

    junk = st.selectbox(
        "Junk Food Intake",
        ["Rare (1-2 times/week)", "Moderate (3-4 times/week)", "Frequent (5+ times/week)"]
    )

    exercise = st.selectbox(
        "Exercise Level",
        ["None", "1-2 days/week", "3-5 days/week", "Daily"]
    )

    sleep = st.slider("Sleep Hours", 0, 12)

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    depression = st.selectbox("Depression", ["No", "Yes"])

    if st.button("🚀 Predict Risk"):
        

        # Convert inputs
        smoking_val, alcohol_val, junk_val, exercise_val = map_lifestyle(
            smoking, alcohol, junk, exercise
        )
        
        


        input_data = np.array([[
            age,
            bmi,
            alcohol_val,
            exercise_val,
            gender_val,
            junk_val,
            sleep,
            smoking_val,
            yn(diabetes),
            yn(hypertension),
            yn(depression)
        ]])

        scaled = scaler.transform(input_data)

        # ================== ML PREDICTION ==================
        risk = model.predict_proba(scaled)[0][1] * 100

        # ================== STRONG HYBRID RISK ENGINE ==================

        # Disease impact
        if yn(diabetes) == 1:
            risk += 25

        if yn(hypertension) == 1:
            risk += 20

        if yn(depression) == 1:
            risk += 12

        # Lifestyle penalties
        risk += smoking_val * 8
        risk += alcohol_val * 6
        risk += junk_val * 6

        # Exercise impact
        if exercise_val == 0:
            risk += 20
        elif exercise_val == 1:
            risk += 10
        elif exercise_val == 2:
            risk += 3
        else:
            risk -= 5

        # Sleep impact
        if sleep < 5:
            risk += 15
        elif sleep < 6:
            risk += 8
        elif sleep > 9:
            risk += 5

        # BMI impact
        if bmi > 30:
            risk += 20
        elif bmi > 25:
            risk += 10

        # Age impact
        if age > 50:
            risk += 15
        elif age > 35:
            risk += 8

        # Final cap
        risk = max(5, min(risk, 95))
        risk = round(risk, 2)

        st.session_state.risk = risk

# ================== OUTPUT ==================
with col2:
    st.subheader("📊 Prediction Result")

    if "risk" in st.session_state:

        risk = st.session_state.risk
        label = risk_label(risk)
        color = risk_color(label)

        st.markdown(f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background-color:#111;
            border:3px solid {color};
        ">
            <h2 style="color:{color};">Risk: {risk}%</h2>
            <h3 style="color:white;">Level: {label}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Enter details and click Predict Risk")