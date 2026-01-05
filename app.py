import streamlit as st
import numpy as np
import joblib

# Load model, scaler, imputer
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
#imputer = joblib.load("model/imputer.pkl")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict heart disease risk.")

# Numerical inputs
age = st.number_input("Age", 1, 100)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
thalach = st.number_input("Maximum Heart Rate Achieved")
oldpeak = st.number_input("ST Depression")

# Step 1: Descriptive categorical inputs
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", 
                  ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
fbs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl", "> 120 mg/dl"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels (0–4)", [0,1,2,3,4])
thal = st.selectbox("Thalassemia", ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"])

# Step 2: Convert descriptive options to numeric
sex_map = {"Female": 0, "Male": 1}
cp_map = {"Typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3}
fbs_map = {"≤ 120 mg/dl":0, "> 120 mg/dl":1}
restecg_map = {"Normal":0, "ST-T Abnormality":1, "Left Ventricular Hypertrophy":2}
exang_map = {"No":0, "Yes":1}
slope_map = {"Upsloping":0, "Flat":1, "Downsloping":2}
thal_map = {"Unknown":0, "Normal":1, "Fixed Defect":2, "Reversible Defect":3}

sex = sex_map[sex]
cp = cp_map[cp]
fbs = fbs_map[fbs]
restecg = restecg_map[restecg]
exang = exang_map[exang]
slope = slope_map[slope]
thal = thal_map[thal]

st.info("Select values according to the description. The app will handle numeric conversion automatically.")

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    
    #input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

