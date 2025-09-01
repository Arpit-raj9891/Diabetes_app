import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title('Diabetes Detection App')
st.write('Enter patient details to predict diabetes risk. Note: This is educational only.')

# Input fields (matching dataset features)
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0.0, max_value=200.0, value=100.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=150.0, value=70.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input('Insulin', min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=0, max_value=100, value=30)

# Predict button
if st.button('Predict'):
    # Create input array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # Probability of diabetes
    
    # Display result
    if prediction == 1:
        st.error(f'High risk of diabetes (Probability: {prob:.2%})')
    else:
        st.success(f'Low risk of diabetes (Probability: {prob:.2%})')