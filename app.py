import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and scaler
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

# App title and description
st.title('Diabetes Detection App')
st.write('Enter patient details to predict diabetes risk or upload a CSV file for batch predictions. Note: This is for educational purposes only.')

# Sidebar for mode selection
mode = st.sidebar.selectbox('Select Mode', ['Single Prediction', 'Batch Prediction'])

if mode == 'Single Prediction':
    # Input fields with validation
    st.subheader('Enter Patient Details')
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose', min_value=0.0, max_value=200.0, value=100.0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=150.0, value=70.0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input('Insulin', min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=100, value=30)

    # Input validation
    if glucose == 0 or bmi == 0 or blood_pressure == 0:
        st.warning("Warning: Glucose, BMI, and Blood Pressure cannot be 0 for realistic predictions.")

    # Predict button
    if st.button('Predict'):
        # Create input array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Display result
        if prediction == 1:
            st.error(f'High risk of diabetes (Probability: {prob:.2%})')
        else:
            st.success(f'Low risk of diabetes (Probability: {prob:.2%})')

        # Feature importance chart (for Random Forest)
        if hasattr(model, 'feature_importances_'):
            st.subheader('Feature Importance')
            features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
            importance = model.feature_importances_
            fig, ax = plt.subplots()
            ax.bar(features, importance, color='teal')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif mode == 'Batch Prediction':
    # File uploader for batch predictions
    st.subheader('Upload CSV for Batch Predictions')
    st.write('CSV must have columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age')
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    
    if uploaded_file is not None:
        # Read and process file
        df = pd.read_csv(uploaded_file)
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Validate columns
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
        else:
            # Replace zeros with NaN and impute with median (as in training)
            df[required_columns[1:6]] = df[required_columns[1:6]].replace(0, np.nan)
            df.fillna(df.median(), inplace=True)
            
            # Scale and predict
            X = df[required_columns]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            # Add predictions to dataframe
            df['Prediction'] = predictions
            df['Diabetes Probability'] = probabilities
            df['Prediction'] = df['Prediction'].map({1: 'High Risk', 0: 'Low Risk'})
            
            # Display results
            st.write('Prediction Results:')
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button('Download Predictions', csv, 'predictions.csv', 'text/csv')
        st.success(f'Low risk of diabetes (Probability: {prob:.2%})')
