# Diabetes Detection App

A machine learning project to predict diabetes risk using the Pima Indians Diabetes Dataset. Built with scikit-learn and deployed as a Streamlit web app.

## Features
- Predicts diabetes risk for individual patients.
- Supports batch predictions via CSV upload.
- Displays feature importance for Random Forest model.
- Deployed on Streamlit Sharing.

## Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run locally: `streamlit run app.py`.

## Dataset
- Source: [Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Files
- `app.py`: Streamlit app code.
- `diabetes_model.pkl`: Trained Random Forest model.
- `scaler.pkl`: StandardScaler for preprocessing.
- `requirements.txt`: Dependencies.

## Deployment
Deployed at: https://diabetesapp-yp8n4bdvb7nayfcgpspobz.streamlit.app/
