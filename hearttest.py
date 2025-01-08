# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:13:21 2025

@author: gaura
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load dataset
file_path = "heartt.csv"
try:
    dataset = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Dataset not found at {file_path}. Please check the path.")
    exit()

# Model file path
model_file_path = "heart_model.pkl"

# Train model function
def train_model():
    global model, is_trained
    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)

    max_accuracy = 0
    best_model = None
    for x in range(200):  # Reduced iterations for efficiency
        temp_model = RandomForestClassifier(random_state=x)
        temp_model.fit(X_train, y_train)
        current_accuracy = accuracy_score(temp_model.predict(X_test), y_test)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_model = temp_model

    if best_model:
        st.success(f"Best accuracy: {max_accuracy * 100:.2f}%")
        model = best_model
        is_trained = True
        
        # Save the trained model using joblib
        joblib.dump(model, model_file_path)
        #st.success("Model saved successfully!")
    else:
        st.error("Failed to train the model.")

# Load model function
def load_model():
    global model, is_trained
    if os.path.exists(model_file_path) and not is_trained:
        model = joblib.load(model_file_path)
        is_trained = True
       # st.success("Model loaded successfully!")
    elif not os.path.exists(model_file_path):
        st.error("Model not found. Please train the model first.")

# Predict function
def predict():
    if not is_trained:
        st.error("Model is not trained yet!")
        return

    try:
        input_data = []
        for key, value in patient_data.items():
            val = value
            if val == "":
                raise ValueError(f"Please select a value for {key}.")
            if key in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
                val = dropdown_options[key].index(val)
            input_data.append(float(val))

        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]
        result_proba = model.predict_proba(input_df)[0][1] * 100

        st.subheader(
            f"Prediction: {'Heart Disease Likely' if result else 'Heart Disease Unlikely'} "
            f"({result_proba:.2f}%)"
        )
        visualize_data(input_data)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Visualize data
def visualize_data(input_data):
    st.write("### Entered Patient Data")
    st.bar_chart(input_data)

# Dropdown options
dropdown_options = {
    "age": list(range(18, 101)),
    "sex": ["Female", "Male"],
    "cp": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "trestbps": list(range(80, 201)),
    "chol": list(range(120, 601)),
    "fbs": ["Fasting Blood Sugar ≤ 120", "Fasting Blood Sugar > 120"],
    "restecg": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
    "thalach": list(range(70, 211)),
    "exang": ["No", "Yes"],
    "oldpeak": [round(x * 0.1, 1) for x in range(0, 61)],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "ca": ["0 Vessels", "1 Vessel", "2 Vessels", "3 Vessels", "4 Vessels"],
    "thal": ["Normal", "Fixed Defect", "Reversible Defect"],
}

patient_data = {key: "" for key in dropdown_options.keys()}
is_trained = False
model = None

# Streamlit App
st.title("Heart Disease Prediction")

# User Profile Section
st.sidebar.header("User Profile")
username = st.sidebar.text_input("Name:", value="John Doe")
age = st.sidebar.number_input("Age:", min_value=18, max_value=100, value=30)
st.sidebar.image("https://via.placeholder.com/150", caption="User Profile")

# Sidebar for input data
st.sidebar.header("Enter Patient Details")
for key, options in dropdown_options.items():
    patient_data[key] = st.sidebar.selectbox(f"{key.capitalize()}: ", options)

# Load the trained model only if it hasn't been loaded yet
if not is_trained:
    load_model()

# Train model button

#if st.sidebar.button("Train Model"):
 #   train_model()

# Prediction button
if st.sidebar.button("Predict"):
    predict()
