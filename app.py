import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

train_data_path = "templates/Training.csv"
test_data_path = "templates/Testing.csv"

# Load training data
df = pd.read_csv(train_data_path)
cols = df.columns[:-1]  # Features columns
x = df[cols]
y = df['prognosis']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialize and train DecisionTreeClassifier
dt = DecisionTreeClassifier()
clf_dt = dt.fit(x_train, y_train)

# Load symptoms from the Testing CSV for the dropdown options
test_df = pd.read_csv(test_data_path)
symptoms = test_df.columns[:-1].tolist()

# Dictionary for symptom to index mapping
symptom_indices = dict(zip(df.columns[:-1], range(len(df.columns[:-1]))))

# Function to predict the disease
def predict_disease(selected_symptoms):
    input_vector = [0] * len(symptoms)
    for symptom in selected_symptoms:
        if symptom in symptom_indices:
            idx = symptom_indices[symptom]
            input_vector[idx] = 1
    input_vector = np.array(input_vector).reshape(1, -1)
    prediction = dt.predict(input_vector)
    return prediction[0]

# Streamlit UI

st.title("Disease Prediction App")
st.write("Select your symptoms to predict the disease.")

# Dropdowns for symptom selection
selected_symptoms = []
symptom_count = 5  # Number of symptoms to select

for i in range(symptom_count):
    selected_symptom = st.selectbox(f"Select Symptom {i+1}", [""] + symptoms, key=f"symptom_{i+1}")
    if selected_symptom:
        selected_symptoms.append(selected_symptom)

# Predict disease when the button is clicked
if st.button("Predict Disease"):
    if selected_symptoms:
        disease_prediction = predict_disease(selected_symptoms)
        st.success(f"Predicted Disease: {disease_prediction}")
    else:
        st.warning("Please select at least one symptom.")
