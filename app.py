import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎓 Student Performance Predictor")

st.write("Enter the student's information:")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

if st.button("Predict Result"):

    student = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=["study_hours", "attendance", "previous_score"]
    )

    prediction = model.predict(student)

    st.write("Prediction:", prediction[0])
