# 🎓 Student Performance Predictor

A beginner-friendly Machine Learning application that predicts whether a student is likely to **Pass or Fail** based on their study hours, attendance, and previous exam score.

This project was built to understand the complete basic Machine Learning workflow — from working with a dataset and training a model to using the trained model inside a web application.

---

## 📌 Project Overview

The application takes three inputs:

- 📚 Study Hours
- 📊 Attendance Percentage
- 📝 Previous Exam Score

A trained **Decision Tree Classifier** then predicts whether the student is likely to:

- `Pass`
- `Fail`

The prediction is displayed through a simple web interface built with **Streamlit**.

---

## 🧠 Machine Learning Workflow

The project follows this basic Machine Learning pipeline:

```text
Student Dataset
      ↓
Load data using Pandas
      ↓
Separate Features (X) and Target (y)
      ↓
Train/Test Split
      ↓
Train Decision Tree
      ↓
Evaluate Model
      ↓
Save Trained Model
      ↓
Load Model in Streamlit
      ↓
User Enters Student Information
      ↓
Model Makes Prediction
      ↓
Pass / Fail
