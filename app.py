import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)


# Load Model, Scaler and Encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))


# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns
        numeric_columns = [
            'Hours_Studied',
            'Attendance',
            'Sleep_Hours',
            'Previous_Scores',
            'Tutoring_Sessions',
            'Physical_Activity'
        ]

        for col in numeric_columns:
            input_df[col] = pd.to_numeric(input_df[col])

        # Encode categorical columns using saved encoders
        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        exam_score = model.predict(input_scaled)[0]

        return render_template(
            'index.html',
            prediction_text=f"Predicted Exam Score: {round(exam_score, 2)}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# Run App
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
