#Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import joblib

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'student_performance.csv')
data = pd.read_csv(file_path)

print(data.head())  # Display the first few rows of the dataset

print(data.dtypes)  # Check the data types of each column

print(data.shape)  # Check the shape of the dataset (number of rows and columns)

# Preprocess the data
print(data.isnull().sum())  # Check for missing values
data = data.dropna()  # Drop rows with missing values
data=data.drop_duplicates()  # Remove duplicate rows

# Encode categorical variables 
encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le=LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le  # Store the encoder for future use


print(data.head())  # Display the first few rows after encoding
print(data.dtypes)  # Check the data types after encoding
print(data.shape)  # Check the shape of the dataset after preprocessing

# Define features and target variable
X = data.drop('Exam_Score', axis=1)  # Features
y= data['Exam_Score']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {np.sqrt(mse)}")
print(f"R-squared Score: {r2}")

# Save the model and scaler
joblib.dump(model, os.path.join(BASE_DIR, 'model.pkl'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
joblib.dump(encoders, os.path.join(BASE_DIR, 'encoders.pkl'))