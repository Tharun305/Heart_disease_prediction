from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('rfc_heart_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form data
    features = np.array([float(request.form[feature]) for feature in [
        'Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'Cholesterol', 
        'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved', 
        'ExerciseInducedAngina', 'ST_Depression', 'ST_Slope', 
        'MajorVessels', 'Thalassemia'
    ]])

    # Reshape and scale the input data
    features = features.reshape(1, -1)
    scaled_data = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(scaled_data)[0]
    result = "Positive" if prediction == 1 else "Negative"

    # Render the result on the webpage
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
