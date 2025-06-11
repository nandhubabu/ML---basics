from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trtbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalachh']),
        float(request.form['exng']),
        float(request.form['oldpeak']),
        float(request.form['slp']),
        float(request.form['caa']),
        float(request.form['thall'])
    ]

    # Scale
    final_data = scaler.transform([features])  # now it has 13 features

    # Predict
    prediction = model.predict(final_data)[0]
    result = "Has Heart Disease ❤️" if prediction == 1 else "No Heart Disease 💖"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
