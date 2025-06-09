from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = 1 if request.form['sex'] == 'male' else 0
    age = float(request.form['age'])
    fare = float(request.form['fare'])

    input_data = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(input_data)

    result = "Survived 🛟" if prediction[0] == 1 else "Did not survive ⚰️"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
