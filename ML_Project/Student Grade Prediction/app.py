from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('grade_module.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    G1 = float(request.form['G1'])
    G2 = float(request.form['G2'])
    studytime = float(request.form['studytime'])
    failures = float(request.form['failures'])
    absences = float(request.form['absences'])

    input_data = np.array([[G1, G2, studytime, failures, absences]])
    predicted_grade = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Final Grade: {predicted_grade:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
