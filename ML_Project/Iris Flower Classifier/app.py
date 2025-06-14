from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]
    return render_template('index.html', prediction_text=f'Predicted species: {result}')

if __name__ == '__main__':
    app.run(debug=True)
