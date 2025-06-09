from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('house_model.pkl')
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[f]) for f in feature_names]
    X_new = np.array([values])
    pred = model.predict(X_new)[0]
    price = pred * 100_000  # convert to actual dollars
    return render_template('index.html',
                           features=feature_names,
                           prediction_text=f'Estimated median house price: ${price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
