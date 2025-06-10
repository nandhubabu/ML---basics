from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])  # Use loaded vectorizer!
    prediction = model.predict(data)[0]
    result = "SPAM 🚫" if prediction == 1 else "NOT SPAM ✅"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
