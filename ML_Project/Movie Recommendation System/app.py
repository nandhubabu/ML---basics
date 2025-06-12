from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load precomputed data
movies = joblib.load('movies.pkl')
cv = joblib.load('vectorizer.pkl')
similarity = joblib.load('similarity.pkl')

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found."]
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sorted_movies]

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommend(movie_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
