import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import joblib

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, left_on='id', right_on='movie_id')
movies = movies[['title_x', 'genres', 'keywords', 'cast', 'crew']]
movies.columns = ['title', 'genres', 'keywords', 'cast', 'crew']

def extract_names(obj, limit=3):
    try:
        return [i['name'] for i in ast.literal_eval(obj)][:limit]
    except:
        return []

def extract_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

# Process
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_names)
movies['crew'] = movies['crew'].apply(extract_director)
movies['soup'] = movies.apply(lambda x: ' '.join(x['genres'] + x['keywords'] + x['cast'] + x['crew']), axis=1)

# Vectorization and similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['soup']).toarray()
similarity = cosine_similarity(vectors)

# Save everything
joblib.dump(movies, 'movies.pkl')
joblib.dump(cv, 'vectorizer.pkl')
joblib.dump(similarity, 'similarity.pkl')
