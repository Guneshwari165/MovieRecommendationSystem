import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies dataset
movies = pd.read_csv('movies.csv')

# Preprocess: Fill empty descriptions
movies['description'] = movies['description'].fillna('')

# Create TF-IDF matrix from descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI
st.title('🎬 Movie Recommendation System')

movie_title = st.selectbox('Select a movie you like:', movies['title'].values)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

if st.button('Show Recommendations'):
    recommendations = get_recommendations(movie_title)
    for index, row in recommendations.iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
        st.image(poster_url, width=150)
        st.write(f"**{row['title']}** | Genres: {row['genres']}")
