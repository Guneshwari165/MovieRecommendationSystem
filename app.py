# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    # Ensure columns exist and no NaNs
    df['description'] = df.get('description', '').fillna('')
    df['genres'] = df.get('genres', '').fillna('')
    df['features'] = (df['genres'].astype(str) + " " + df['description'].astype(str))
    return df

@st.cache_data
def build_similarity_matrix(features_series):
    cv = CountVectorizer(stop_words='english')
    vectors = cv.fit_transform(features_series)
    sim = cosine_similarity(vectors)
    return sim

def get_recommendations(title, df, sim_matrix, top_n=5):
    # If title not in dataset, return empty list
    if title not in df['title'].values:
        return []
    idx = int(df[df['title'] == title].index[0])
    distances = sim_matrix[idx]
    # Get indices of top matches (excluding the movie itself)
    idx_sorted = np.argsort(distances)[::-1]
    idx_sorted = idx_sorted[idx_sorted != idx]  # remove itself
    top_indices = idx_sorted[:top_n]
    return df.iloc[top_indices].reset_index(drop=True)

# --- Load data and similarity matrix ---
movies = load_data("movies.csv")
similarity = build_similarity_matrix(movies['features'])

# --- UI ---
st.title("🎬 Movie Recommendation System (Content-based)")
st.write("Select a movie and click Recommend to see similar movies.")

col1, col2 = st.columns([2, 3])

with col1:
    movie_selected = st.selectbox("Choose a movie", movies['title'].tolist())
    n_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    recommend_btn = st.button("Recommend")

with col2:
    if recommend_btn:
        recommendations = get_recommendations(movie_selected, movies, similarity, top_n=n_recs)
        if recommendations.empty:
            st.warning("No recommendations found. Check your dataset or pick another title.")
        else:
            st.subheader("Recommended Movies")
            for i, row in recommendations.iterrows():
                st.markdown(f"{row['title']}**  \nGenres: {row.get('genres','')}")
                if 'description' in row:
                    st.write(row['description'])
                st.write("---")
    else:
        st.info("Pick a movie and press Recommend to see suggestions.")
