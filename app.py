import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API Key
TMDB_API_KEY = "f7a140679c93b137c2879b1682284343"

# Load dataset
df = pd.read_csv("main_data.csv")
df = df.dropna(subset=['movie_title', 'comb'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['comb'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie Index Mapping
indices = pd.Series(df.index, index=df['movie_title'].str.strip().str.lower()).drop_duplicates()

# Fetch movie poster and release year from TMDB
def fetch_tmdb_data(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            release_date = data['results'][0].get('release_date', '')
            year = release_date.split("-")[0] if release_date else "Unknown"
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            return poster_url, year
    return None, "Unknown"

# Recommendation Logic
def recommend(title, selected_genre=None, selected_actor=None):
    title = title.strip().lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]

    recommended_movies = []
    for i, _ in sim_scores:
        movie = df.iloc[i]
        movie_title = movie['movie_title'].strip()
        genre_match = selected_genre in movie['genres'] if selected_genre else True
        actor_match = any(selected_actor in str(movie[col]) for col in ['actor_1_name', 'actor_2_name', 'actor_3_name']) if selected_actor else True

        if genre_match and actor_match:
            poster_url, year = fetch_tmdb_data(movie_title)
            recommended_movies.append((movie_title, poster_url, year))
        if len(recommended_movies) >= 5:
            break

    return recommended_movies

# ---------------- UI ----------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender")
st.markdown("Get personalized movie recommendations with posters, genre & actor filters!")

# Search Input
movie_input = st.text_input("Enter a movie name", "")

# Optional filters
genre_options = sorted(set(g for genre in df['genres'] for g in genre.split()))
actor_options = sorted(set(df['actor_1_name'].dropna().unique()) | set(df['actor_2_name'].dropna().unique()) | set(df['actor_3_name'].dropna().unique()))

col1, col2 = st.columns(2)
with col1:
    selected_genre = st.selectbox("Filter by Genre (Optional)", [""] + genre_options)
    selected_genre = selected_genre if selected_genre else None
with col2:
    selected_actor = st.selectbox("Filter by Actor (Optional)", [""] + actor_options)
    selected_actor = selected_actor if selected_actor else None

# Recommend Button
if st.button("Recommend"):
    if movie_input:
        recommendations = recommend(movie_input, selected_genre, selected_actor)
        if recommendations:
            st.subheader("Top Recommendations:")
            cols = st.columns(5)
            for i, (title, poster_url, year) in enumerate(recommendations):
                with cols[i % 5]:
                    st.image(poster_url or "https://via.placeholder.com/300x450?text=No+Image", width=150)
                    st.caption(f"**{title}** ({year})")
        else:
            st.warning("No recommendations found for your filters.")
    else:
        st.error("Please enter a movie name.")
