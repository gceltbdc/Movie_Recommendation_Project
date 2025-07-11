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

# TF-IDF and Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['comb'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index map
indices = pd.Series(df.index, index=df['movie_title'].str.strip().str.lower()).drop_duplicates()

# TMDB API call
def fetch_tmdb_data(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            result = data['results'][0]
            poster_path = result.get('poster_path')
            overview = result.get('overview', 'No overview available.')
            release_date = result.get('release_date', '')
            year = release_date.split("-")[0] if release_date else "Unknown"
            trailer_url = None

            movie_id = result.get('id')
            if movie_id:
                video_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
                video_response = requests.get(video_url).json()
                for video in video_response.get('results', []):
                    if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                        trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                        break

            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            return poster_url, year, overview, trailer_url
    return None, "Unknown", "Overview not found.", None

# Recommend function
def recommend(title, genres=None, actors=None, sort_by="Similarity"):
    title = title.strip().lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
    results = []

    for i, score in sim_scores:
        movie = df.iloc[i]
        movie_title = movie['movie_title'].strip()

        genre_match = all(g in movie['genres'] for g in genres) if genres else True
        actor_match = any(a in str(movie[col]) for col in ['actor_1_name', 'actor_2_name', 'actor_3_name'] for a in actors) if actors else True

        if genre_match and actor_match:
            poster_url, year, overview, trailer_url = fetch_tmdb_data(movie_title)
            results.append({
                'title': movie_title,
                'poster': poster_url,
                'year': year,
                'overview': overview,
                'score': score,
                'trailer': trailer_url
            })

        if len(results) >= 10:
            break

    return sorted(results, key=lambda x: x['year'] if sort_by == "Year" else x['score'], reverse=True)

# ----- STREAMLIT UI -----
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Custom CSS for dark OTT background
st.markdown("""
    <style>
    body {
        background-color: #111;
        color: white;
    }
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/black-felt.png");
        background-size: cover;
    }
    .css-1v0mbdj, .css-1y4p8pa {
        color: white;
    }
    .stSelectbox > div > div {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍿 Netflix-Style Movie Recommender")
st.markdown("### Get smart recommendations with trailers, posters, and overview!")

movie_input = st.text_input("🎬 Enter a movie name", "")

# Multi-select filters
col1, col2, col3 = st.columns([3, 3, 2])

with col1:
    genre_set = sorted(set(g for genre in df['genres'].dropna() for g in genre.split()))
    selected_genres = st.multiselect("🎭 Filter by Genres", genre_set)

with col2:
    actor_set = pd.unique(df[['actor_1_name', 'actor_2_name', 'actor_3_name']].values.ravel('K'))
    actor_set = sorted([a for a in actor_set if pd.notna(a)])
    selected_actors = st.multiselect("🧑‍🎤 Filter by Actors", actor_set)

with col3:
    sort_by = st.radio("📊 Sort by", ["Similarity", "Year"])

# Recommend
if st.button("🚀 Recommend"):
    if movie_input:
        results = recommend(movie_input, selected_genres, selected_actors, sort_by)
        if results:
            for chunk in range(0, len(results), 5):
                row = results[chunk:chunk+5]
                cols = st.columns(len(row))
                for i, movie in enumerate(row):
                    with cols[i]:
                        st.image(movie['poster'] or "https://via.placeholder.com/300x450?text=No+Image", width=150)
                        st.markdown(f"**{movie['title']} ({movie['year']})**")
                        with st.expander("ℹ️ Overview"):
                            st.write(movie['overview'])
                        if movie['trailer']:
                            st.markdown(f"[▶️ Watch Trailer]({movie['trailer']})", unsafe_allow_html=True)
        else:
            st.warning("No recommendations found with those filters.")
    else:
        st.error("Please enter a movie title to get started.")
