# Final regenerated working app.py for Streamlit Movie Recommender with Purple Font, IMDb Links, Hover Effects & Genre Sections

import streamlit as st
import pandas as pd
import requests
import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# TMDB API key
TMDB_API_KEY = "f7a140679c93b137c2879b1682284343"

# Load data
df = pd.read_csv("main_data.csv")
df = df.dropna(subset=['movie_title', 'comb'])

# TF-IDF & Cosine similarity setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['comb'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['movie_title'].str.strip().str.lower()).drop_duplicates()

# Fetch TMDB details
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
            imdb_url = f"https://www.imdb.com/find?q={urllib.parse.quote(title)}"
            return poster_url, year, overview, trailer_url, imdb_url
    return None, "Unknown", "Overview not found.", None, "https://www.imdb.com"

# Recommend movies
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
            poster_url, year, overview, trailer_url, imdb_url = fetch_tmdb_data(movie_title)
            results.append({
                'title': movie_title,
                'poster': poster_url,
                'year': year,
                'overview': overview,
                'score': score,
                'trailer': trailer_url,
                'imdb': imdb_url,
                'genre': movie['genres'].split()[0] if pd.notna(movie['genres']) else "Misc"
            })
        if len(results) >= 20:
            break
    return sorted(results, key=lambda x: x['year'] if sort_by == "Year" else x['score'], reverse=True)

# Inject hover style & color override
st.markdown("""
    <style>
    body {
        color: #a864e3;
    }
    a {
        color: #a864e3 !important;
        text-decoration: none;
    }
    a:hover {
        color: #cba0f8 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Theme toggle and layout
st.set_page_config(page_title="Movie Recommendations", layout="wide")
theme = st.sidebar.radio("Choose Theme:", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#000"
    text_color = "#a864e3"
    card_bg = "rgba(0, 0, 0, 0.7)"
    label_color = "#d0aaff"
    sidebar_label_color = "#d0aaff"
else:
    bg_color = "#f5f5f5"
    text_color = "#4b0082"
    card_bg = "#ffffff"
    label_color = "#6a1b9a"
    sidebar_label_color = "#4b0082"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    background-image: url('https://images.unsplash.com/photo-1497032628192-86f99bcd76bc?auto=format&fit=crop&w=1470&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: {text_color};
}}
.stApp {{
    background-color: {card_bg};
    padding: 2rem;
    border-radius: 12px;
    backdrop-filter: blur(8px);
}}
label, section[data-testid="stSidebar"] label {{
    color: {sidebar_label_color} !important;
    font-weight: bold;
}}
.st-expanderHeader, .st-expanderContent, .stTextArea label, .stSlider label, textarea {{
    color: {text_color} !important;
    background-color: #222 !important;
}}
h1, h2, h3, .stMarkdown, .stButton, .stTextInput, .stSelectbox, .stMultiSelect, .stRadio {{
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# Trending with IMDb links
st.markdown("<h1 style='text-align:center'>🎬 Recommending Movies for you 😉</h1>", unsafe_allow_html=True)
st.subheader("🔥 Trending Picks")
trending_titles = random.sample(df['movie_title'].dropna().unique().tolist(), 5)
trending_cols = st.columns(5)
for i, title in enumerate(trending_titles):
    with trending_cols[i]:
        poster, year, overview, trailer, imdb = fetch_tmdb_data(title)
        st.image(poster or "https://via.placeholder.com/300x450?text=No+Image", width=150)
        st.markdown(f"[{title} ({year})]({imdb})", unsafe_allow_html=True)
        if trailer:
            st.markdown(f"[▶️ Trailer]({trailer})", unsafe_allow_html=True)

# Search and filters
search_input = st.text_input("Search for a movie you like")
genre_options = sorted(set(g for genre in df['genres'].dropna() for g in genre.split()))
actor_set = pd.unique(df[['actor_1_name', 'actor_2_name', 'actor_3_name']].values.ravel('K'))
actor_options = sorted([a for a in actor_set if pd.notna(a)])
selected_genres = st.sidebar.multiselect("Genres", genre_options)
selected_actors = st.sidebar.multiselect("Actors", actor_options)
sort_by = st.sidebar.radio("Sort by", ["Similarity", "Year"])

# Log feedback
def log_feedback(title, feedback_text, rating_value):
    log = {
        "movie_title": title,
        "rating": rating_value,
        "comment": feedback_text,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_log = pd.DataFrame([log])
    df_log.to_csv("feedback.csv", mode="a", header=not pd.io.common.file_exists("feedback.csv"), index=False)

# Recommendations section
if st.button("Recommend"):
    if search_input:
        with st.spinner("Fetching recommendations..."):
            results = recommend(search_input, selected_genres, selected_actors, sort_by)
        if results:
            st.subheader("🎯 Recommendations:")
            genre_groups = {}
            for movie in results:
                genre = movie['genre']
                genre_groups.setdefault(genre, []).append(movie)
            for genre, movies in genre_groups.items():
                st.markdown(f"### 🎭 {genre.title()} Movies")
                for chunk in range(0, len(movies), 5):
                    row = movies[chunk:chunk+5]
                    cols = st.columns(len(row))
                    for i, movie in enumerate(row):
                        with cols[i]:
                            st.image(movie['poster'] or "https://via.placeholder.com/300x450?text=No+Image", width=150)
                            st.markdown(f"[{movie['title']} ({movie['year']})]({movie['imdb']})", unsafe_allow_html=True)
                            with st.expander("ℹ️ Overview"):
                                st.write(movie['overview'])
                            if movie['trailer']:
                                st.markdown(f"[▶️ Watch Trailer]({movie['trailer']})", unsafe_allow_html=True)
                            share_text = f"I recommend watching '{movie['title']}'! Check it out 🎬"
                            twitter = f"https://twitter.com/intent/tweet?text={share_text}"
                            whatsapp = f"https://wa.me/?text={share_text}"
                            st.markdown(f"[📤 Share on Twitter]({twitter})", unsafe_allow_html=True)
                            st.markdown(f"[📤 Share on WhatsApp]({whatsapp})", unsafe_allow_html=True)
                            with st.expander("📝 Leave a comment or rating"):
                                rating = st.slider("Rating", 0, 5, step=1, key=f"rate_{movie['title']}")
                                comment = st.text_area("Your thoughts:", key=f"comment_{movie['title']}")
                                if st.button("Submit Feedback", key=f"submit_{movie['title']}"):
                                    log_feedback(movie['title'], comment, rating)
                                    st.success("Thank you for your feedback!")
        else:
            st.warning("No results matched your filters.")
    else:
        st.error("Please enter a movie title to begin.")
