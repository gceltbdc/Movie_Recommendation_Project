# 🔧 Final app.py with all UI fixes and safe Unicode labels
# Make sure you have 'main_data.csv' in the same directory

import streamlit as st
import pandas as pd
import requests
import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "f7a140679c93b137c2879b1682284343"

# Load and process data
df = pd.read_csv("main_data.csv")
df = df.dropna(subset=['movie_title', 'comb'])

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['comb'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['movie_title'].str.strip().str.lower()).drop_duplicates()

# Fetch movie data from TMDB
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

# Log user feedback
def log_feedback(title, feedback_text, rating_value):
    log = {
        "movie_title": title,
        "rating": rating_value,
        "comment": feedback_text,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_log = pd.DataFrame([log])
    df_log.to_csv("feedback.csv", mode="a", header=not pd.io.common.file_exists("feedback.csv"), index=False)

# Streamlit setup
st.set_page_config(page_title="Movie Recommendations", layout="wide")
theme = st.sidebar.radio("Choose Theme:", ["Dark", "Light"])

# Style block
if theme == "Dark":
    bg_color = "#000"
    text_color = "#fff"
    card_bg = "rgba(0, 0, 0, 0.7)"
    label_color = "#d0aaff"
    sidebar_label_color = "#d0aaff"
else:
    bg_color = "#f5f5f5"
    text_color = "#111"
    card_bg = "#ffffff"
    label_color = "#6a1b9a"
    sidebar_label_color = "#4b0082"

# Inject custom CSS
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
    h1, h2, h3, .stMarkdown {{
        color: {text_color} !important;
    }}
    label, .css-1cpxqw2, .stRadio > div, .stSelectbox > div, .stMultiSelect > div {{
        color: {label_color} !important;
        font-weight: bold;
    }}
    section[data-testid="stSidebar"] label {{
        color: {sidebar_label_color} !important;
        font-weight: bold;
    }}
    .st-expanderHeader {{
        color: white !important;
    }}
    .st-expanderContent, .stTextArea label, .stSlider label {{
        color: white !important;
    }}
    .css-1r6slb0 {{
        color: white !important;
    }}
    textarea {{
        background-color: #222 !important;
        color: white !important;
    }}
    button:hover {{
        background-color: #7e57c2 !important;
        transform: scale(1.03);
        transition: all 0.3s ease;
    }}
    </style>
""", unsafe_allow_html=True)
