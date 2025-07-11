import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load your main dataset
data = pd.read_csv("main_data.csv")

# Preprocess: Drop rows with missing movie_title or features
data = data.dropna(subset=['movie_title', 'genres', 'plot_keywords'])

# Combine relevant text features
def combine_features(row):
    return str(row['genres']) + " " + str(row['plot_keywords'])

data['combined_features'] = data.apply(combine_features, axis=1)

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a reverse index of movie titles
indices = pd.Series(data.index, index=data['movie_title'].str.strip()).drop_duplicates()

# Recommend function
def recommend(title, num_recommendations=5):
    title = title.strip()
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return data['movie_title'].iloc[movie_indices].tolist()

# ---------------- Streamlit UI ---------------- #
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on genres and plot keywords.")

movie_list = data['movie_title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie:", sorted(movie_list))

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommended Movies:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
