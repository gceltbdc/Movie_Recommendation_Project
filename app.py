import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("main_data.csv")

# Drop rows with missing important data
data = data.dropna(subset=['movie_title', 'comb'])

# Vectorize the 'comb' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['comb'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(data.index, index=data['movie_title'].str.strip().str.lower()).drop_duplicates()

# Recommendation function
def recommend(title, num_recommendations=5):
    title = title.strip().lower()
    if title not in indices:
        return ["Movie not found. Please check the title."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return data['movie_title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on cast, director, and genres!")

# User input
movie_list = sorted(data['movie_title'].dropna().unique())
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommended Movies:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
