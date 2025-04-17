import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your movies.csv file", type="csv")

if uploaded_file is not None:
    movies_data = pd.read_csv(uploaded_file)

    # Selecting the relevant features for recommendation
    selected_features = ['genre', 'desc', 'rating', 'votes']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combining features
    combined_features = movies_data['genre'].astype(str) + ' ' + movies_data['desc'].astype(str) + ' ' + movies_data['rating'].astype(str) + ' ' + movies_data['votes'].astype(str)

    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vectors)

    # Movie input
    movie_name = st.text_input("Enter your favourite movie name:")

    if movie_name:
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if find_close_match:
            close_match = find_close_match[0]
            index_of_movie = movies_data[movies_data.title == close_match].index[0]
            similarity_score = list(enumerate(similarity[index_of_movie]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            st.subheader("Top 10 Movie Recommendations:")
            for i, (index, score) in enumerate(sorted_similar_movies[1:11]):
                recommended_title = movies_data.iloc[index]['title']
                st.write(f"{i + 1}. {recommended_title}")
        else:
            st.warning("Movie not found! Please try another title.")
