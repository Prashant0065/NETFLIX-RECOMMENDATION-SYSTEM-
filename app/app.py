import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

@st.cache_data
def load_data():
    return pd.read_csv(OUTPUT_DIR / "netflix_processed.csv")

@st.cache_resource
def load_models():
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(MODEL_DIR / "tfidf_matrix.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
    return tfidf, tfidf_matrix, kmeans

df = load_data()
tfidf, tfidf_matrix, kmeans = load_models()

st.title("🎬 Netflix Recommendation System")
st.markdown("### AI-Powered Movie & TV Show Recommendations")
st.write("Discover Netflix titles similar to your favorite movies and TV shows.")

total_titles = df["title"].nunique()
total_movies = (df["type"] == "Movie").sum()
total_tv_shows = (df["type"] == "TV Show").sum()
total_clusters = df["cluster"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎬 Total Titles", total_titles)
col2.metric("🎥 Movies", total_movies)
col3.metric("📺 TV Shows", total_tv_shows)
col4.metric("🧠 Clusters", total_clusters)

st.divider()

st.subheader("🔎 Find Your Next Watch")

title = st.selectbox(
    "Search Netflix titles",
    sorted(df["title"].dropna().unique()),
    index=None,
    placeholder="Type a movie or TV show name...",
    filter_mode="contains"
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("🎯 Get Recommendations", type="primary"):

    if title is None:
        st.warning("Please select a Netflix movie or TV show first.")
        st.stop()

    matches = df[df["title"].str.lower() == title.lower()]

    if matches.empty:
        st.error("Title not found.")
    else:
        index = matches.index[0]

        cluster = df.loc[index, "cluster"]

        cluster_indices = df.index[
            df["cluster"] == cluster
        ]

        similarity_scores = cosine_similarity(
            tfidf_matrix[index],
            tfidf_matrix[cluster_indices]
        ).flatten()

        ranked_positions = similarity_scores.argsort()[::-1]

        recommended_indices = []
        recommended_scores = []

        for position in ranked_positions:

            candidate_index = cluster_indices[position]

            if candidate_index != index:
                recommended_indices.append(candidate_index)
                recommended_scores.append(
                    similarity_scores[position]
                )

            if len(recommended_indices) == num_recommendations:
                break

        recommendations = df.loc[
            recommended_indices,
            ["title", "type", "release_year", "listed_in"]
        ].copy()

        recommendations["similarity"] = recommended_scores

        st.divider()

        st.subheader("🍿 Recommended For You")

        st.caption(
            f"These recommendations are based on Cluster {cluster} "
            f"and content similarity."
        )

        for _, row in recommendations.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"### 🎬 {row['title']}"
                )

                col1, col2, col3 = st.columns(3)

                col1.write(
                    f"**Type**\n\n{row['type']}"
                )

                col2.write(
                    f"**Release Year**\n\n{row['release_year']}"
                )

                col3.write(
                    f"**Similarity**\n\n{row['similarity']:.3f}"
                )

                st.write(
                    f"**Genres:** {row['listed_in']}"
                )

st.divider()

with st.expander("🧠 How does this recommendation system work?"):

    st.write("""
    This Netflix recommendation system uses machine learning to find
    titles that are similar to the selected movie or TV show.
    """)

    st.markdown("### 1. 📂 Netflix Dataset")

    st.write(
        "The Netflix Movies and TV Shows dataset contains information "
        "about thousands of movies and TV shows."
    )

    st.markdown("### 2. 🧹 Data Preparation")

    st.write(
        "Important information such as director, cast, country, genres "
        "and description is combined to create a single feature."
    )

    st.markdown("### 3. 🔤 TF-IDF")

    st.write(
        "TF-IDF converts the textual information into numerical vectors "
        "so that machine learning algorithms can compare titles."
    )

    st.markdown("### 4. 🧠 K-Means Clustering")

    st.write(
        "K-Means groups Netflix titles into clusters containing titles "
        "with similar characteristics."
    )

    st.markdown("### 5. 📐 Cosine Similarity")

    st.write(
        "Cosine similarity measures how similar the selected title is "
        "to other titles inside its cluster."
    )

    st.markdown("### 6. 🍿 Recommendations")

    st.write(
        "The titles with the highest similarity scores are displayed "
        "as recommendations."
    )