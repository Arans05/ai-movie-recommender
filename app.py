import streamlit as st
import pandas as pd
import requests
import sqlite3
import hashlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- Page Configuration ---
st.set_page_config(page_title="Movie Recommendation",
                   page_icon="🍿", layout="wide")

# --- Get the absolute path for the database file ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'user_data.db')

# --- Database Setup Function ---


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (username TEXT NOT NULL, movieId INTEGER NOT NULL, PRIMARY KEY (username, movieId))''')
    conn.commit()
    conn.close()


setup_database()

# --- Database Functions ---


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def signup(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False


def add_to_watchlist(username, movie_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist (username, movieId) VALUES (?, ?)",
              (username, int(movie_id)))
    conn.commit()
    conn.close()


def get_watchlist_set(username):
    conn = get_db_connection()
    watchlist_df = pd.read_sql_query(
        "SELECT movieId FROM watchlist WHERE username = ?", conn, params=(username,))
    conn.close()
    return set(watchlist_df['movieId'].tolist())


def remove_from_watchlist(username, movie_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE username = ? AND movieId = ?",
              (username, int(movie_id)))
    conn.commit()
    conn.close()

# --- TMDb API & Data Loading ---


@st.cache_data
def fetch_movie_details(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    details = {"poster": "https://via.placeholder.com/500x750.png?text=No+Poster",
               "overview": "No summary available.", "release_date": "N/A"}
    try:
        response = requests.get(url).json()
        if response.get('poster_path'):
            details["poster"] = f"https://image.tmdb.org/t/p/w500/{response['poster_path']}"
        if response.get('overview'):
            details["overview"] = response['overview']
        if response.get('release_date'):
            details["release_date"] = response['release_date']
    except:
        pass
    return details


@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    links = pd.read_csv('links.csv')
    movies_with_links = pd.merge(
        movies, links, on='movieId').dropna(subset=['tmdbId'])
    movies_with_links['tmdbId'] = movies_with_links['tmdbId'].astype('int')
    return movies_with_links


movies_df = load_data()

# --- Content-Based Recommender (Optimized with KNN) ---


@st.cache_resource
def train_content_model(movies_data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_data['genres'].fillna(''))
    knn_model = NearestNeighbors(
        n_neighbors=11, metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    indices = pd.Series(movies_data.index,
                        index=movies_data['title']).drop_duplicates()
    return knn_model, tfidf_matrix, indices


knn_model, tfidf_matrix, content_indices = train_content_model(movies_df)


def get_content_recommendations(title, model, matrix, indices):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    movie_vector = matrix[idx]
    distances, movie_indices_rec = model.kneighbors(movie_vector)
    recommended_indices = movie_indices_rec.flatten()[1:]
    return movies_df.iloc[recommended_indices]


# --- Main App ---
st.title('🎬 Movie Recommendation')

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.show_details = None
    st.session_state.recommendations = pd.DataFrame()

# Sidebar for Login/Signup
with st.sidebar:
    st.header("User Account")
    if st.session_state.logged_in:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        login_form, signup_form = st.tabs(["Login", "Sign Up"])
        with login_form:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        with signup_form:
            with st.form("signup_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account"):
                    if signup(new_username, new_password):
                        st.success("Account created! Please log in.")
                    else:
                        st.error("Username already exists.")

# Main Content
if st.session_state.logged_in:
    user_watchlist = get_watchlist_set(st.session_state.username)
    tab1, tab2 = st.tabs(["**Recommend Movies**", "**My Watchlist**"])

    with tab1:
        st.header("Find movies with similar genres")
        selected_movie_title = st.selectbox(
            'Search for a movie:', movies_df['title'].values)
        if st.button('Recommend Based on Genre', use_container_width=True):
            st.session_state.recommendations = get_content_recommendations(
                selected_movie_title, knn_model, tfidf_matrix, content_indices)
            st.session_state.show_details = None

        if not st.session_state.recommendations.empty:
            st.subheader("Recommended Movies:")
            cols = st.columns(5)
            for i, (idx, movie) in enumerate(st.session_state.recommendations.iterrows()):
                with cols[i % 5]:
                    st.image(fetch_movie_details(movie['tmdbId'])[
                             'poster'], use_container_width=True)
                    st.caption(movie['title'])

                    in_watchlist = movie['movieId'] in user_watchlist
                    button_text = "✔️ In Watchlist" if in_watchlist else "➕ Add to Watchlist"
                    if st.button(button_text, key=f"toggle_content_{movie['movieId']}", use_container_width=True):
                        if in_watchlist:
                            remove_from_watchlist(
                                st.session_state.username, movie['movieId'])
                        else:
                            add_to_watchlist(
                                st.session_state.username, movie['movieId'])
                        st.rerun()

                    if st.button("Details", key=f"details_content_{movie['movieId']}", use_container_width=True):
                        st.session_state.show_details = movie
                        st.rerun()

    with tab2:
        st.header(f"{st.session_state.username}'s Watchlist")
        if not user_watchlist:
            st.info("Your watchlist is empty.")
        else:
            watchlist_movies = movies_df[movies_df['movieId'].isin(
                user_watchlist)]
            cols = st.columns(5)
            for i, (idx, movie) in enumerate(watchlist_movies.iterrows()):
                with cols[i % 5]:
                    st.image(fetch_movie_details(movie['tmdbId'])[
                             'poster'], use_container_width=True)
                    st.caption(movie['title'])

                    col1, col2 = st.columns(2)
                    if col1.button("Details", key=f"details_watchlist_{movie['movieId']}"):
                        st.session_state.show_details = movie
                        st.rerun()
                    if col2.button("➖ Remove", key=f"remove_watchlist_{movie['movieId']}"):
                        remove_from_watchlist(
                            st.session_state.username, movie['movieId'])
                        st.toast(
                            f"Removed '{movie['title']}' from your watchlist.")
                        st.rerun()

    # --- Dialog Box Logic ---
    if st.session_state.show_details is not None:
        movie = st.session_state.show_details
        details = fetch_movie_details(movie['tmdbId'])

        @st.dialog(f"{movie['title']}")
        def show_details_dialog():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(details['poster'], use_container_width=True)
            with col2:
                st.write(
                    f"**Release Date:** {details.get('release_date', 'N/A')}")
                st.write(f"**Genres:** {movie['genres'].replace('|', ', ')}")
                st.write("**Summary:**")
                st.write(details['overview'])
            if st.button("Close"):
                st.session_state.show_details = None
                st.rerun()
        show_details_dialog()

else:
    st.info(
        "Please log in or sign up using the sidebar to use the recommendation features.")
