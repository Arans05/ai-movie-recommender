import streamlit as st
import pandas as pd
import requests
import sqlite3
import hashlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# --- Page Configuration ---
st.set_page_config(page_title="Movie Recommendation",
                   page_icon="🍿", layout="wide")

# --- Get the absolute path for the database file ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'user_data.db')

# --- Custom CSS for a Clean Blue & White Theme ---


def local_css():
    css = """
    <style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6; /* Light gray background */
    }
    /* Main content block with a card-like feel */
    .main .block-container {
        background-color: #FFFFFF; /* White content cards */
        border-radius: 10px;
        padding: 2rem;
        border: 1px solid #E0E0E0;
    }
    /* Style the recommendation cards */
    [data-testid="stVerticalBlock"] .st-emotion-cache-1f83x9k {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stVerticalBlock"] .st-emotion-cache-1f83x9k:hover {
        transform: scale(1.05) translateY(-10px);
        box-shadow: 0 15px 30px 0 rgba(0, 0, 0, 0.15);
    }
    /* Style the tabs */
    [data-testid="stTabs"] {
        border-top: 1px solid #E0E0E0;
        margin-top: 2rem;
    }
    /* Style for the 'In Watchlist' button */
    .stButton>button:contains('In Watchlist') {
        background-color: #2E8B57; /* SeaGreen for added items */
        color: white;
    }
    .stButton>button {
        border-radius: 8px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


local_css()

# --- Database Setup Function ---


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (username TEXT NOT NULL, movieId INTEGER NOT NULL, PRIMARY KEY (username, movieId))''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_ratings (username TEXT NOT NULL, movieId INTEGER NOT NULL, rating REAL NOT NULL, PRIMARY KEY (username, movieId))''')
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


def save_user_rating(username, movie_id, rating):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_ratings (username, movieId, rating) VALUES (?, ?, ?)",
              (username, int(movie_id), float(rating)))
    conn.commit()
    conn.close()


def get_user_ratings(username):
    conn = get_db_connection()
    return pd.read_sql_query("SELECT movieId, rating FROM user_ratings WHERE username = ?", conn, params=(username,))

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
    ratings = pd.read_csv('ratings.csv')
    movies_with_links = pd.merge(
        movies, links, on='movieId').dropna(subset=['tmdbId'])
    movies_with_links['tmdbId'] = movies_with_links['tmdbId'].astype('int')
    return movies_with_links, ratings


movies_df, ratings_df = load_data()

# --- Content-Based Recommender ---


@st.cache_data
def calculate_content_similarity(movies_data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_data['genres'].fillna(''))
    return cosine_similarity(tfidf_matrix), pd.Series(movies_data.index, index=movies_data['title']).drop_duplicates()


cosine_sim_matrix, content_indices = calculate_content_similarity(movies_df)


def get_content_recommendations(title, cosine_sim, indices):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    return movies_df.iloc[[i[0] for i in sim_scores]]


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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["**Recommend by Movie**", "**Recommend by Your Taste**", "**My Ratings**", "**My Watchlist**"])

    with tab1:
        st.header("Find movies with similar genres")
        selected_movie_title = st.selectbox(
            'Search for a movie:', movies_df['title'].values)
        if st.button('Recommend Based on Genre', use_container_width=True):
            st.session_state.recommendations = get_content_recommendations(
                selected_movie_title, cosine_sim_matrix, content_indices)
            st.session_state.show_details = None

        if not st.session_state.recommendations.empty:
            st.subheader("Recommended Movies:")
            cols = st.columns(5)
            for i, (idx, movie) in enumerate(st.session_state.recommendations.iterrows()):
                with cols[i % 5]:
                    with st.container(border=True):
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
        st.header("Get personalized recommendations")
        selected_movies_for_rating = st.multiselect(
            "Select 5 or more movies you've seen to rate them:",
            movies_df['title'].unique(),
            placeholder="Start typing to search for movies..."
        )
        if len(selected_movies_for_rating) >= 5:
            with st.form("ratings_form"):
                st.write("Rate the movies you selected from 1 (bad) to 5 (great):")
                for movie_title in selected_movies_for_rating:
                    rating = st.slider(
                        f"Rating for {movie_title}", 1.0, 5.0, 3.0, step=0.5, key=f"slider_{movie_title}")
                    movie_id = movies_df[movies_df['title']
                                         == movie_title]['movieId'].iloc[0]
                    save_user_rating(
                        st.session_state.username, movie_id, rating)

                if st.form_submit_button("Get My Recommendations"):
                    st.toast("Your ratings have been saved!")
                    with st.spinner("Training a model based on your taste..."):
                        user_ratings_df = get_user_ratings(
                            st.session_state.username)
                        user_ratings_df['userId'] = 0

                        combined_ratings_df = pd.concat(
                            [ratings_df, user_ratings_df], ignore_index=True)

                        reader = Reader(rating_scale=(1, 5))
                        data = Dataset.load_from_df(
                            combined_ratings_df[['userId', 'movieId', 'rating']], reader)
                        trainset = data.build_full_trainset()
                        model = SVD()
                        model.fit(trainset)

                        all_movie_ids = combined_ratings_df['movieId'].unique()
                        rated_movie_ids = combined_ratings_df[combined_ratings_df['userId'] == 0]['movieId'].unique(
                        )
                        unrated_movie_ids = [
                            mid for mid in all_movie_ids if mid not in rated_movie_ids]

                        predictions = [model.predict(
                            0, mid) for mid in unrated_movie_ids]
                        predictions.sort(key=lambda x: x.est, reverse=True)
                        top_n_movie_ids = [
                            pred.iid for pred in predictions[:10]]
                        st.session_state.recommendations = movies_df[movies_df['movieId'].isin(
                            top_n_movie_ids)]
                    st.rerun()
        elif len(selected_movies_for_rating) > 0:
            st.info(
                "Please select at least 5 movies to get an accurate recommendation.")

    with tab3:
        st.header(f"Your Rated Movies")
        user_ratings_df = get_user_ratings(st.session_state.username)
        if user_ratings_df.empty:
            st.info("You haven't rated any movies yet.")
        else:
            user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)
            user_rated_movies = pd.merge(
                movies_df, user_ratings_df, on='movieId')
            cols = st.columns(5)
            for i, (idx, movie) in enumerate(user_rated_movies.iterrows()):
                with cols[i % 5]:
                    with st.container(border=True):
                        st.image(fetch_movie_details(movie['tmdbId'])[
                                 'poster'], use_container_width=True)
                        st.caption(f"{movie['title']}")
                        st.write(f"You rated: **{movie['rating']}** ★")

    with tab4:
        st.header(f"{st.session_state.username}'s Watchlist")
        if not user_watchlist:
            st.info("Your watchlist is empty.")
        else:
            watchlist_movies = movies_df[movies_df['movieId'].isin(
                user_watchlist)]
            cols = st.columns(5)
            for i, (idx, movie) in enumerate(watchlist_movies.iterrows()):
                with cols[i % 5]:
                    with st.container(border=True):
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
