import streamlit as st
import pandas as pd
import requests
import re
from urllib.parse import quote

st.set_page_config(
    page_title="Smart Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_movie_recommender():
    from src.movie_recommender import recommend
    return recommend


@st.cache_resource
def load_music_recommender():
    from src.music_recommender import recommend
    return recommend


recommend_movie = load_movie_recommender()
recommend_song = load_music_recommender()

@st.cache_data
def load_movie_data():

    movies = pd.read_csv("data/movielens/movie.csv")
    links = pd.read_csv("data/movielens/link.csv")

    links = links.dropna(subset=["imdbId"])
    links["imdbId"] = links["imdbId"].astype(int)

    movies = movies[movies["movieId"].isin(links["movieId"])]

    imdb_map = dict(zip(links["movieId"], links["imdbId"]))

    return movies, imdb_map


movies, imdb_map = load_movie_data()

movie_titles = movies["title"].tolist()
title_to_movieid = dict(zip(movies["title"], movies["movieId"]))

@st.cache_data
def load_music_data():

    songs = pd.read_csv("data/spotify/data.csv")
    songs["song"] = songs["name"] + " - " + songs["artists"]

    return songs


songs = load_music_data()
song_list = songs["song"].tolist()
songs["song_lower"] = songs["song"].str.lower()

OMDB_API_KEY = st.secrets["OMDB_API_KEY"]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_data(imdb_id):

    imdb_id_str = f"tt{str(imdb_id).zfill(7)}"
    url = f"https://www.omdbapi.com/?i={imdb_id_str}&apikey={OMDB_API_KEY}"

    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        if data.get("Response") == "True":
            return data

    except Exception:
        return None

    return None


def get_imdb_id(movie_id):
    return imdb_map.get(movie_id)

tab1, tab2 = st.tabs(["🎬 Movies", "🎧 Music"])

with tab1:

    st.title("🎬 Smart Movie Recommender")
    st.caption("Hybrid recommendation system (Cosine Similarity + SVD)")
    st.markdown("---")

    selected_movie = st.selectbox(
        "🎥 Search for a movie",
        options=movie_titles,
        index=None,
        placeholder="Start typing a movie name..."
    )

    if selected_movie and st.button("Recommend Movies 🍿"):
        st.session_state["run_movie"] = selected_movie
        st.session_state.pop("movie_recs", None)

    if "run_movie" in st.session_state:

        with st.spinner("Fetching recommendations..."):

            if "movie_recs" not in st.session_state:
                st.session_state["movie_recs"] = recommend_movie(
                    st.session_state["run_movie"]
                )

            recommendations = st.session_state["movie_recs"]

            st.markdown(
                f"#### Because you liked **{st.session_state['run_movie']}**:"
            )

            cols = st.columns(5)

            for i, movie in enumerate(recommendations):

                movie_id = title_to_movieid.get(movie)
                imdb_id = get_imdb_id(movie_id)

                movie_data = None

                if imdb_id:
                    movie_data = fetch_movie_data(imdb_id)

                with cols[i]:

                    if movie_data:

                        poster = movie_data.get("Poster")

                        if poster and poster != "N/A":
                            st.image(poster, use_container_width=True)

                        st.markdown(
                            f"""
                            **{movie}**  
                            ⭐ {movie_data.get('imdbRating', 'N/A')}  
                            📅 {movie_data.get('Year', 'N/A')}
                            """
                        )

                    else:
                        st.markdown(f"**{movie}**")

LASTFM_API_KEY = st.secrets["LASTFM_API_KEY"]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_album_cover(song_name, artist_name):

    clean_name = re.sub(
        r'\s*[-–]\s*(\d{4}\s+)?remaster.*',
        '',
        song_name,
        flags=re.IGNORECASE
    )

    clean_name = re.sub(
        r'\s*[-–]\s*(extended|live|radio|mono|stereo).*',
        '',
        clean_name,
        flags=re.IGNORECASE
    )

    r = requests.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={
            "method": "track.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist": artist_name,
            "track": clean_name,
            "format": "json"
        }
    )

    try:
        images = r.json()["track"]["album"]["image"]
        url = images[-1]["#text"]

        if url:
            return url

    except (KeyError, IndexError):
        pass

    r = requests.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={
            "method": "track.search",
            "api_key": LASTFM_API_KEY,
            "artist": artist_name,
            "track": clean_name,
            "format": "json",
            "limit": 1
        }
    )

    try:
        image = r.json()["results"]["trackmatches"]["track"][0]["image"]
        url = image[-1]["#text"]

        if url:
            return url

    except (KeyError, IndexError):
        pass

    return None

with tab2:

    st.title("🎧 Music Recommender")
    st.caption("Audio similarity recommender (PCA + KNN)")
    st.markdown("---")

    if "run_song" not in st.session_state:

        query = st.text_input("🎵 Search for a song")

        if query:

            matches = songs[
                songs["song_lower"].str.startswith(query.lower())
            ].head(8)

            if matches.empty:
                matches = songs[
                    songs["song_lower"].str.contains(query.lower(), na=False)
                ].head(8)

            if len(matches) > 0:

                st.markdown("#### Select a song")

                cols = st.columns(4)

                for i, (_, row) in enumerate(matches.iterrows()):

                    artist_clean = row["artists"].strip("[]'\"")

                    cover = fetch_album_cover(row["name"], artist_clean)

                    with cols[i % 4]:

                        if cover:
                            st.image(cover, use_container_width=True)

                        if st.button(
                            f"{row['name']} — {artist_clean}",
                            key=f"song_{i}"
                        ):
                            st.session_state["run_song"] = row["song"]
                            st.session_state.pop("song_recs", None)
                            st.rerun()

    if "run_song" in st.session_state:

        if st.button("🔄 Choose another song"):
            st.session_state.pop("run_song", None)
            st.session_state.pop("song_recs", None)
            st.rerun()

    if "run_song" in st.session_state:

        with st.spinner("Finding similar songs..."):

            if "song_recs" not in st.session_state:
                st.session_state["song_recs"] = recommend_song(
                    st.session_state["run_song"]
                )

            recs = st.session_state["song_recs"]

        if recs is None or recs.empty:

            st.warning("No recommendations found.")

        else:

            song_name = st.session_state["run_song"].split(" - ")[0]
            artist_name = st.session_state["run_song"].split(" - ")[1].strip("[]'\"")

            st.markdown(
                f"#### Because you liked **{song_name} — {artist_name}**:"
            )

            cols = st.columns(min(5, len(recs)))

            for i, (_, row) in enumerate(recs.head(5).iterrows()):

                artist_clean = row["artists"].strip("[]'\"")
                popularity = row.get("popularity", "N/A")
                year = row.get("year", "N/A")

                query = quote(f"{row['name']} {artist_clean}")
                spotify_link = f"https://open.spotify.com/search/{query}"

                cover_url = fetch_album_cover(row["name"], artist_clean)

                with cols[i]:

                    if cover_url:
                        st.image(cover_url, use_container_width=True)
                    else:
                        st.markdown("🎵")

                    st.markdown(
                        f"""
                        **{row['name']}**  
                        👤 {artist_clean}  
                        ⭐ {popularity}  
                        📅 {year}

                        [▶ Listen on Spotify]({spotify_link})
                        """
                    )
