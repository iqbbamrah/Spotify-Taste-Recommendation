"""Streamlit app: log in with Spotify, see your taste profile, get
recommendations. Each visitor's Spotify session lives only in their own
st.session_state - nothing about one user's data is shared with another,
so this works for anyone who opens the app, not just the developer.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import spotipy
import streamlit as st
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.oauth2 import SpotifyOAuth

from taste_engine.catalog import build_artist_feature_table, compute_feature_stats, load_catalog
from taste_engine.evaluate import evaluate_artist_holdout
from taste_engine.features import join_user_tracks
from taste_engine.mood_query import apply_mood_to_profile, interpret_mood_query
from taste_engine.profile import compute_track_weights, build_taste_profile
from taste_engine.recommend import (
    explain_recommendation,
    recommend_from_mood,
    recommend_from_profile,
    recommend_from_seed,
)
from taste_engine.spotify_client import SCOPES, fetch_all_listening_history

st.set_page_config(page_title="Your Spotify Taste, Mapped", page_icon="🎧", layout="wide")


def _secret_or_env(key: str, default: str | None = None) -> str | None:
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)


@st.cache_resource(show_spinner="Loading the public track catalog (one-time)...")
def load_catalog_resources():
    catalog = load_catalog()
    artist_table = build_artist_feature_table(catalog)
    feature_stats = compute_feature_stats(catalog)
    return catalog, artist_table, feature_stats


def get_auth_manager() -> SpotifyOAuth:
    client_id = _secret_or_env("SPOTIFY_CLIENT_ID")
    client_secret = _secret_or_env("SPOTIFY_CLIENT_SECRET")
    redirect_uri = _secret_or_env("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
    if not client_id or not client_secret:
        st.error(
            "Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET. Set them in "
            ".streamlit/secrets.toml locally, or in the Streamlit Cloud app's "
            "Secrets settings when deployed."
        )
        st.stop()
    # A per-session, in-memory cache handler - never touches disk, so one
    # visitor's token can never leak into another visitor's session.
    if "token_cache_handler" not in st.session_state:
        st.session_state.token_cache_handler = MemoryCacheHandler()
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_handler=st.session_state.token_cache_handler,
        show_dialog=False,
    )


def ensure_logged_in() -> spotipy.Spotify | None:
    auth_manager = get_auth_manager()

    query_params = st.query_params
    if "code" in query_params and "spotify_client" not in st.session_state:
        code = query_params["code"]
        auth_manager.get_access_token(code, check_cache=True)
        st.session_state.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        st.query_params.clear()
        st.rerun()

    if "spotify_client" in st.session_state:
        return st.session_state.spotify_client

    auth_url = auth_manager.get_authorize_url()
    st.title("🎧 Your Spotify Taste, Mapped")
    st.write(
        "Log in with your own Spotify account to see your real taste profile "
        "and get song recommendations built from it. Nothing you see here is "
        "shared with other visitors - your data lives only in your browser session."
    )
    st.link_button("Log in with Spotify", auth_url, type="primary")
    return None


@st.cache_data(show_spinner="Pulling your real Spotify listening history...")
def _pull_history(_sp: spotipy.Spotify, user_id: str) -> pd.DataFrame:
    # user_id is part of the cache key so different users don't share results
    return fetch_all_listening_history(_sp)


def build_profile_for_user(sp: spotipy.Spotify, catalog, artist_table, feature_stats):
    me = sp.current_user()
    history = _pull_history(sp, me["id"])
    joined = join_user_tracks(history, catalog, artist_table=artist_table)
    joined["weight"] = compute_track_weights(joined)
    profile = build_taste_profile(joined, feature_stats)
    return me, joined, profile


def render_profile(me, joined, profile):
    st.header(f"Hey, {me['display_name']} 👋")
    match_note = (
        f"Matched {profile.matched_track_count} of {profile.track_count} tracks "
        f"({profile.match_rate * 100:.0f}%) against the public catalog "
        "(exact track, or an artist-level average when the exact song wasn't in it - "
        "Spotify's own audio-features/genre endpoints are closed to new apps)."
    )
    st.caption(match_note)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Your top genres")
        genre_df = profile.genre_distribution.reset_index()
        genre_df.columns = ["genre", "weight"]
        fig = px.bar(genre_df, x="weight", y="genre", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="share of your taste", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Your sound profile")
        centroid_df = profile.feature_centroid.reset_index()
        centroid_df.columns = ["feature", "z_score"]
        fig2 = px.bar(centroid_df, x="z_score", y="feature", orientation="h")
        fig2.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="standard deviations vs. the average track",
            yaxis_title=None,
        )
        fig2.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)


def render_recommendations(catalog, feature_stats, profile, known_ids):
    st.subheader("Recommended for you")
    n = st.slider("How many recommendations?", 5, 40, 15)
    recs = recommend_from_profile(catalog, feature_stats, profile, known_track_ids=known_ids, n=n)
    if recs.empty:
        st.info("Couldn't find confident recommendations - try again once more of your library is matched.")
        return
    for _, row in recs.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['track_name']}** — {row['primary_artist']}  \n*{row['track_genre']}*")
            st.caption(explain_recommendation(row, profile, feature_stats))


def render_mood_search(catalog, feature_stats, profile, known_ids):
    st.subheader("Or: describe a mood or moment")
    st.caption(
        "e.g. \"something moody for a rainy commute\" or \"hype for a workout\" - "
        "this adjusts your taste profile toward the mood rather than replacing it."
    )
    query = st.text_input("What are you in the mood for?", key="mood_query_input")
    strength = st.slider(
        "How far from your usual taste?", 0.0, 1.0, 0.6, key="mood_strength",
        help="0 = ignore the mood, stick to your regular taste. 1 = fully chase the mood.",
    )
    if not query:
        return

    api_key = _secret_or_env("ANTHROPIC_API_KEY")
    if not api_key:
        st.info(
            "Mood search needs an ANTHROPIC_API_KEY (in .streamlit/secrets.toml "
            "locally, or Streamlit Cloud's Secrets settings when deployed)."
        )
        return

    if st.button("Find tracks for this mood"):
        with st.spinner("Interpreting your mood..."):
            mood = interpret_mood_query(query)
        if mood.interpretation:
            st.caption(f"🧭 {mood.interpretation}")
        target_vector, target_genres = apply_mood_to_profile(profile, mood, strength=strength)
        recs = recommend_from_mood(
            catalog, feature_stats, target_vector, target_genres, known_track_ids=known_ids, n=15
        )
        if recs.empty:
            st.info("Couldn't find confident recommendations for that mood.")
            return
        for _, row in recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['track_name']}** — {row['primary_artist']}  \n*{row['track_genre']}*")


def render_seed_search(sp, catalog, feature_stats, known_ids):
    st.subheader("Or: recommend from a specific song")
    query = st.text_input("Search a track (e.g. 'Travis Scott Fein')")
    if not query:
        return
    results = sp.search(q=query, type="track", limit=5)
    options = {
        f"{t['name']} — {t['artists'][0]['name']}": t for t in results["tracks"]["items"]
    }
    if not options:
        st.info("No results.")
        return
    choice = st.selectbox("Pick a track", list(options.keys()))
    seed_track = options[choice]

    seed_catalog_row = catalog[catalog["track_id"] == seed_track["id"]]
    if seed_catalog_row.empty:
        st.warning(
            "This exact track isn't in the public feature catalog, so I can't "
            "compute audio-feature similarity for it. Try a more well-known track."
        )
        return

    recs = recommend_from_seed(catalog, feature_stats, seed_track["id"], known_track_ids=known_ids, n=10)
    for _, row in recs.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['track_name']}** — {row['primary_artist']}  \n*{row['track_genre']}*")


def render_evaluation(joined, catalog, feature_stats, artist_table):
    with st.expander("How good are these recommendations, really?"):
        st.write(
            "Leave-one-artist-out evaluation: for each artist you demonstrably "
            "like, I rebuild your profile *without* that artist, then check "
            "whether their other catalog tracks land back in your top "
            "recommendations. Compared against the random-chance baseline for "
            "the same candidate pool size."
        )
        if st.button("Run evaluation (takes a few seconds)"):
            with st.spinner("Evaluating..."):
                k = 20
                result = evaluate_artist_holdout(joined, catalog, feature_stats, artist_table, k=k)
                pool_size = len(catalog[catalog["popularity"] >= 20])
                baseline = k / pool_size if pool_size else 0
                st.metric("Hit rate @20", f"{result.get('hit_rate_at_k', 0) * 100:.1f}%")
                st.metric("Random baseline @20", f"{baseline * 100:.2f}%")
                if baseline > 0:
                    st.metric("Lift over random", f"{result.get('hit_rate_at_k', 0) / baseline:.0f}x")
                st.caption(f"Based on {result['n']} held-out artist folds.")


def main():
    sp = ensure_logged_in()
    if sp is None:
        return

    catalog, artist_table, feature_stats = load_catalog_resources()
    me, joined, profile = build_profile_for_user(sp, catalog, artist_table, feature_stats)

    known_ids = set(joined["spotify_track_id"].dropna()) | set(joined["catalog_track_id"].dropna())

    render_profile(me, joined, profile)
    st.divider()
    tab1, tab2, tab3 = st.tabs(["For you", "From a song", "By mood"])
    with tab1:
        render_recommendations(catalog, feature_stats, profile, known_ids)
    with tab2:
        render_seed_search(sp, catalog, feature_stats, known_ids)
    with tab3:
        render_mood_search(catalog, feature_stats, profile, known_ids)
    st.divider()
    render_evaluation(joined, catalog, feature_stats, artist_table)

    if st.sidebar.button("Log out"):
        for key in ["spotify_client", "token_cache_handler"]:
            st.session_state.pop(key, None)
        st.rerun()


if __name__ == "__main__":
    main()
