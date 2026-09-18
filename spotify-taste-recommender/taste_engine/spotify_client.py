"""Pulls a user's real Spotify listening history via OAuth (top tracks,
recently played, saved tracks, playlists) using endpoints that remain open
to new developer apps (unlike Audio Features/Recommendations, closed since
Nov 2024 - see taste_engine.catalog for the workaround).
"""
from __future__ import annotations

import os

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

SCOPES = (
    "user-top-read user-read-recently-played user-library-read "
    "playlist-read-private playlist-modify-private playlist-modify-public"
)

TOP_TRACKS_TIME_RANGES = {
    "short_term": "top_tracks_short_term",
    "medium_term": "top_tracks_medium_term",
    "long_term": "top_tracks_long_term",
}


def get_auth_manager() -> SpotifyOAuth:
    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_path=".spotify_token_cache",
        open_browser=False,
    )


def get_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=get_auth_manager())


def _track_row(track: dict, source: str, rank: int | None = None) -> dict:
    artists = track.get("artists") or []
    return {
        "spotify_track_id": track["id"],
        "track_name": track["name"],
        "artist_name": artists[0]["name"] if artists else "",
        "source": source,
        "rank": rank,
    }


def fetch_top_tracks(sp: spotipy.Spotify) -> pd.DataFrame:
    rows = []
    for time_range, source in TOP_TRACKS_TIME_RANGES.items():
        results = sp.current_user_top_tracks(limit=50, time_range=time_range)
        for i, track in enumerate(results["items"], start=1):
            rows.append(_track_row(track, source, rank=i))
    return pd.DataFrame(rows)


def fetch_recently_played(sp: spotipy.Spotify, limit: int = 50) -> pd.DataFrame:
    results = sp.current_user_recently_played(limit=limit)
    rows = [_track_row(item["track"], "recently_played") for item in results["items"]]
    return pd.DataFrame(rows)


def fetch_saved_tracks(sp: spotipy.Spotify, max_tracks: int = 1000) -> pd.DataFrame:
    rows = []
    offset = 0
    while offset < max_tracks:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = results["items"]
        if not items:
            break
        rows.extend(_track_row(item["track"], "saved_tracks") for item in items)
        offset += len(items)
        if len(items) < 50:
            break
    return pd.DataFrame(rows)


def fetch_playlist_tracks(sp: spotipy.Spotify, max_playlists: int = 20) -> pd.DataFrame:
    rows = []
    me = sp.current_user()
    playlists = sp.current_user_playlists(limit=max_playlists)["items"]
    for playlist in playlists:
        if playlist["owner"]["id"] != me["id"]:
            continue  # only the user's own playlists, not followed ones
        offset = 0
        while True:
            results = sp.playlist_items(
                playlist["id"], limit=100, offset=offset, additional_types=["track"]
            )
            items = results["items"]
            if not items:
                break
            for item in items:
                track = item.get("track")
                if track and track.get("id"):
                    rows.append(_track_row(track, "playlist"))
            offset += len(items)
            if len(items) < 100:
                break
    return pd.DataFrame(rows)


def create_playlist_from_tracks(
    sp: spotipy.Spotify,
    name: str,
    track_ids: list[str],
    public: bool = False,
    description: str = "",
) -> dict:
    """Create a playlist on the current user's account and fill it with the
    given tracks, in order. Returns the created playlist object (has 'id'
    and 'external_urls'['spotify']).

    Uses current_user_playlist_create (POST /me/playlists), not the older
    user_playlist_create (POST /users/{user_id}/playlists) - Spotify's
    February 2026 Web API migration removed the latter, which now returns
    a bare 403 for every caller regardless of scope.
    """
    playlist = sp.current_user_playlist_create(
        name=name, public=public, description=description
    )
    sp.playlist_add_items(playlist["id"], [f"spotify:track:{tid}" for tid in track_ids])
    return playlist


def fetch_all_listening_history(sp: spotipy.Spotify) -> pd.DataFrame:
    """Combine every source into one user_tracks table ready for
    taste_engine.features.join_user_tracks."""
    frames = [
        fetch_top_tracks(sp),
        fetch_recently_played(sp),
        fetch_saved_tracks(sp),
        fetch_playlist_tracks(sp),
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True)
