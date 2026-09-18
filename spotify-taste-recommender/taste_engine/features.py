"""Joins a user's real Spotify listening history against the public audio
feature catalog, since Spotify's own Audio Features endpoint (and, as of
this project, artist genres/popularity/related-artists/artist-top-tracks
too) is closed to new developer apps.
"""
from __future__ import annotations

import pandas as pd

from taste_engine.catalog import AUDIO_FEATURE_COLUMNS, _normalize_key, match_key_for

REQUIRED_USER_COLUMNS = {"spotify_track_id", "track_name", "artist_name"}


def join_user_tracks(
    user_tracks: pd.DataFrame, catalog: pd.DataFrame, artist_table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach audio features + genre to the user's tracks.

    Match precedence:
    1. Exact Spotify track_id.
    2. Normalized track+artist name key (different release/remaster with a
       different track_id, but the same song).
    3. Artist-level fallback (from catalog.build_artist_feature_table): the
       exact track isn't in the catalog, but the artist has other tracks in
       it, so we use that artist's average feature profile as a proxy. This
       matters a lot in practice - exact-track coverage of a static catalog
       against a real listening history can be under 5%, while artist-level
       coverage is often 10x higher.

    Unmatched tracks are kept with NaN feature columns so callers can decide
    how to handle them (the taste profile drops them; the UI reports the
    match rate broken down by match type).
    """
    missing = REQUIRED_USER_COLUMNS - set(user_tracks.columns)
    if missing:
        raise ValueError(f"user_tracks is missing required columns: {missing}")

    catalog_by_id = catalog.set_index("track_id")
    catalog_by_key = catalog.set_index("match_key")

    feature_cols = AUDIO_FEATURE_COLUMNS + ["track_genre", "popularity"]

    rows = []
    for row in user_tracks.itertuples(index=False):
        match = None
        matched_on = None
        catalog_track_id = None
        if row.spotify_track_id in catalog_by_id.index:
            match = catalog_by_id.loc[row.spotify_track_id]
            matched_on = "track_id"
            catalog_track_id = row.spotify_track_id  # index of catalog_by_id
        else:
            key = match_key_for(row.track_name, row.artist_name)
            if key in catalog_by_key.index:
                match = catalog_by_key.loc[key]
                # a duplicate match_key (rare, post-dedupe) yields a DataFrame
                if isinstance(match, pd.DataFrame):
                    match = match.iloc[0]
                matched_on = "name_artist"
                catalog_track_id = match["track_id"]
            elif artist_table is not None:
                artist_key = _normalize_key(row.artist_name)
                if artist_key in artist_table.index:
                    match = artist_table.loc[artist_key]
                    matched_on = "artist_level"

        record = row._asdict()
        record["matched_on"] = matched_on
        # The specific catalog row this came from, only meaningful for exact
        # matches (artist_level is an average across many catalog rows, not
        # any single one) - used by evaluate.py to know which catalog item
        # counts as the "correct" one to rank highly in a held-out test.
        record["catalog_track_id"] = catalog_track_id
        for col in feature_cols:
            record[col] = match[col] if match is not None else None
        rows.append(record)

    result = pd.DataFrame(rows)
    result["is_matched"] = result["matched_on"].notna()
    return result


def match_rate(joined: pd.DataFrame) -> float:
    if len(joined) == 0:
        return 0.0
    return float(joined["is_matched"].mean())
