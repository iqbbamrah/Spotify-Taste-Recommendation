"""Loads and cleans the public audio-feature catalog (Kaggle/Hugging Face
'Spotify Tracks Dataset') used as the candidate pool and feature source,
since Spotify's live Audio Features endpoint is closed to new apps.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CATALOG_PATH = PROJECT_ROOT / "data/raw/spotify_tracks_dataset.csv"


def _normalize_key(name: str) -> str:
    """Lowercase, strip punctuation/whitespace so track/artist names from
    different sources (Spotify API vs. the catalog CSV) can be matched."""
    name = name.lower()
    name = re.sub(r"\(.*?\)|\[.*?\]", " ", name)  # drop "(feat. X)" etc.
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


def load_catalog(path: Path = RAW_CATALOG_PATH) -> pd.DataFrame:
    """Load the raw catalog CSV, dedupe by track_id, and add match keys."""
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog CSV not found at {path}. Run scripts/build_dataset.py first."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)

    df["primary_artist"] = df["artists"].str.split(";").str[0]
    df["match_key"] = (
        df["track_name"].map(_normalize_key) + "|" + df["primary_artist"].map(_normalize_key)
    )
    # A track can appear multiple times (different album/remaster) under the
    # same match_key; keep the most popular version as the canonical one.
    df = df.sort_values("popularity", ascending=False).drop_duplicates(
        subset="match_key", keep="first"
    )
    return df.reset_index(drop=True)


def match_key_for(track_name: str, artist_name: str) -> str:
    """Build the same match key used in load_catalog, for a user's track."""
    return f"{_normalize_key(track_name)}|{_normalize_key(artist_name)}"


def build_artist_feature_table(catalog: pd.DataFrame) -> pd.DataFrame:
    """Per-artist audio-feature profile, exploded across every collaborator
    on a track (not just primary_artist), so a track by "A;B" contributes to
    both A's and B's profile. This is the fallback used when a user's exact
    track isn't in the catalog but the artist has other tracks in it -
    Spotify's own artist-genre/popularity/related-artist endpoints are
    closed to new apps, so this is the only artist-level signal available.
    """
    exploded = catalog.assign(artist=catalog["artists"].str.split(";")).explode("artist")
    exploded["artist_key"] = exploded["artist"].map(_normalize_key)

    agg = exploded.groupby("artist_key").agg(
        **{col: (col, "mean") for col in AUDIO_FEATURE_COLUMNS},
        track_genre=("track_genre", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
        popularity=("popularity", "mean"),
        track_count=("track_id", "count"),
    )
    return agg


def compute_feature_stats(catalog: pd.DataFrame) -> pd.DataFrame:
    """Per-feature mean/std over the whole catalog, used to standardize both
    the catalog and any user profile into the same z-scored space so cosine
    similarity isn't dominated by large-range features like tempo/loudness."""
    stats = catalog[AUDIO_FEATURE_COLUMNS].agg(["mean", "std"]).T
    stats["std"] = stats["std"].replace(0, 1.0)  # guard against a constant column
    return stats


def standardize_features(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with each audio feature column replaced by its
    z-score under `stats` (from compute_feature_stats). Rows with NaN
    features stay NaN."""
    out = df.copy()
    for col in AUDIO_FEATURE_COLUMNS:
        out[col] = (out[col] - stats.loc[col, "mean"]) / stats.loc[col, "std"]
    return out
