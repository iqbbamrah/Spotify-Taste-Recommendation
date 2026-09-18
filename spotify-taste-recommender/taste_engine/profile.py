"""Builds a personal taste profile (a weighted audio-feature centroid +
genre distribution) from a user's joined, matched listening history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from taste_engine.catalog import AUDIO_FEATURE_COLUMNS

# How much a track's source contributes to the taste profile. Saved/top
# tracks are explicit, durable signals of liking a song; recently-played
# includes background listening, shuffle, and skips, so it counts less.
SOURCE_WEIGHTS = {
    "saved_tracks": 1.0,
    "top_tracks_short_term": 1.0,
    "top_tracks_medium_term": 0.85,
    "top_tracks_long_term": 0.7,
    "playlist": 0.6,
    "recently_played": 0.4,
}

TOP_GENRES_IN_PROFILE = 15


@dataclass
class TasteProfile:
    feature_centroid: pd.Series  # standardized (z-scored) audio features
    genre_distribution: pd.Series  # genre -> normalized weight, sorted desc
    track_count: int
    matched_track_count: int
    match_rate: float
    source_breakdown: dict = field(default_factory=dict)


def compute_track_weights(user_tracks: pd.DataFrame) -> pd.Series:
    """weight = source importance * a linear rank decay within that source
    (rank 1 = most weight). Tracks with no rank (e.g. recently_played,
    saved_tracks) get the full source weight."""
    if "source" not in user_tracks.columns:
        raise ValueError("user_tracks must have a 'source' column")

    source_w = user_tracks["source"].map(SOURCE_WEIGHTS).fillna(0.3)

    if "rank" in user_tracks.columns:
        max_rank_by_source = user_tracks.groupby("source")["rank"].transform("max")
        rank = user_tracks["rank"]
        rank_decay = np.where(
            rank.notna() & (max_rank_by_source > 0),
            (max_rank_by_source - rank + 1) / max_rank_by_source,
            1.0,
        )
    else:
        rank_decay = 1.0

    return source_w * rank_decay


def build_taste_profile(
    joined_tracks: pd.DataFrame, feature_stats: pd.DataFrame
) -> TasteProfile:
    """joined_tracks: output of features.join_user_tracks, with a 'weight'
    column already attached (see compute_track_weights)."""
    if "weight" not in joined_tracks.columns:
        raise ValueError("joined_tracks must have a 'weight' column")

    matched = joined_tracks[joined_tracks["is_matched"]].copy()
    matched_count = len(matched)
    total_count = len(joined_tracks)

    if matched_count == 0:
        raise ValueError(
            "No tracks matched the catalog - can't build a taste profile. "
            "Check the match rate before calling this."
        )

    from taste_engine.catalog import standardize_features

    standardized = standardize_features(matched, feature_stats)
    weights = standardized["weight"].to_numpy()
    weight_sum = weights.sum()

    centroid = pd.Series(
        {
            col: float(np.average(standardized[col], weights=weights))
            for col in AUDIO_FEATURE_COLUMNS
        }
    )

    genre_weight = (
        matched.assign(weight=weights)
        .groupby("track_genre")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    genre_distribution = (genre_weight / genre_weight.sum()).head(TOP_GENRES_IN_PROFILE)

    source_breakdown = joined_tracks.groupby("source").size().to_dict()

    return TasteProfile(
        feature_centroid=centroid,
        genre_distribution=genre_distribution,
        track_count=total_count,
        matched_track_count=matched_count,
        match_rate=matched_count / total_count,
        source_breakdown=source_breakdown,
    )
