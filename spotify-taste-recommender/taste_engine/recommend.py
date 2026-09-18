"""Content-based recommender: ranks catalog tracks by similarity to a taste
profile (or a single seed track), with a genre bonus and a diversity control
so results aren't just a wall of near-duplicates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taste_engine.catalog import AUDIO_FEATURE_COLUMNS, standardize_features

GENRE_BONUS_SCALE = 0.08  # bonus = target_genres[genre] * this, so a genre
# that's 45% of the user's profile counts far more than one that's 1% of it.
# Chosen via leave-one-artist-out evaluation (see evaluate.py / notebook):
# scale >= 0.15 let genre-tag matching dominate cosine similarity entirely
# and hit-rate@k dropped to 0 on real data; 0.0-0.10 were statistically
# indistinguishable given the small number of held-out folds, so a small
# positive value is kept as a mild tiebreaker without overriding the
# audio-feature signal.
DEFAULT_MIN_POPULARITY = 20  # filters out near-silent/mislabeled catalog rows


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = matrix_norm * vector_norm
    denom[denom == 0] = 1e-9
    return (matrix @ vector) / denom


def score_candidates(
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    target_vector: pd.Series,
    target_genres: pd.Series | None = None,
    min_popularity: int = DEFAULT_MIN_POPULARITY,
) -> pd.DataFrame:
    """Score every catalog track against a target feature vector (already in
    the same standardized space as feature_stats produces). Returns catalog
    rows with a 'similarity' column, sorted descending."""
    candidates = catalog[catalog["popularity"] >= min_popularity].copy()
    standardized = standardize_features(candidates, feature_stats)
    matrix = standardized[AUDIO_FEATURE_COLUMNS].to_numpy()
    vector = target_vector[AUDIO_FEATURE_COLUMNS].to_numpy()

    similarity = _cosine_similarity(matrix, vector)

    if target_genres is not None and len(target_genres) > 0:
        genre_weight = candidates["track_genre"].map(target_genres).fillna(0.0).to_numpy()
        similarity = similarity + genre_weight * GENRE_BONUS_SCALE

    candidates["similarity"] = similarity
    return candidates.sort_values("similarity", ascending=False).reset_index(drop=True)


def recommend_from_profile(
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    profile,  # taste_engine.profile.TasteProfile
    known_track_ids: set[str],
    n: int = 20,
    diversity_per_artist: int = 2,
) -> pd.DataFrame:
    """Top-N recommendations from a full taste profile, excluding tracks the
    user already has and capping how many picks come from any one artist so
    the list isn't dominated by a single act."""
    scored = score_candidates(
        catalog, feature_stats, profile.feature_centroid, profile.genre_distribution
    )
    scored = scored[~scored["track_id"].isin(known_track_ids)]
    return _apply_diversity_cap(scored, n, diversity_per_artist)


def recommend_from_seed(
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    seed_track_id: str,
    known_track_ids: set[str],
    n: int = 20,
    diversity_per_artist: int = 2,
) -> pd.DataFrame:
    """Top-N recommendations similar to a single seed track."""
    seed_rows = catalog[catalog["track_id"] == seed_track_id]
    if seed_rows.empty:
        raise ValueError(f"Seed track_id {seed_track_id!r} not found in catalog")
    seed = standardize_features(seed_rows, feature_stats).iloc[0]
    target_genres = pd.Series([1.0], index=[seed_rows.iloc[0]["track_genre"]])

    scored = score_candidates(catalog, feature_stats, seed, target_genres)
    scored = scored[~scored["track_id"].isin(known_track_ids | {seed_track_id})]
    return _apply_diversity_cap(scored, n, diversity_per_artist)


def recommend_from_mood(
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    target_vector: pd.Series,
    target_genres: pd.Series,
    known_track_ids: set[str],
    n: int = 20,
    diversity_per_artist: int = 2,
) -> pd.DataFrame:
    """Top-N recommendations against an arbitrary target vector/genre weights,
    e.g. a taste profile blended with a mood query via
    taste_engine.mood_query.apply_mood_to_profile. Identical ranking/exclusion/
    diversity behavior to recommend_from_profile - only the source of the
    target differs."""
    scored = score_candidates(catalog, feature_stats, target_vector, target_genres)
    scored = scored[~scored["track_id"].isin(known_track_ids)]
    return _apply_diversity_cap(scored, n, diversity_per_artist)


def _apply_diversity_cap(scored: pd.DataFrame, n: int, diversity_per_artist: int) -> pd.DataFrame:
    picked_rows = []
    artist_counts: dict[str, int] = {}
    for row in scored.itertuples(index=False):
        artist = row.primary_artist
        if artist_counts.get(artist, 0) >= diversity_per_artist:
            continue
        picked_rows.append(row)
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if len(picked_rows) >= n:
            break
    return pd.DataFrame(picked_rows)


def explain_recommendation(row: pd.Series, profile, feature_stats: pd.DataFrame) -> str:
    """Plain-English reason a track was recommended, based on which
    standardized features are closest to the profile centroid. `row` must
    carry raw (unstandardized) audio features, as returned by
    score_candidates/recommend_from_*; feature_stats standardizes it here so
    it's compared in the same z-scored space as profile.feature_centroid."""
    diffs = {}
    for col in AUDIO_FEATURE_COLUMNS:
        raw = row.get(col, np.nan)
        if pd.isna(raw):
            continue
        z = (raw - feature_stats.loc[col, "mean"]) / feature_stats.loc[col, "std"]
        diffs[col] = abs(z - profile.feature_centroid.get(col, np.nan))
    diffs = {k: v for k, v in diffs.items() if not np.isnan(v)}
    if not diffs:
        return "Matches your overall taste profile."
    closest = sorted(diffs, key=diffs.get)[:2]
    readable = {
        "danceability": "danceability",
        "energy": "energy",
        "valence": "mood/positivity",
        "acousticness": "acoustic feel",
        "tempo": "tempo",
        "instrumentalness": "instrumental feel",
        "speechiness": "vocal style",
        "liveness": "live-performance feel",
        "loudness": "loudness",
    }
    traits = " and ".join(readable.get(c, c) for c in closest)
    genre = row.get("track_genre")
    genre_note = f" in a genre ({genre}) you listen to a lot" if genre in profile.genre_distribution.index else ""
    return f"Similar {traits} to your top tracks{genre_note}."
