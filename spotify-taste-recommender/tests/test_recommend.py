import numpy as np
import pandas as pd
import pytest

from taste_engine.catalog import compute_feature_stats
from taste_engine.profile import TasteProfile
from taste_engine.recommend import (
    explain_recommendation,
    recommend_from_profile,
    recommend_from_seed,
    score_candidates,
)

FEATURE_COLS = [
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


def _track(track_id, artist, genre, popularity, **feats):
    row = {
        "track_id": track_id,
        "primary_artist": artist,
        "track_name": track_id,
        "track_genre": genre,
        "popularity": popularity,
    }
    defaults = dict(
        danceability=0.5, energy=0.5, loudness=-8.0, speechiness=0.05,
        acousticness=0.3, instrumentalness=0.0, liveness=0.1, valence=0.5, tempo=110.0,
    )
    defaults.update(feats)
    row.update(defaults)
    return row


@pytest.fixture
def catalog():
    return pd.DataFrame(
        [
            _track("close_1", "Artist Near", "pop", 60, danceability=0.85, energy=0.8, valence=0.75),
            _track("close_2", "Artist Near2", "pop", 60, danceability=0.82, energy=0.78, valence=0.7),
            _track("far_1", "Artist Far", "classical", 60, danceability=0.05, energy=0.05, valence=0.05, acousticness=0.95),
            _track("low_pop", "Artist Low", "pop", 5, danceability=0.85, energy=0.8, valence=0.75),
        ]
    )


@pytest.fixture
def feature_stats(catalog):
    return compute_feature_stats(catalog)


def test_score_candidates_ranks_similar_tracks_higher(catalog, feature_stats):
    target = pd.Series({c: 0.0 for c in FEATURE_COLS})
    # target centroid built in the same standardized space; a track with
    # high danceability/energy/valence (close_1/close_2) should z-score
    # positive and thus be closer to a "high taste" target than far_1.
    high_taste_target = pd.Series(
        {"danceability": 1.5, "energy": 1.5, "valence": 1.5, "loudness": 0.0,
         "speechiness": 0.0, "acousticness": -0.5, "instrumentalness": 0.0,
         "liveness": 0.0, "tempo": 0.0}
    )
    scored = score_candidates(catalog, feature_stats, high_taste_target, min_popularity=0)
    ranked_ids = scored["track_id"].tolist()
    assert ranked_ids.index("close_1") < ranked_ids.index("far_1")


def test_score_candidates_filters_low_popularity(catalog, feature_stats):
    target = pd.Series({c: 0.0 for c in FEATURE_COLS})
    scored = score_candidates(catalog, feature_stats, target, min_popularity=20)
    assert "low_pop" not in scored["track_id"].tolist()


def test_recommend_from_profile_excludes_known_tracks(catalog, feature_stats):
    profile = TasteProfile(
        feature_centroid=pd.Series({c: 1.0 for c in FEATURE_COLS}),
        genre_distribution=pd.Series({"pop": 1.0}),
        track_count=1,
        matched_track_count=1,
        match_rate=1.0,
    )
    recs = recommend_from_profile(
        catalog, feature_stats, profile, known_track_ids={"close_1"}, n=10
    )
    assert "close_1" not in recs["track_id"].tolist()


def test_recommend_from_seed_excludes_seed_itself(catalog, feature_stats):
    recs = recommend_from_seed(
        catalog, feature_stats, seed_track_id="close_1", known_track_ids=set(), n=10
    )
    assert "close_1" not in recs["track_id"].tolist()
    assert "far_1" in recs["track_id"].tolist()  # low-pop excluded by default filter, far_1 still qualifies


def test_recommend_from_seed_unknown_id_raises(catalog, feature_stats):
    with pytest.raises(ValueError):
        recommend_from_seed(catalog, feature_stats, seed_track_id="nope", known_track_ids=set())


def test_genre_bonus_scales_with_profile_weight(catalog, feature_stats):
    # far_1 (classical) is feature-distant from a "high energy/dance" target,
    # but if classical dominates the profile (weight 0.9) it should still be
    # able to outrank a merely feature-close, off-genre track once the bonus
    # is scaled by that weight rather than being a flat per-genre constant.
    target = pd.Series({c: 1.5 if c in ("danceability", "energy", "valence") else 0.0 for c in FEATURE_COLS})
    heavy_classical = pd.Series({"classical": 0.9, "pop": 0.1})
    scored = score_candidates(catalog, feature_stats, target, heavy_classical, min_popularity=0)
    far_1_similarity = scored.set_index("track_id").loc["far_1", "similarity"]
    light_classical = pd.Series({"classical": 0.05, "pop": 0.95})
    scored_light = score_candidates(catalog, feature_stats, target, light_classical, min_popularity=0)
    far_1_similarity_light = scored_light.set_index("track_id").loc["far_1", "similarity"]
    assert far_1_similarity > far_1_similarity_light


def test_explain_recommendation_uses_standardized_space(catalog, feature_stats):
    profile = TasteProfile(
        feature_centroid=pd.Series({c: 0.0 for c in FEATURE_COLS}),
        genre_distribution=pd.Series({"pop": 1.0}),
        track_count=1,
        matched_track_count=1,
        match_rate=1.0,
    )
    row = catalog[catalog["track_id"] == "close_1"].iloc[0]
    explanation = explain_recommendation(row, profile, feature_stats)
    assert isinstance(explanation, str) and len(explanation) > 0


def test_diversity_cap_limits_picks_per_artist(feature_stats):
    catalog = pd.DataFrame(
        [_track(f"t{i}", "Same Artist", "pop", 60, danceability=0.8) for i in range(5)]
    )
    stats = compute_feature_stats(catalog)
    profile = TasteProfile(
        feature_centroid=pd.Series({c: 0.0 for c in FEATURE_COLS}),
        genre_distribution=pd.Series({"pop": 1.0}),
        track_count=1,
        matched_track_count=1,
        match_rate=1.0,
    )
    recs = recommend_from_profile(
        catalog, stats, profile, known_track_ids=set(), n=10, diversity_per_artist=2
    )
    assert len(recs) == 2
