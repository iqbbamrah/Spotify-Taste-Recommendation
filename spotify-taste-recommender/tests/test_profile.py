import pandas as pd
import pytest

from taste_engine.profile import build_taste_profile, compute_track_weights


@pytest.fixture
def feature_stats():
    # mean=0, std=1 for every feature -> "standardized" values equal raw
    # values, which makes the centroid math easy to check by hand.
    return pd.DataFrame(
        {"mean": [0.0] * 9, "std": [1.0] * 9},
        index=[
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ],
    )


def _base_features(danceability, energy, valence):
    return {
        "danceability": danceability,
        "energy": energy,
        "loudness": -5.0,
        "speechiness": 0.05,
        "acousticness": 0.1,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": valence,
        "tempo": 120.0,
    }


def test_compute_track_weights_rank_decay_and_source():
    user_tracks = pd.DataFrame(
        [
            {"source": "top_tracks_short_term", "rank": 1},
            {"source": "top_tracks_short_term", "rank": 50},
            {"source": "recently_played", "rank": None},
        ]
    )
    weights = compute_track_weights(user_tracks)
    # rank 1 in a source weighted 1.0 should get the full weight...
    assert weights.iloc[0] == pytest.approx(1.0)
    # ...rank 50 (the last slot) should get much less...
    assert weights.iloc[1] < weights.iloc[0]
    # ...and recently_played (source weight 0.4) with no rank gets flat 0.4.
    assert weights.iloc[2] == pytest.approx(0.4)


def test_build_taste_profile_weights_favor_saved_over_recently_played(feature_stats):
    joined = pd.DataFrame(
        [
            {
                "is_matched": True,
                "source": "saved_tracks",
                "track_genre": "pop",
                "weight": 1.0,
                **_base_features(0.9, 0.9, 0.9),
            },
            {
                "is_matched": True,
                "source": "recently_played",
                "track_genre": "jazz",
                "weight": 0.1,
                **_base_features(0.1, 0.1, 0.1),
            },
        ]
    )
    profile = build_taste_profile(joined, feature_stats)
    # centroid should sit much closer to the heavily-weighted saved track
    assert profile.feature_centroid["danceability"] > 0.7
    assert profile.matched_track_count == 2
    assert profile.match_rate == 1.0


def test_build_taste_profile_raises_on_no_matches(feature_stats):
    joined = pd.DataFrame(
        [{"is_matched": False, "source": "saved_tracks", "track_genre": None, "weight": 1.0, **_base_features(0.5, 0.5, 0.5)}]
    )
    with pytest.raises(ValueError):
        build_taste_profile(joined, feature_stats)


def test_genre_distribution_normalizes_to_one(feature_stats):
    joined = pd.DataFrame(
        [
            {"is_matched": True, "source": "saved_tracks", "track_genre": "pop", "weight": 1.0, **_base_features(0.5, 0.5, 0.5)},
            {"is_matched": True, "source": "saved_tracks", "track_genre": "rock", "weight": 1.0, **_base_features(0.5, 0.5, 0.5)},
        ]
    )
    profile = build_taste_profile(joined, feature_stats)
    assert profile.genre_distribution.sum() == pytest.approx(1.0)
    assert set(profile.genre_distribution.index) == {"pop", "rock"}
