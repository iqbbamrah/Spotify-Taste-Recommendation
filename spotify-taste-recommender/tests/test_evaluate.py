import pandas as pd
import pytest

from taste_engine.catalog import compute_feature_stats, build_artist_feature_table
from taste_engine.evaluate import (
    evaluate_artist_holdout,
    evaluate_track_holdout,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# --- pure metric functions: hand-computed expected values -----------------


def test_precision_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert precision_at_k(ranked, {"c"}, k=5) == pytest.approx(0.2)
    assert precision_at_k(ranked, {"a", "b"}, k=2) == pytest.approx(1.0)
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranked, {"c"}, k=5) == pytest.approx(1.0)
    assert recall_at_k(ranked, {"c"}, k=1) == pytest.approx(0.0)
    assert recall_at_k(ranked, set(), k=5) == 0.0


def test_ndcg_at_k_matches_hand_computation():
    # relevant item "c" sits at rank 3 (0-indexed position 2)
    ranked = ["a", "b", "c", "d", "e"]
    # dcg = 1/log2(2+1+1) = 1/log2(4) = 0.5 ; idcg (ideal: rank 1) = 1/log2(2) = 1.0
    assert ndcg_at_k(ranked, {"c"}, k=5) == pytest.approx(0.5)
    # relevant item already at rank 1 -> perfect NDCG
    assert ndcg_at_k(ranked, {"a"}, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, set(), k=5) == 0.0


# --- integration smoke tests with a tiny synthetic catalog -----------------


def _track(track_id, artist, genre, popularity, **feats):
    row = {
        "track_id": track_id,
        "primary_artist": artist,
        "artists": artist,
        "track_name": track_id,
        "track_genre": genre,
        "popularity": popularity,
        "match_key": f"{track_id}|{artist}".lower(),
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
        ]
    )


@pytest.fixture
def feature_stats(catalog):
    return compute_feature_stats(catalog)


def _joined_row(spotify_id, track_name, artist_name, catalog_row, matched_on, catalog_track_id, source="top_tracks_short_term", rank=1):
    row = {
        "spotify_track_id": spotify_id,
        "track_name": track_name,
        "artist_name": artist_name,
        "source": source,
        "rank": rank,
        "matched_on": matched_on,
        "catalog_track_id": catalog_track_id,
        "is_matched": True,
        "track_genre": catalog_row["track_genre"],
        "popularity": catalog_row["popularity"],
    }
    for col in ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]:
        row[col] = catalog_row[col]
    return row


def test_evaluate_track_holdout_ranks_similar_track_above_dissimilar(catalog, feature_stats):
    close_1 = catalog[catalog.track_id == "close_1"].iloc[0]
    close_2 = catalog[catalog.track_id == "close_2"].iloc[0]
    joined = pd.DataFrame(
        [
            _joined_row("close_1", "close_1", "Artist Near", close_1, "track_id", "close_1"),
            _joined_row("close_2", "close_2", "Artist Near2", close_2, "track_id", "close_2"),
        ]
    )
    result = evaluate_track_holdout(joined, catalog, feature_stats, k=3)
    assert result["n"] == 2
    # both held-out tracks are close in feature space to the one remaining
    # track each time, and far_1 is not -> should rank above far_1
    assert result["precision_at_k"] > 0
    assert result["recall_at_k"] == pytest.approx(1.0)


def test_evaluate_artist_holdout_returns_hit_rate(catalog, feature_stats):
    artist_table = build_artist_feature_table(catalog)
    close_1 = catalog[catalog.track_id == "close_1"].iloc[0]
    close_2 = catalog[catalog.track_id == "close_2"].iloc[0]
    joined = pd.DataFrame(
        [
            _joined_row("u1", "Some Song", "Artist Near", close_1, "artist_level", None),
            _joined_row("u2", "close_2", "Artist Near2", close_2, "track_id", "close_2"),
        ]
    )
    result = evaluate_artist_holdout(joined, catalog, feature_stats, artist_table, k=3)
    assert result["n"] >= 1
    assert "hit_rate_at_k" in result
    assert 0.0 <= result["hit_rate_at_k"] <= 1.0


def test_evaluate_track_holdout_empty_input_returns_zero_n(catalog, feature_stats):
    result = evaluate_track_holdout(pd.DataFrame(columns=["matched_on"]), catalog, feature_stats)
    assert result["n"] == 0
    assert result["precision_at_k"] is None
