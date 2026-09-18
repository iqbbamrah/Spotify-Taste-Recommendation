import pandas as pd
import pytest

from taste_engine.catalog import (
    AUDIO_FEATURE_COLUMNS,
    build_artist_feature_table,
    compute_feature_stats,
    load_catalog,
    standardize_features,
)


@pytest.fixture
def raw_catalog_csv(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "track_id": "solo_track",
                "artists": "Solo Artist",
                "album_name": "Album",
                "track_name": "Solo Song",
                "popularity": 50,
                "danceability": 0.6,
                "energy": 0.6,
                "loudness": -6.0,
                "speechiness": 0.05,
                "acousticness": 0.2,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.5,
                "tempo": 100.0,
                "track_genre": "pop",
            },
            {
                "track_id": "collab_track",
                "artists": "Solo Artist;Feature Artist",
                "album_name": "Album 2",
                "track_name": "Collab Song",
                "popularity": 70,
                "danceability": 0.4,
                "energy": 0.4,
                "loudness": -10.0,
                "speechiness": 0.1,
                "acousticness": 0.3,
                "instrumentalness": 0.0,
                "liveness": 0.2,
                "valence": 0.3,
                "tempo": 90.0,
                "track_genre": "rap",
            },
        ]
    ).to_csv(path)
    return path


def test_load_catalog_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "does_not_exist.csv")


def test_build_artist_feature_table_credits_all_collaborators(raw_catalog_csv):
    catalog = load_catalog(raw_catalog_csv)
    table = build_artist_feature_table(catalog)
    # Solo Artist appears on both tracks -> averaged across both
    assert table.loc["solo artist", "danceability"] == pytest.approx((0.6 + 0.4) / 2)
    assert table.loc["solo artist", "track_count"] == 2
    # Feature Artist only appears on the collab track
    assert table.loc["feature artist", "danceability"] == pytest.approx(0.4)
    assert table.loc["feature artist", "track_count"] == 1


def test_compute_and_apply_feature_stats(raw_catalog_csv):
    catalog = load_catalog(raw_catalog_csv)
    stats = compute_feature_stats(catalog)
    standardized = standardize_features(catalog, stats)
    # standardized danceability should average out to ~0 across the catalog
    assert standardized["danceability"].mean() == pytest.approx(0.0, abs=1e-9)


def test_compute_feature_stats_guards_zero_std():
    catalog = pd.DataFrame({col: [0.5, 0.5, 0.5] for col in AUDIO_FEATURE_COLUMNS})
    stats = compute_feature_stats(catalog)
    assert (stats["std"] == 1.0).all()
