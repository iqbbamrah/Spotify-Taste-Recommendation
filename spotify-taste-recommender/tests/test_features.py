import pandas as pd
import pytest

from taste_engine.catalog import build_artist_feature_table, load_catalog
from taste_engine.features import join_user_tracks, match_rate


@pytest.fixture
def tiny_catalog(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "track_id": "id_a",
                "artists": "Artist A",
                "album_name": "Album A",
                "track_name": "Song A",
                "popularity": 80,
                "danceability": 0.8,
                "energy": 0.7,
                "loudness": -5.0,
                "speechiness": 0.05,
                "acousticness": 0.1,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.6,
                "tempo": 120.0,
                "track_genre": "pop",
            },
            {
                # Same song, remastered under a different track_id but lower
                # popularity -> should be dropped in favor of id_a when
                # match_key collides, and should NOT be picked over an exact
                # track_id match.
                "track_id": "id_a_remaster",
                "artists": "Artist A",
                "album_name": "Album A (Remaster)",
                "track_name": "Song A",
                "popularity": 10,
                "danceability": 0.1,
                "energy": 0.1,
                "loudness": -20.0,
                "speechiness": 0.01,
                "acousticness": 0.9,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.1,
                "tempo": 90.0,
                "track_genre": "pop",
            },
            {
                "track_id": "id_b",
                "artists": "Artist B;Artist C",
                "album_name": "Album B",
                "track_name": "Song B (feat. Artist C)",
                "popularity": 50,
                "danceability": 0.4,
                "energy": 0.5,
                "loudness": -8.0,
                "speechiness": 0.1,
                "acousticness": 0.2,
                "instrumentalness": 0.0,
                "liveness": 0.2,
                "valence": 0.4,
                "tempo": 100.0,
                "track_genre": "rock",
            },
        ]
    ).to_csv(csv_path)
    return load_catalog(csv_path)


def test_exact_track_id_match_wins_over_name_match(tiny_catalog):
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "id_a", "track_name": "Song A", "artist_name": "Artist A"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog)
    assert joined.loc[0, "matched_on"] == "track_id"
    assert joined.loc[0, "danceability"] == 0.8


def test_fallback_name_artist_match(tiny_catalog):
    # Different track_id than anything in the catalog, but same normalized
    # name/artist as "Song B (feat. Artist C)" once parentheticals are
    # stripped -> should fall back to the name/artist key.
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "not_in_catalog", "track_name": "Song B", "artist_name": "Artist B"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog)
    assert joined.loc[0, "matched_on"] == "name_artist"
    assert joined.loc[0, "track_genre"] == "rock"


def test_unmatched_track_has_null_features(tiny_catalog):
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "totally_unknown", "track_name": "Nonexistent Song", "artist_name": "Nobody"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog)
    assert joined.loc[0, "is_matched"] == False  # noqa: E712
    assert pd.isna(joined.loc[0, "danceability"])


def test_match_rate(tiny_catalog):
    user_tracks = pd.DataFrame(
        [
            {"spotify_track_id": "id_a", "track_name": "Song A", "artist_name": "Artist A"},
            {"spotify_track_id": "unknown", "track_name": "??", "artist_name": "??"},
        ]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog)
    assert match_rate(joined) == 0.5


def test_missing_required_column_raises(tiny_catalog):
    with pytest.raises(ValueError):
        join_user_tracks(pd.DataFrame([{"track_name": "x"}]), tiny_catalog)


def test_artist_level_fallback_when_no_track_match(tiny_catalog):
    # tiny_catalog already deduped id_a/id_a_remaster down to the more
    # popular id_a (see load_catalog's match_key dedupe), so Artist A's
    # artist-level profile is just that one surviving track.
    artist_table = build_artist_feature_table(tiny_catalog)
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "unknown_id", "track_name": "A Totally Different Song", "artist_name": "Artist A"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog, artist_table=artist_table)
    assert joined.loc[0, "matched_on"] == "artist_level"
    assert joined.loc[0, "is_matched"] == True  # noqa: E712
    assert joined.loc[0, "danceability"] == pytest.approx(0.8)


def test_artist_level_fallback_not_used_when_track_matches_directly(tiny_catalog):
    artist_table = build_artist_feature_table(tiny_catalog)
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "id_a", "track_name": "Song A", "artist_name": "Artist A"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog, artist_table=artist_table)
    assert joined.loc[0, "matched_on"] == "track_id"
    assert joined.loc[0, "danceability"] == 0.8  # exact track, not the artist average


def test_catalog_track_id_resolves_correctly_for_both_exact_match_types(tiny_catalog):
    # track_id match: catalog_track_id should equal the user's own id
    exact = join_user_tracks(
        pd.DataFrame([{"spotify_track_id": "id_a", "track_name": "Song A", "artist_name": "Artist A"}]),
        tiny_catalog,
    )
    assert exact.loc[0, "catalog_track_id"] == "id_a"

    # name/artist match: catalog_track_id should be the CATALOG's id, not
    # the (different) id the user's own Spotify library uses for that song.
    name_match = join_user_tracks(
        pd.DataFrame([{"spotify_track_id": "users_own_different_id", "track_name": "Song B", "artist_name": "Artist B"}]),
        tiny_catalog,
    )
    assert name_match.loc[0, "catalog_track_id"] == "id_b"


def test_artist_level_fallback_none_when_artist_unknown(tiny_catalog):
    artist_table = build_artist_feature_table(tiny_catalog)
    user_tracks = pd.DataFrame(
        [{"spotify_track_id": "unknown", "track_name": "??", "artist_name": "Totally Unknown Artist"}]
    )
    joined = join_user_tracks(user_tracks, tiny_catalog, artist_table=artist_table)
    assert joined.loc[0, "is_matched"] == False  # noqa: E712
