import pandas as pd
import pytest

from taste_engine.mood_query import (
    MoodTarget,
    apply_mood_to_profile,
    interpret_mood_query,
)
from taste_engine.profile import TasteProfile

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


class _FakeToolUseBlock:
    def __init__(self, input_):
        self.type = "tool_use"
        self.input = input_


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, payload):
        self._payload = payload

    def create(self, **kwargs):
        return _FakeResponse([_FakeToolUseBlock(self._payload)])


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMessages(payload)


@pytest.fixture
def profile():
    return TasteProfile(
        feature_centroid=pd.Series({c: 0.0 for c in FEATURE_COLS}),
        genre_distribution=pd.Series({"pop": 0.7, "rock": 0.3}),
        track_count=10,
        matched_track_count=10,
        match_rate=1.0,
    )


# ---- apply_mood_to_profile: pure blending logic, no LLM involved ----

def test_strength_zero_ignores_mood(profile):
    mood = MoodTarget(feature_targets={"valence": -2.0}, genre_hints=["lo-fi"])
    target_vector, target_genres = apply_mood_to_profile(profile, mood, strength=0.0)
    assert target_vector["valence"] == pytest.approx(0.0)
    assert target_genres.equals(profile.genre_distribution)


def test_strength_one_fully_adopts_mood_value(profile):
    mood = MoodTarget(feature_targets={"valence": -2.0}, genre_hints=[])
    target_vector, _ = apply_mood_to_profile(profile, mood, strength=1.0)
    assert target_vector["valence"] == pytest.approx(-2.0)


def test_partial_strength_blends_linearly(profile):
    mood = MoodTarget(feature_targets={"energy": 2.0}, genre_hints=[])
    target_vector, _ = apply_mood_to_profile(profile, mood, strength=0.25)
    assert target_vector["energy"] == pytest.approx(0.5)


def test_unset_features_are_left_at_baseline(profile):
    mood = MoodTarget(feature_targets={"valence": -1.0}, genre_hints=[])
    target_vector, _ = apply_mood_to_profile(profile, mood, strength=1.0)
    for col in FEATURE_COLS:
        if col != "valence":
            assert target_vector[col] == pytest.approx(0.0)


def test_genre_hints_boost_and_renormalize(profile):
    mood = MoodTarget(feature_targets={}, genre_hints=["lo-fi"])
    _, target_genres = apply_mood_to_profile(profile, mood, strength=1.0)
    assert target_genres.sum() == pytest.approx(1.0)
    assert target_genres["lo-fi"] > target_genres["pop"]


def test_invalid_strength_raises(profile):
    with pytest.raises(ValueError):
        apply_mood_to_profile(profile, MoodTarget(), strength=1.5)


# ---- interpret_mood_query: LLM call, network mocked out via a fake client ----

def test_interpret_mood_query_parses_valid_payload():
    payload = {
        "feature_targets": {"valence": -1.2, "energy": -0.5, "not_a_real_feature": 5.0},
        "genre_hints": ["Lo-Fi", "Acoustic"],
        "interpretation": "Something mellow and downbeat.",
    }
    mood = interpret_mood_query("something moody for a rainy day", client=_FakeClient(payload))
    assert mood.feature_targets == {"valence": -1.2, "energy": -0.5}
    assert mood.genre_hints == ["lo-fi", "acoustic"]
    assert mood.interpretation == "Something mellow and downbeat."


def test_interpret_mood_query_clamps_extreme_values():
    payload = {
        "feature_targets": {"energy": 999.0, "valence": -999.0},
        "genre_hints": [],
        "interpretation": "",
    }
    mood = interpret_mood_query("hype", client=_FakeClient(payload))
    assert mood.feature_targets["energy"] == 3.0
    assert mood.feature_targets["valence"] == -3.0


def test_interpret_mood_query_caps_genre_hints():
    payload = {
        "feature_targets": {},
        "genre_hints": ["a", "b", "c", "d", "e", "f", "g"],
        "interpretation": "",
    }
    mood = interpret_mood_query("anything", client=_FakeClient(payload))
    assert len(mood.genre_hints) == 5


def test_interpret_mood_query_tolerates_malformed_payload():
    payload = {"feature_targets": "not a dict", "genre_hints": "not a list", "interpretation": 42}
    mood = interpret_mood_query("anything", client=_FakeClient(payload))
    assert mood.feature_targets == {}
    assert mood.genre_hints == []
    assert mood.interpretation == ""


def test_interpret_mood_query_rejects_empty_query():
    with pytest.raises(ValueError):
        interpret_mood_query("   ", client=_FakeClient({}))
