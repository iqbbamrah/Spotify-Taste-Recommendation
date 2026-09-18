"""Turns a free-text mood/activity request ("something moody for a rainy
commute") into structured audio-feature targets and genre hints, using an
LLM for the one thing LLMs are actually reliable at here: mapping messy
language onto a fixed, known schema. The LLM never picks songs - it only
produces the parameters that feed into the same deterministic cosine-
similarity ranking (recommend.score_candidates) used everywhere else in
this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from taste_engine.catalog import AUDIO_FEATURE_COLUMNS
from taste_engine.profile import TasteProfile

# LLM output is clamped to this range before it ever touches the recommender.
# The standardized feature space is unbounded in principle, but real catalog
# tracks rarely sit beyond +-3 std devs of the mean - this stops a malformed
# or adversarial response (e.g. prompt injection via the mood text) from
# producing a target vector that silently breaks ranking.
MAX_ABS_FEATURE_TARGET = 3.0
MAX_GENRE_HINTS = 5

MOOD_TOOL = {
    "name": "set_mood_targets",
    "description": (
        "Translate a listener's free-text mood or activity request into target "
        "audio-feature values and genre hints for a music recommender."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "feature_targets": {
                "type": "object",
                "description": (
                    "Only include features you're confident the request implies "
                    "something about - omit the rest rather than guessing. Each "
                    "value is a standardized z-score, roughly in [-2, 2], where 0 "
                    "is an average track, positive is more of that quality than "
                    "average, negative is less. Keys must be chosen from: "
                    + ", ".join(AUDIO_FEATURE_COLUMNS)
                ),
                "additionalProperties": {"type": "number"},
            },
            "genre_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "0-5 lowercase genre tags the request suggests (e.g. "
                    "'lo-fi', 'acoustic', 'trap'). Leave empty if no genre is implied."
                ),
            },
            "interpretation": {
                "type": "string",
                "description": (
                    "One short sentence, shown directly to the listener, explaining "
                    "how their request was interpreted."
                ),
            },
        },
        "required": ["feature_targets", "genre_hints", "interpretation"],
    },
}


@dataclass
class MoodTarget:
    feature_targets: dict = field(default_factory=dict)  # feature -> target z-score
    genre_hints: list = field(default_factory=list)
    interpretation: str = ""


def _client_error(query: str) -> "MoodTarget":
    return MoodTarget(
        feature_targets={},
        genre_hints=[],
        interpretation=f"Couldn't interpret {query!r} - showing your regular taste profile instead.",
    )


def interpret_mood_query(query: str, client=None, model: str = "claude-sonnet-5") -> MoodTarget:
    """Call the LLM once to turn `query` into a MoodTarget. `client` is an
    anthropic.Anthropic-compatible object exposing .messages.create(...) -
    injectable so this is testable without a real API call/key."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=512,
        tools=[MOOD_TOOL],
        tool_choice={"type": "tool", "name": "set_mood_targets"},
        messages=[
            {
                "role": "user",
                "content": f"Listener's mood/activity request: {query.strip()!r}",
            }
        ],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        return _client_error(query)

    payload = tool_use.input or {}
    return _parse_mood_payload(payload)


def _parse_mood_payload(payload: dict) -> MoodTarget:
    """Validate and clamp raw tool-call output before it's trusted anywhere
    near the recommender. Never raises on malformed input - falls back to an
    empty (no-op) MoodTarget instead, since a bad mood parse should degrade to
    "ignore the mood" rather than break recommendations."""
    raw_targets = payload.get("feature_targets", {})
    feature_targets = {}
    if isinstance(raw_targets, dict):
        for key, value in raw_targets.items():
            if key not in AUDIO_FEATURE_COLUMNS:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            feature_targets[key] = max(-MAX_ABS_FEATURE_TARGET, min(MAX_ABS_FEATURE_TARGET, num))

    raw_genres = payload.get("genre_hints", [])
    genre_hints = []
    if isinstance(raw_genres, list):
        for g in raw_genres:
            if isinstance(g, str) and g.strip():
                genre_hints.append(g.strip().lower())
            if len(genre_hints) >= MAX_GENRE_HINTS:
                break

    interpretation = payload.get("interpretation", "")
    interpretation = interpretation.strip() if isinstance(interpretation, str) else ""

    return MoodTarget(feature_targets=feature_targets, genre_hints=genre_hints, interpretation=interpretation)


def apply_mood_to_profile(
    profile: TasteProfile, mood: MoodTarget, strength: float = 0.6
) -> tuple[pd.Series, pd.Series]:
    """Blend a MoodTarget onto a user's existing taste profile - pure,
    LLM-free, and unit-testable on its own. strength=0 ignores the mood
    entirely (pure historical taste); strength=1 fully replaces, for any
    feature the mood expressed an opinion on, the baseline with the mood's
    target. Returns (target_vector, target_genres) in the same shapes
    recommend.score_candidates already expects.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    target_vector = profile.feature_centroid.copy()
    for feature, mood_value in mood.feature_targets.items():
        if feature not in target_vector.index:
            continue
        baseline = target_vector[feature]
        target_vector[feature] = (1 - strength) * baseline + strength * mood_value

    target_genres = profile.genre_distribution.copy()
    if mood.genre_hints and strength > 0:
        boost = pd.Series(1.0, index=mood.genre_hints)
        combined = target_genres.add(boost * strength, fill_value=0.0)
        total = combined.sum()
        target_genres = combined / total if total > 0 else combined

    return target_vector, target_genres
