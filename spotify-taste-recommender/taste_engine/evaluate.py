"""Offline evaluation: does the recommender actually rank tracks/artists the
user demonstrably likes highly, when those are held out of the profile it's
built from? This is the standard answer to "how do you evaluate a
recommender without live A/B testing."

Two strategies are provided:
- evaluate_track_holdout: classic precision/recall/NDCG@k using exact
  catalog track matches as ground truth. Works in general, but on this
  project's real data only ~7 tracks have an exact catalog match, so it has
  very little statistical power here (see README).
- evaluate_artist_holdout: leave-one-artist-out. For each artist the user
  has *some* catalog presence for (much more common - artist-level match
  covers 10x more of the library), rebuild the profile with that artist's
  listening entries removed, then check whether that artist's catalog
  tracks land in the top-K recommendations. Far more folds -> more reliable
  on real, sparse personal data.
"""
from __future__ import annotations

import math

import pandas as pd

from taste_engine.profile import build_taste_profile, compute_track_weights
from taste_engine.recommend import recommend_from_profile


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for tid in top_k if tid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = sum(1 for tid in relevant_ids if tid in top_k)
    return hits / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    top_k = ranked_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 2) for i, tid in enumerate(top_k) if tid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_track_holdout(
    joined_tracks: pd.DataFrame,
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    k: int = 20,
) -> dict:
    """Leave-one-exact-track-out evaluation. Returns per-track results plus
    averaged precision/recall/NDCG@k."""
    exact_matches = joined_tracks[joined_tracks["matched_on"].isin(["track_id", "name_artist"])]
    results = []
    for idx in exact_matches.index:
        held_out = joined_tracks.loc[idx]
        remaining = joined_tracks.drop(index=idx)
        remaining = remaining[remaining["is_matched"]]
        if remaining.empty:
            continue

        remaining = remaining.copy()
        remaining["weight"] = compute_track_weights(remaining)
        try:
            profile = build_taste_profile(remaining, feature_stats)
        except ValueError:
            continue

        known_ids = set(remaining["spotify_track_id"].dropna()) | set(
            remaining["catalog_track_id"].dropna()
        )
        ranked = recommend_from_profile(
            catalog, feature_stats, profile, known_track_ids=known_ids, n=max(k, 50)
        )
        ranked_ids = ranked["track_id"].tolist()
        relevant = {held_out["catalog_track_id"]}

        results.append(
            {
                "track_name": held_out["track_name"],
                "artist_name": held_out["artist_name"],
                "precision_at_k": precision_at_k(ranked_ids, relevant, k),
                "recall_at_k": recall_at_k(ranked_ids, relevant, k),
                "ndcg_at_k": ndcg_at_k(ranked_ids, relevant, k),
            }
        )

    return _summarize(results, k)


def evaluate_artist_holdout(
    joined_tracks: pd.DataFrame,
    catalog: pd.DataFrame,
    feature_stats: pd.DataFrame,
    artist_table: pd.DataFrame,
    k: int = 20,
    min_artist_catalog_tracks: int = 1,
) -> dict:
    """Leave-one-artist-out evaluation: for each matched artist, rebuild the
    profile without that artist's entries and check whether the held-out
    artist's own catalog tracks land in the top-K recommendations."""
    from taste_engine.catalog import _normalize_key  # local import, internal helper

    matched = joined_tracks[joined_tracks["is_matched"]].copy()
    artist_keys = matched["artist_name"].map(_normalize_key).unique()

    results = []
    for artist_key in artist_keys:
        if artist_key not in artist_table.index:
            continue  # only meaningful for artists the catalog actually has
        artist_catalog_tracks = set(
            catalog.assign(a=catalog["artists"].str.split(";"))
            .explode("a")
            .assign(a=lambda d: d["a"].map(_normalize_key))
            .loc[lambda d: d["a"] == artist_key, "track_id"]
        )
        if len(artist_catalog_tracks) < min_artist_catalog_tracks:
            continue

        remaining = matched[matched["artist_name"].map(_normalize_key) != artist_key].copy()
        if remaining.empty:
            continue
        remaining["weight"] = compute_track_weights(remaining)
        try:
            profile = build_taste_profile(remaining, feature_stats)
        except ValueError:
            continue

        known_ids = set(remaining["spotify_track_id"].dropna()) | set(
            remaining["catalog_track_id"].dropna()
        )
        ranked = recommend_from_profile(
            catalog, feature_stats, profile, known_track_ids=known_ids, n=max(k, 50)
        )
        ranked_ids = ranked["track_id"].tolist()

        results.append(
            {
                "artist_name": artist_key,
                "precision_at_k": precision_at_k(ranked_ids, artist_catalog_tracks, k),
                "recall_at_k": recall_at_k(ranked_ids, artist_catalog_tracks, k),
                "ndcg_at_k": ndcg_at_k(ranked_ids, artist_catalog_tracks, k),
                "hit_at_k": any(tid in artist_catalog_tracks for tid in ranked_ids[:k]),
            }
        )

    summary = _summarize(results, k)
    if results:
        summary["hit_rate_at_k"] = sum(r["hit_at_k"] for r in results) / len(results)
    return summary


def _summarize(results: list[dict], k: int) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "k": k, "precision_at_k": None, "recall_at_k": None, "ndcg_at_k": None, "results": []}
    return {
        "n": n,
        "k": k,
        "precision_at_k": sum(r["precision_at_k"] for r in results) / n,
        "recall_at_k": sum(r["recall_at_k"] for r in results) / n,
        "ndcg_at_k": sum(r["ndcg_at_k"] for r in results) / n,
        "results": results,
    }
