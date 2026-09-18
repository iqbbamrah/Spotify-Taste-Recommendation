"""Shared data pipeline for all three framework implementations.

Every implementation (TensorFlow, Keras, PyTorch) calls `load_splits()` so
they train/validate/test on byte-identical splits and feature scaling -- the
only thing allowed to differ between them is the modeling code itself.

Task: predict a track's genre (114 balanced classes, 1000 tracks each) from
its numeric audio features. This is the same catalog used as the candidate
pool in ../spotify-taste-recommender, repurposed here as a supervised
learning dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CATALOG_PATH = PROJECT_ROOT / "data/raw/spotify_tracks_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_ms",
]
TARGET_COLUMN = "track_genre"

RANDOM_SEED = 42


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _load_clean_dataframe() -> pd.DataFrame:
    df = pd.read_csv(RAW_CATALOG_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df["explicit"] = df["explicit"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
    return df.reset_index(drop=True)


def load_splits(cache: bool = True) -> Splits:
    """Load, clean, encode, scale, and split the catalog.

    Splits are stratified by genre (80/10/10 train/val/test) so every genre
    is represented in each split despite there being 114 of them. Scaling
    statistics are fit on the training split only, to avoid leaking
    val/test information into the features.
    """
    cache_path = PROCESSED_DIR / "splits.npz"
    if cache and cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        return Splits(
            X_train=blob["X_train"],
            y_train=blob["y_train"],
            X_val=blob["X_val"],
            y_val=blob["y_val"],
            X_test=blob["X_test"],
            y_test=blob["y_test"],
            feature_names=list(blob["feature_names"]),
            class_names=list(blob["class_names"]),
        )

    df = _load_clean_dataframe()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET_COLUMN]).astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    splits = Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=FEATURE_COLUMNS,
        class_names=list(encoder.classes_),
    )

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_val=splits.X_val,
            y_val=splits.y_val,
            X_test=splits.X_test,
            y_test=splits.y_test,
            feature_names=np.array(splits.feature_names),
            class_names=np.array(splits.class_names),
        )

    return splits


if __name__ == "__main__":
    s = load_splits(cache=True)
    print(f"train={s.X_train.shape} val={s.X_val.shape} test={s.X_test.shape}")
    print(f"features={s.n_features} classes={s.n_classes}")
