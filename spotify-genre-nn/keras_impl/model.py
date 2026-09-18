"""The same architecture as ../tensorflow_impl/model.py -- two hidden
Dense+ReLU+Dropout blocks and a linear output layer -- expressed with
Keras's high-level Sequential API instead of raw ops. Compare line counts.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_features: int, n_classes: int, hidden=(128, 64), dropout=0.3) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(hidden[0], activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden[1], activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(n_classes),  # logits; loss uses from_logits=True
        ]
    )
    return model
