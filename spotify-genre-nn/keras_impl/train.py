"""Trains the same architecture as tensorflow_impl/train.py, but the Keras
way: model.compile() + model.fit() with callbacks instead of a hand-rolled
loop. Run with:

    python -m keras_impl.train
"""
from __future__ import annotations

import time

import tensorflow as tf
from tensorflow import keras

from common.data import load_splits
from common.metrics import compute_metrics, save_run
from keras_impl.model import build_model

BATCH_SIZE = 256
EPOCHS = 30


def main():
    splits = load_splits()
    model = build_model(splits.n_features, splits.n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            "results/keras_best_model.keras", monitor="val_loss", save_best_only=True
        ),
    ]

    start = time.time()
    model.fit(
        splits.X_train,
        splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - start

    logits = model.predict(splits.X_test, verbose=0)
    probs = tf.nn.softmax(logits).numpy()
    test_metrics = compute_metrics(splits.y_test, probs, splits.n_classes)
    print(f"\nTest metrics: {test_metrics}")

    n_params = model.count_params()
    save_run("keras", test_metrics, n_params, train_seconds, EPOCHS)


if __name__ == "__main__":
    main()
