"""Trains GenreClassifier with a hand-rolled training loop: tf.GradientTape
for backprop and a manually implemented SGD-with-momentum update rule (no
tf.keras.optimizers). Run with:

    python -m tensorflow_impl.train
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from common.data import load_splits
from common.metrics import compute_metrics, save_run
from tensorflow_impl.model import GenreClassifier

LEARNING_RATE = 0.05
MOMENTUM = 0.9
BATCH_SIZE = 256
EPOCHS = 30


def loss_fn(logits, labels):
    per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(per_example)


def sgd_momentum_step(model: GenreClassifier, velocities: list, grads: list):
    """What `tf.keras.optimizers.SGD(momentum=...)` does under the hood:
    v = momentum * v - lr * grad; w += v
    """
    for var, grad, v in zip(model.trainable_variables, grads, velocities):
        v.assign(MOMENTUM * v - LEARNING_RATE * grad)
        var.assign_add(v)


@tf.function
def train_step(model: GenreClassifier, velocities, x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(logits, y_batch)
    grads = tape.gradient(loss, model.trainable_variables)
    sgd_momentum_step(model, velocities, grads)
    return loss


def evaluate(model: GenreClassifier, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    logits = model(tf.constant(X), training=False)
    probs = tf.nn.softmax(logits).numpy()
    return compute_metrics(y, probs, n_classes)


def main():
    splits = load_splits()
    model = GenreClassifier(splits.n_features, splits.n_classes)
    velocities = [tf.Variable(tf.zeros_like(v)) for v in model.trainable_variables]

    n_train = splits.X_train.shape[0]
    steps_per_epoch = n_train // BATCH_SIZE

    start = time.time()
    for epoch in range(EPOCHS):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled = splits.X_train[perm], splits.y_train[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            lo, hi = step * BATCH_SIZE, (step + 1) * BATCH_SIZE
            loss = train_step(
                model, velocities, tf.constant(X_shuffled[lo:hi]), tf.constant(y_shuffled[lo:hi])
            )
            epoch_loss += float(loss)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_metrics = evaluate(model, splits.X_val, splits.y_val, splits.n_classes)
            print(
                f"epoch {epoch:3d} | train_loss={epoch_loss / steps_per_epoch:.4f} "
                f"| val_acc={val_metrics['accuracy']:.4f}"
            )
    train_seconds = time.time() - start

    test_metrics = evaluate(model, splits.X_test, splits.y_test, splits.n_classes)
    print(f"\nTest metrics: {test_metrics}")

    save_run("tensorflow", test_metrics, model.n_params(), train_seconds, EPOCHS)


if __name__ == "__main__":
    main()
