"""A dense feedforward net built from raw TensorFlow ops -- no tf.keras.

This is deliberately "low-level" so it teaches what tf.keras normally hides:
weights are plain tf.Variables you initialize yourself, the forward pass is
explicit matmuls, and there is no `.compile()`/`.fit()` anywhere. Compare
this file to ../keras_impl/model.py, which builds the *same* architecture
in about a tenth of the code.
"""
from __future__ import annotations

import tensorflow as tf


class DenseLayer:
    """One fully-connected layer: y = activation(x @ W + b)."""

    def __init__(self, n_in: int, n_out: int, activation=None, seed: int = 0):
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.W = tf.Variable(initializer((n_in, n_out)), name="W")
        self.b = tf.Variable(tf.zeros((n_out,)), name="b")
        self.activation = activation

    @property
    def trainable_variables(self):
        return [self.W, self.b]

    def __call__(self, x, training: bool = False):
        z = tf.matmul(x, self.W) + self.b
        return self.activation(z) if self.activation is not None else z


class GenreClassifier:
    """Two hidden layers with dropout, applied manually (only active when
    `training=True`), feeding a linear output layer. Softmax is folded into
    the loss function (`tf.nn.sparse_softmax_cross_entropy_with_logits`)
    rather than the model, which is the numerically stable way to do it."""

    def __init__(self, n_features: int, n_classes: int, hidden=(128, 64), dropout=0.3, seed=0):
        self.dropout_rate = dropout
        dims = [n_features, *hidden]
        self.hidden_layers = [
            DenseLayer(dims[i], dims[i + 1], activation=tf.nn.relu, seed=seed + i)
            for i in range(len(hidden))
        ]
        self.output_layer = DenseLayer(dims[-1], n_classes, activation=None, seed=seed + 99)

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.hidden_layers:
            variables += layer.trainable_variables
        variables += self.output_layer.trainable_variables
        return variables

    def __call__(self, x, training: bool = False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            if training and self.dropout_rate > 0:
                x = tf.nn.dropout(x, rate=self.dropout_rate)
        return self.output_layer(x, training=training)

    def n_params(self) -> int:
        return int(sum(tf.size(v).numpy() for v in self.trainable_variables))
