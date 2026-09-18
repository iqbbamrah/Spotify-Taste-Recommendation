"""Same architecture again -- two hidden Linear+ReLU+Dropout blocks and a
linear output layer -- as an nn.Module. PyTorch sits between the other two
in abstraction level: layers are pre-built (like Keras) but there's no
`.fit()` -- you write the training loop yourself (like the raw TF version).
"""
from __future__ import annotations

import torch
from torch import nn


class GenreClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
