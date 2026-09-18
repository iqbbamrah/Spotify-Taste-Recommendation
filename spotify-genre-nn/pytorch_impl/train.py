"""Trains the same architecture with PyTorch's explicit-but-not-manual
style: real Dataset/DataLoader/optimizer objects, but you still write the
epoch loop and call loss.backward()/optimizer.step() yourself. Run with:

    python -m pytorch_impl.train
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.data import load_splits
from common.metrics import compute_metrics, save_run
from pytorch_impl.model import GenreClassifier

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5


def to_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, X, y, n_classes, device):
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return compute_metrics(y, probs, n_classes)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()

    train_loader = to_loader(splits.X_train, splits.y_train, BATCH_SIZE, shuffle=True)

    model = GenreClassifier(splits.n_features, splits.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(splits.X_val).to(device))
            val_loss = criterion(val_logits, torch.from_numpy(splits.y_val).to(device)).item()

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_metrics = evaluate(model, splits.X_val, splits.y_val, splits.n_classes, device)
            print(
                f"epoch {epoch:3d} | train_loss={epoch_loss / len(train_loader):.4f} "
                f"| val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )

        # Manual early stopping -- Keras gives you this as a one-line callback.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    train_seconds = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, splits.X_test, splits.y_test, splits.n_classes, device)
    print(f"\nTest metrics: {test_metrics}")

    n_params = sum(p.numel() for p in model.parameters())
    save_run("pytorch", test_metrics, n_params, train_seconds, epoch + 1)


if __name__ == "__main__":
    main()
