"""Framework-agnostic evaluation so all three implementations report
directly comparable numbers from the same code path."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, n_classes: int) -> dict:
    y_pred = y_pred_probs.argmax(axis=1)
    accuracy = float((y_pred == y_true).mean())
    top3 = float(
        top_k_accuracy_score(y_true, y_pred_probs, k=3, labels=np.arange(n_classes))
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": accuracy, "top3_accuracy": top3, "macro_f1": macro_f1}


def save_run(
    framework: str,
    metrics: dict,
    n_params: int,
    train_seconds: float,
    epochs: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": framework,
        "n_params": n_params,
        "train_seconds": round(train_seconds, 2),
        "epochs": epochs,
        **metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = RESULTS_DIR / f"{framework}_metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
