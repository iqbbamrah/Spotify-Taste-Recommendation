"""Loads results/{framework}_metrics.json for whichever of the three
implementations have been run, and prints a side-by-side comparison table.

    python -m tensorflow_impl.train
    python -m keras_impl.train
    python -m pytorch_impl.train
    python compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRAMEWORKS = ["tensorflow", "keras", "pytorch"]
COLUMNS = ["accuracy", "top3_accuracy", "macro_f1", "n_params", "train_seconds", "epochs"]


def main():
    rows = {}
    for framework in FRAMEWORKS:
        path = RESULTS_DIR / f"{framework}_metrics.json"
        if path.exists():
            rows[framework] = json.loads(path.read_text())

    if not rows:
        print("No results yet. Run one or more of the train.py scripts first.")
        return

    header = f"{'metric':<16}" + "".join(f"{fw:>14}" for fw in rows)
    print(header)
    print("-" * len(header))
    for col in COLUMNS:
        line = f"{col:<16}"
        for fw in rows:
            val = rows[fw].get(col, "-")
            line += f"{val:>14.4f}" if isinstance(val, float) else f"{val:>14}"
        print(line)


if __name__ == "__main__":
    main()
