# Genre Classification, Three Ways

The same neural network -- a small feedforward classifier -- implemented
once each in raw TensorFlow, in Keras, and in PyTorch, trained on the same
data with the same splits, so the only thing that varies is the framework
itself. Built to actually learn the three tools, not just read about them.

## The task

Predict a track's genre from 13 numeric audio features (danceability,
energy, loudness, tempo, key, valence, etc). The dataset is the public
[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(also used in the companion [spotify-taste-recommender](../spotify-taste-recommender)
project) -- 114,000 tracks, perfectly balanced across 114 genres (1,000
tracks each). It's a real, moderately hard multi-class classification
problem: some genres are cleanly separable from audio features alone
(`classical` vs `death-metal`), others are nearly indistinguishable this way
(`techno` vs `minimal-techno` vs `detroit-techno` sound different to a human
mostly for reasons -- artist, production era, cultural context -- that
aren't in these 13 numbers at all). Expect solid but unspectacular top-1
accuracy and a much better top-3 accuracy; that gap is itself worth noticing.

## Why three implementations of the same thing

Reading about TensorFlow/PyTorch/Keras doesn't build muscle memory. Building
the identical model three times does, and it also makes the actual
differences between the frameworks concrete instead of abstract:

| | `tensorflow_impl/` | `keras_impl/` | `pytorch_impl/` |
|---|---|---|---|
| Layers | hand-written (`tf.Variable`, `tf.matmul`) | `keras.layers.Dense` | `nn.Linear` |
| Forward pass | manual Python loop over layers | built by `Sequential` | `nn.Sequential` |
| Training loop | hand-written, `tf.GradientTape` | `model.fit(...)` | hand-written, `loss.backward()` |
| Optimizer | hand-rolled SGD+momentum update rule | `keras.optimizers.Adam` | `torch.optim.Adam` |
| Early stopping | not implemented (see note) | `EarlyStopping` callback | hand-written patience counter |
| Abstraction level | lowest -- you feel every operation | highest -- describe, don't implement | middle -- objects provided, loop is yours |

`tensorflow_impl` is intentionally the "hard mode" version: it never touches
`tf.keras`, so you see what a `Dense` layer, a training step, and an
optimizer update actually *are* underneath the convenience API everyone
normally uses (`tf.keras` is in fact where Keras itself lives when paired
with a TensorFlow backend -- so `keras_impl/` and "using TensorFlow the easy
way" are, in industry practice, often the same thing). `pytorch_impl` sits
in between: PyTorch gives you real layer and optimizer objects, but never
hides the training loop from you the way `.fit()` does.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running it

Each implementation is standalone and writes its results to `results/`:

```bash
python -m tensorflow_impl.train
python -m keras_impl.train
python -m pytorch_impl.train
python compare.py
```

`compare.py` reads whichever `results/*_metrics.json` files exist and prints
a side-by-side table of accuracy, top-3 accuracy, macro-F1, parameter count,
and training time -- run any subset of the three and it'll compare just
those.

The data pipeline (`common/data.py`) is shared and cached to
`data/processed/splits.npz` on first run, so all three trainers see byte-
identical train/val/test splits and feature scaling -- differences in the
results table reflect the modeling code, not different data.

## What to actually pay attention to

- **Line count and readability**: open all three `train.py` files side by
  side. Keras is the shortest by a wide margin -- that's the entire point of
  a high-level API, and also its main risk (it's easy to call `.fit()`
  without understanding what it's doing).
- **Where bugs are easy to introduce**: in `tensorflow_impl`, forgetting to
  guard dropout with `training=True/False` silently changes train vs. eval
  behavior; in raw PyTorch, forgetting `model.eval()` before evaluation (or
  `optimizer.zero_grad()` before a backward pass) does the same. Keras
  handles both automatically -- notice what convenience is actually buying
  you.
- **Top-1 vs top-3 accuracy**: a big gap between them here isn't a bug, it's
  a real signal that many genres are only separable from each other by
  something other than these 13 audio features.
- **Training time vs. parameter count**: all three models have the same
  architecture, so parameter counts should match almost exactly; training
  time differences come from the framework/hardware path, not the model.

## Project layout

```
common/data.py        shared load/clean/split/scale pipeline (used by all three)
common/metrics.py      shared accuracy/top-3/macro-F1 evaluation
tensorflow_impl/       raw TensorFlow: manual layers, GradientTape, hand-rolled SGD+momentum
keras_impl/            Keras Sequential API: model.compile()/model.fit()
pytorch_impl/          PyTorch: nn.Module, DataLoader, manual training loop
compare.py             prints the side-by-side results table
tests/test_data.py     sanity checks on the shared data pipeline
```
