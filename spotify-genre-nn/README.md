# Genre Classification, Three Ways

## Problem

Predict a track's genre from 13 numeric audio features (danceability, energy, loudness, tempo, key, valence, etc.), and use that task to learn TensorFlow, Keras, and PyTorch hands-on. The same small feedforward network is implemented once in each framework and trained on identical data and splits, so the only thing that varies is the framework. Reading about the frameworks doesn't build muscle memory. Building the identical model three times does, and it makes the real differences between them concrete.

It's a moderately hard multi-class problem. Some genres separate cleanly on audio features alone (`classical` vs. `death-metal`), while others are nearly indistinguishable this way (`techno` vs. `minimal-techno` vs. `detroit-techno` differ mostly in artist, production era, and cultural context, none of which are in these 13 numbers).

## Data

- **Source:** the public [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset), also used by the companion [spotify-taste-recommender](../spotify-taste-recommender) project.
- 114,000 tracks, perfectly balanced across **114 genres** (1,000 tracks each), with 13 numeric audio features per track.
- **Splits:** stratified 80/10/10 train/validation/test by genre (seed 42), so every genre appears in each split. Scaling is fit on the training split only, to avoid leaking validation/test information. Splits are cached, so all three implementations train on byte-identical data.

## Methodology

**Architecture (identical in all three):** 13 inputs → Dense 128 + ReLU + Dropout 0.3 → Dense 64 + ReLU + Dropout 0.3 → 114 logits (17,458 parameters). Batch size 256, up to 30 epochs.

| | `tensorflow_impl/` | `keras_impl/` | `pytorch_impl/` |
|---|---|---|---|
| Layers | hand-written (`tf.Variable`, `tf.matmul`) | `keras.layers.Dense` | `nn.Linear` |
| Forward pass | manual Python loop over layers | built by `Sequential` | `nn.Sequential` |
| Training loop | hand-written, `tf.GradientTape` | `model.fit(...)` | hand-written, `loss.backward()` |
| Optimizer | hand-rolled SGD + momentum | `keras.optimizers.Adam` | `torch.optim.Adam` |
| Early stopping | not implemented | `EarlyStopping` callback (patience 5) | hand-written patience counter (5) |
| Abstraction level | lowest: you write every operation | highest: describe, don't implement | middle: objects provided, the loop is yours |

- `tensorflow_impl` is deliberately the "hard mode" version: it never touches `tf.keras`, so it shows what a `Dense` layer, a training step, and an optimizer update actually are underneath the convenience API.
- **Evaluation:** a shared metrics module reports top-1 accuracy, top-3 accuracy, macro-F1, parameter count, and training time on the held-out test set, and `compare.py` prints them side by side.
- **Tools:** Python, TensorFlow, Keras, PyTorch, scikit-learn, pandas, NumPy, pytest.

## Results

Test-set results from the most recent full runs:

| Framework | Top-1 accuracy | Top-3 accuracy | Macro-F1 | Parameters | Training time |
|---|---|---|---|---|---|
| Keras | 19.2% | 36.4% | 0.154 | 17,458 | 38 s |
| PyTorch | 19.1% | 36.8% | 0.155 | 17,458 | 61 s |

For scale, random guessing across 114 genres gives 0.9% top-1 and 2.6% top-3. The raw TensorFlow implementation's saved result is from a 2-epoch check run rather than a full 30-epoch run, so it isn't included in the comparison.

## Key takeaways

- **Same architecture, same data, same result:** Keras and PyTorch land within a few tenths of a point of each other on every metric, which confirms the differences between frameworks are in the developer experience, not the model.
- **Top-1 vs. top-3 is the real finding.** Top-3 accuracy is nearly double top-1 because many genres are only separable by information that isn't in these 13 audio features. That gap is a property of the data, not a bug.
- **Convenience hides failure modes.** In raw TensorFlow, forgetting to guard dropout with `training=True/False` silently changes train vs. eval behaviour. In PyTorch, forgetting `model.eval()` or `optimizer.zero_grad()` does the same. Keras handles both automatically, which is exactly what makes it easy to call `.fit()` without understanding what it does.
- **Code length tracks abstraction:** model plus training code is 83 lines in Keras, 128 in PyTorch, and 153 in raw TensorFlow.

## How to run

From this folder:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m tensorflow_impl.train
.venv/Scripts/python -m keras_impl.train
.venv/Scripts/python -m pytorch_impl.train
.venv/Scripts/python compare.py
```

Each trainer writes its metrics to `results/`, and `compare.py` prints whichever runs exist side by side.

## Repo structure

```
├── common/
│   ├── data.py            # shared load / clean / split / scale pipeline (used by all three)
│   └── metrics.py         # shared accuracy / top-3 / macro-F1 evaluation
├── tensorflow_impl/       # raw TensorFlow: manual layers, GradientTape, hand-rolled SGD + momentum
├── keras_impl/            # Keras Sequential API: model.compile() / model.fit()
├── pytorch_impl/          # PyTorch: nn.Module, DataLoader, manual training loop
├── compare.py             # prints the side-by-side results table
├── tests/test_data.py     # sanity checks on the shared data pipeline
├── data/raw/              # Spotify Tracks Dataset CSV
├── requirements.txt
└── README.md
```
