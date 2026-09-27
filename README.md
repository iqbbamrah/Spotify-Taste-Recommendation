# Spotify Projects

Two projects built on Spotify data, kept in one repo because they share a domain and a dataset. Each subfolder's README has the full details.

## Problem

- **[spotify-taste-recommender](spotify-taste-recommender/):** recommend new songs from a person's *real* Spotify listening history, with explanations and an evaluation, even though Spotify closed its audio-feature and recommendation endpoints to new developer apps in late 2024.
- **[spotify-genre-nn](spotify-genre-nn/):** predict a track's genre from its audio features, implementing the same neural network in raw TensorFlow, Keras, and PyTorch to compare the frameworks directly.

## Data

- **[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)** (used by both): 114,000 tracks across 114 genres with 13 numeric audio features (deduplicated to ~77K unique tracks for the recommender).
- **Real listening history** (recommender only): top tracks, recently played, saved tracks, and playlists, pulled through Spotify OAuth.

## Methodology

- **Recommender:** match listening history to the public catalog (with an artist-level fallback when the exact track is missing), build a weighted taste profile, rank candidates by cosine similarity with a profile-weighted genre bonus and a per-artist diversity cap, and evaluate with leave-one-artist-out testing against a random baseline. Includes an LLM-assisted mood search and a Streamlit app.
- **Genre classifier:** a two-hidden-layer feedforward network (17,458 parameters) trained on identical stratified 80/10/10 splits in each framework, evaluated on top-1 and top-3 accuracy, macro-F1, and training time.

## Results

| Project | Headline result |
|---|---|
| Recommender | Catalog match rate raised from 3.5% to 61.7% of real listening history, and hit-rate@20 of ~3.6% vs. ~0.03% random baseline (~100x lift) |
| Genre classifier | Keras and PyTorch both reach ~19% top-1 and ~36–37% top-3 accuracy across 114 genres (random: 0.9% / 2.6%), with macro-F1 ≈ 0.155 |

## Key takeaways

- **Data work decides whether a project is viable.** The recommender only works because of the artist-level fallback that recovered coverage after Spotify's API closures.
- **Evaluate against a baseline.** A 3.6% hit rate sounds small until it's compared with 0.03% by chance.
- **Frameworks change the developer experience, not the model.** Identical architectures trained on identical data give near-identical results in Keras and PyTorch.
- **Audio features alone have limits.** The large gap between top-1 and top-3 genre accuracy shows many genres differ in ways the features don't capture.

## How to run

Each project is standalone, with its own `requirements.txt`. See the How to run section in [spotify-taste-recommender](spotify-taste-recommender/) or [spotify-genre-nn](spotify-genre-nn/).

## Repo structure

```
├── spotify-taste-recommender/   # content-based recommender + Streamlit app (see its README)
├── spotify-genre-nn/            # one genre classifier in TensorFlow, Keras, and PyTorch (see its README)
└── README.md
```
