# Your Spotify Taste, Mapped

## Problem

Build a content-based music recommender on **real** Spotify listening history, not a demo dataset, that recommends real songs you don't already have, explains each pick, and can answer *"how do you know this works?"* with an actual evaluation.

The hard part isn't the recommender math (cosine similarity over audio features is standard). It's the data. **Spotify closed its Audio Features, Recommendations, Related Artists, and Artist-Top-Tracks endpoints to new developer apps in late 2024.** I confirmed this empirically: those endpoints return `403 Forbidden` for a new app, and even artist objects come back with `genres` and `popularity` stripped. That rules out the obvious approach of calling Spotify for audio features, so they have to come from somewhere else.

## Data

- **Your own listening history, via Spotify OAuth:** top tracks (short-, medium- and long-term), recently played, saved tracks, and playlists, the parts of the API still open to new apps.
- **Public audio-feature catalog:** the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (114K tracks, deduplicated to ~77K unique tracks) with danceability, energy, valence, acousticness, tempo, genre, and popularity.
- **Coverage problem:** the catalog is a frozen 2023-era snapshot, so matching my library on exact track covered only **3.5%** of it. Most current listening (recent hip-hop/trap) simply isn't in the file.

## Methodology

- **Matching with an artist-level fallback:** tracks are matched to the catalog by exact track, then by name/artist, and finally by a per-artist average audio-feature profile (crediting every collaborator on a track). The artist fallback is what makes the rest of the pipeline viable.
- **Taste profile:** a weighted centroid over 9 standardized audio-feature dimensions plus a normalized genre distribution. Saved tracks and short-term top tracks count fully, long-term top tracks and playlist tracks count somewhat less, and recently played counts least, since it includes background listening, shuffle, and skips.
- **Recommending:** rank the ~77K candidate catalog by cosine similarity to the taste centroid, add a genre bonus *scaled by how much that genre matters in your profile*, filter out tracks you already have, and cap picks per artist for diversity. Each recommendation is explained by the audio features closest to your profile, compared in the same standardized space the model uses.
- **Evaluation (offline):** leave-one-artist-out. For every artist you demonstrably like, rebuild the profile *without* them and check whether their other tracks land back in the top-K recommendations. Reported as hit-rate@K, precision@K, recall@K, and NDCG@K against a random baseline (K / candidate pool size). The genre-bonus weight was chosen by sweeping values against this harness, not by eyeballing output.
- **Mood search:** free-text requests ("something moody for a rainy commute") are mapped by an LLM (Claude, via a forced tool call) onto structured audio-feature targets and genre hints, which are blended onto your taste profile and ranked by the same scorer. The LLM never sees the catalog or picks songs, and its output is treated as untrusted: types are validated, values clamped, and genre hints capped, so a malformed response degrades to "ignore the mood."
- **App:** Streamlit, with in-app OAuth, taste-profile visualizations, recommendations, mood search, the evaluation, and one-click saving of any recommendation list as a real Spotify playlist.
- **Tools:** Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, spotipy, Anthropic API, pytest (42 unit tests, including 11 for mood search with a mocked client).

## Results

- **Catalog match rate: 3.5% → 61.7%** of my real listening history after adding the artist-level fallback.
- **Recommendation quality:** hit-rate@20 of **~3.6%** against a random baseline of **~0.03%**, roughly a **100x lift over chance**, across 28 held-out artist folds.
- Two ranking bugs were caught and fixed along the way: a flat genre bonus let a genre making up 1% of my taste compete equally with one making up 45%, and feature explanations compared raw features against a standardized centroid.

## Key takeaways

- **The data pipeline mattered more than the model.** Without the artist-level fallback, the recommender would have silently ignored most of my current taste in favour of whatever was old enough to be in a static file.
- **Offline evaluation turns "it seems good" into a number.** Leave-one-artist-out against a random baseline is the practical substitute for an A/B test when there's one user, and it's what tuned the genre weighting.
- **Use LLMs for narrow translation, not decisions.** Limiting the model to mapping language onto a validated schema keeps ranking deterministic and robust to bad responses.
- **One person's library is a small sample.** 28 held-out folds give wide confidence intervals, so the ~100x lift is strong evidence of real signal but not a precise estimate.

## How to run

1. Create a free app at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) with **Web API** checked and the Redirect URI set to `http://127.0.0.1:8501` (Spotify requires the literal `127.0.0.1`, not `localhost`).
2. Copy `.env.example` to `.env` and fill in your Client ID and Secret. Optionally add an `ANTHROPIC_API_KEY` to enable mood search. Everything else works without it.
3. From this folder:
   ```bash
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt
   .venv/Scripts/python scripts/build_dataset.py              # downloads the audio-feature catalog
   .venv/Scripts/streamlit run app.py --server.address 127.0.0.1
   ```
4. Run the tests with `.venv/Scripts/python -m pytest`.

## Repo structure

```
├── taste_engine/
│   ├── catalog.py          # loads and cleans the audio-feature catalog, artist-level fallback, z-scoring
│   ├── spotify_client.py   # OAuth, pulls listening history, creates playlists from recommendations
│   ├── features.py         # joins user tracks to the catalog (exact → name/artist → artist fallback)
│   ├── profile.py          # weighted taste profile (feature centroid + genre distribution)
│   ├── recommend.py        # cosine-similarity ranking, genre bonus, diversity cap, explanations
│   ├── evaluate.py         # precision / recall / NDCG@K, leave-one-artist-out evaluation
│   └── mood_query.py       # free-text mood → validated feature targets (LLM-assisted)
├── scripts/
│   ├── build_dataset.py    # downloads the public audio-feature catalog
│   ├── print_auth_url.py   # prints the Spotify OAuth URL (CLI flow)
│   ├── complete_login.py   # local callback listener that completes OAuth (CLI flow)
│   └── ingest.py           # pulls listening history to a local CSV (CLI flow)
├── tests/                  # pytest unit tests for every taste_engine module
├── app.py                  # Streamlit app
├── .env.example            # Spotify / Anthropic credential template
├── pyproject.toml
├── requirements.txt
└── README.md
```

