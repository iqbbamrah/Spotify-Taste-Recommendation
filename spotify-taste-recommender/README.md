# Your Spotify Taste, Mapped

A content-based music recommender built on my **real** Spotify listening history — not a Kaggle demo dataset. It pulls your actual top tracks, recently played, saved songs, and playlists, builds a quantitative "taste profile" from them, and recommends real songs you don't already have, with an explanation for each one and an offline evaluation of whether the recommendations are actually any good.

Live app: *(add your Streamlit Community Cloud URL here after deploying)*

## Why this exists

I wanted a project that was genuinely useful to me (I listen to a lot of music) while exercising the same muscles product data science roles actually test: turning ambiguous behavioral data into a defensible model, being honest about data limitations, and being able to answer *"how do you know this works?"* with something better than a vibe.

The interesting part of this project isn't the recommender math — cosine similarity over audio features is standard. It's the data pipeline: **Spotify closed its Audio Features, Recommendations, Related Artists, and Artist-Top-Tracks endpoints to new developer apps in late 2024**, and I confirmed empirically (not just from documentation) that even artist genres and popularity are stripped from a new app's responses. That kills the obvious approach of calling Spotify for enrichment. See [How the pipeline actually works](#how-the-pipeline-actually-works) for how this was worked around, including a real data bug I hit and fixed along the way (match rate went from 3.5% to 61.7%).

## What it does

1. You log in with your own Spotify account (OAuth — I never see your password).
2. It pulls your real top tracks (short/medium/long-term), recently played, saved tracks, and playlists.
3. It matches those tracks against a public audio-feature catalog to recover danceability, energy, valence, acousticness, tempo, genre, etc. — features Spotify's own API won't hand a new app anymore.
4. It builds a weighted "taste profile": an audio-feature centroid + genre distribution, weighted so tracks you've saved or kept in your top-tracks for a long time count more than a song that shuffled past once.
5. It ranks a 77K-track candidate catalog by similarity to your profile (cosine similarity + a genre-weighted bonus), filters out anything you already have, and caps how many picks come from any one artist so the list isn't one artist five times.
6. It explains *why* each track was recommended (which audio features it's closest to, whether the genre matches your listening).
7. You can also describe a mood or moment in plain English ("something moody for a rainy commute") and it'll blend that onto your taste profile rather than replacing it — an LLM only translates the request into structured audio-feature targets, it never picks songs itself (see [Mood search](#mood-search)).
8. It evaluates itself: leave-one-artist-out testing checks whether an artist you demonstrably like would have been rediscovered from your *other* listening, compared against a random-chance baseline.

## How the pipeline actually works

### The data problem, and how I found it

Spotify's `audio-features`, `recommendations`, `related-artists`, and `artist-top-tracks` endpoints return `403 Forbidden` for apps created after Spotify's November 2024 policy change — I hit this directly while building (see `403` errors were reproduced against the live API). Even artist objects returned by search/lookup came back with `genres: None` and `popularity: None`. In practice, a new personal Spotify app can only reliably read: **your own library** (top tracks, recently played, saved tracks, playlists) and basic track/album metadata (name, release date, duration).

The workaround: source audio features from a free, public, no-login-required catalog instead — the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (114K tracks, mirrored on Hugging Face, deduplicated down to ~77K unique tracks with danceability/energy/valence/acousticness/tempo/genre/popularity).

**First attempt matched on exact track — and only covered 3.5% of my real library.** A frozen 2023-era catalog simply doesn't have most of what I actually listen to (current hip-hop/trap: Travis Scott B-sides, Lil Uzi Vert, Nemzzz, etc.). That's not a viable foundation for a taste profile — it would silently and systematically ignore your most current taste in favor of whatever happens to be old enough to be in a static file.

**Fix: an artist-level fallback.** Even when the *exact song* isn't in the catalog, the *artist* often has other tracks in it. `catalog.build_artist_feature_table()` builds a per-artist average audio-feature profile (crediting every collaborator on a track, not just the primary artist), and `features.join_user_tracks()` falls back to it when an exact match fails. That took real match rate from **3.5% → 61.7%** on my own listening history — the artist-level signal is what makes the rest of the pipeline viable at all.

### Taste profiling

`profile.build_taste_profile()` builds a weighted centroid over 9 standardized audio-feature dimensions, plus a normalized genre distribution. Weights come from `profile.compute_track_weights()`: saved tracks and top-tracks (short-term) count fully, long-term top tracks and playlist tracks count somewhat less, and recently-played counts least (it captures background listening, shuffle, and skips — a noisier signal of actually liking a song than showing up in your top tracks repeatedly).

### Recommending

`recommend.score_candidates()` ranks the full catalog by cosine similarity to the taste centroid, with a genre bonus *scaled by how much that genre matters in your profile* (not a flat "genre matches, yes/no" bonus — an earlier version of this bug let a 1%-of-your-taste genre compete equally with a 45%-of-your-taste genre, which visibly skewed recommendations toward a single unrepresentative regional sub-scene before it was caught and fixed). `_apply_diversity_cap()` limits how many picks come from one artist. `explain_recommendation()` reports which specific audio features are closest to your profile, standardized into the same z-scored space the model actually reasons in (an earlier bug compared raw features against a standardized centroid — different scales, meaningless comparison — also caught and fixed; see commit history).

### Evaluating

There's no way to run a live A/B test on "is this a good song recommendation" the way a product team could test a UI change. `evaluate.py` implements the standard offline substitute:

- **Leave-one-artist-out**: for every artist I demonstrably like, rebuild the taste profile *without* that artist, then check whether their other catalog tracks land back in my top-K recommendations. Report as hit-rate@K, precision@K, recall@K, NDCG@K.
- Compared against a **random-baseline** (K / candidate pool size) to contextualize whether a small-looking number is actually meaningful. On my own data: hit-rate@20 of ~3.6% against a random baseline of ~0.03% — roughly **a 100x lift over chance**, from 28 held-out artist folds.
- I'm upfront that 28 folds gives wide confidence intervals — this is a real limitation of evaluating against one person's personal library rather than a large user base, and the honest caveat is more useful than a false-precision number.

This evaluation harness is also what justified the genre-bonus weighting (`GENRE_BONUS_SCALE` in `recommend.py`) — chosen by sweeping values and checking hit-rate@K, not by eyeballing the output.

### Mood search

`mood_query.py` handles free-text requests like "hype for a workout." The LLM (Claude, via a forced tool call) is used for exactly one narrow job — mapping messy natural language onto a fixed schema of standardized audio-feature targets (roughly in [-2, 2]) and a handful of genre hints — and nothing else. It never sees the catalog and never picks a track; its only output feeds `apply_mood_to_profile()`, a pure function that linearly blends the mood's targets onto your existing taste centroid (a `strength` slider controls how far to lean into the mood vs. your regular taste), which then goes through the exact same `score_candidates()` ranking as every other recommendation path.

Worth calling out for anyone reading this as a portfolio piece: the LLM output is treated as untrusted input, not a trusted decision. `_parse_mood_payload()` validates types, clamps every feature value into a fixed range, and caps genre hints at 5 — so a malformed or adversarial response degrades to "ignore the mood" rather than breaking the ranking pipeline. All of this is unit-tested with a mocked client (11 tests), so the logic is fully verified independent of whether the API key in use has credit on it.

This is an optional feature — it needs an `ANTHROPIC_API_KEY` with available credit (a few cents per query, not a subscription). Without one, the rest of the app works exactly the same; the mood tab just shows a message asking for a key.

## Project structure

```
taste_engine/
  catalog.py       # loads + cleans the public audio-feature catalog, artist-level fallback table,
                    # feature standardization (z-scoring)
  spotify_client.py# OAuth + pulls real listening history via endpoints still open to new apps
  features.py      # joins user tracks -> catalog (exact match -> name/artist match -> artist-level fallback)
  profile.py        # builds the weighted taste profile (feature centroid + genre distribution)
  recommend.py      # cosine-similarity ranking, genre bonus, diversity cap, explanations
  evaluate.py        # precision/recall/NDCG@k, leave-one-artist-out evaluation
  mood_query.py       # free-text mood -> structured feature targets (LLM-assisted, validated/clamped)

scripts/
  build_dataset.py   # downloads the public audio-feature catalog
  print_auth_url.py  # prints the Spotify OAuth URL (for the CLI flow)
  complete_login.py  # local callback listener that completes OAuth + caches a token (CLI flow)
  ingest.py           # pulls your listening history to a local CSV (CLI flow, for offline analysis)

app.py                # Streamlit app: in-app OAuth, taste profile viz, recommendations, mood search, evaluation
tests/                # pytest unit tests for every taste_engine module (42 tests)
data/raw/              # downloaded catalog + your own pulled history (git-ignored)
```

## Running it locally

1. **Create a free Spotify Developer app** at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create app → check **Web API** only → set Redirect URI to `http://127.0.0.1:8501` (note: Spotify requires the literal `127.0.0.1`, not `localhost`).
2. Copy `.env.example` to `.env` and fill in your Client ID/Secret, or put them in `.streamlit/secrets.toml` (same keys) for the Streamlit app. Optionally add an `ANTHROPIC_API_KEY` (from [console.anthropic.com](https://console.anthropic.com), a workspace-scoped key with available credit) to enable mood search — everything else works without it.
3. Install dependencies:
   ```
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
   ```
4. Download the public catalog:
   ```
   python scripts/build_dataset.py
   ```
5. Run the app:
   ```
   .venv/Scripts/streamlit run app.py --server.address 127.0.0.1
   ```
6. Run the tests:
   ```
   .venv/Scripts/python -m pytest
   ```

## What I'd do with more data

The evaluation is honest about its own limitation: it's one person's library, so 28 held-out folds is a small sample. With more usage (multiple users, or the same account over more months), the natural next steps are: a proper train/test split by time instead of leave-one-out, confidence intervals on the hit-rate estimate (Wilson score interval, given the small-sample skew of a raw proportion), and a genuinely held-out "did I actually save any of these" online metric instead of only an offline proxy.

## Tech stack

Python, pandas, numpy, scikit-learn (cosine similarity), Streamlit, Plotly, spotipy (Spotify OAuth), pytest.
