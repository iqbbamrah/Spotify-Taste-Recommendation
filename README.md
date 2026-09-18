# Spotify Projects

Two independent projects built on Spotify data, kept in one repo since they share a
domain (and one dataset) but explore different skills.

## [spotify-taste-recommender](spotify-taste-recommender/)

A content-based music recommender built on **real** Spotify listening history pulled live via
OAuth — taste profiling, cosine-similarity recommendations, an LLM-assisted mood search, and an
offline leave-one-artist-out evaluation against a random baseline. Streamlit app.

## [spotify-genre-nn](spotify-genre-nn/)

The same small feedforward genre classifier implemented three times — raw TensorFlow, Keras, and
PyTorch — on identical data and splits, to compare the frameworks directly rather than just read
about them. Both projects draw on the same public [Spotify Tracks
Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).

Each subfolder has its own README with full details, setup instructions, and results.
