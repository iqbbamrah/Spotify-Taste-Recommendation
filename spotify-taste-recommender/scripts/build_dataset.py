"""Downloads the public audio-feature catalog (114K-track 'Spotify Tracks
Dataset', mirrored on Hugging Face - no login required) used as the
candidate pool and feature source, since Spotify's own Audio Features
endpoint is closed to new developer apps.

    python scripts/build_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import requests

CATALOG_URL = (
    "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset"
    "/resolve/main/dataset.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data/raw/spotify_tracks_dataset.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading catalog from {CATALOG_URL} ...")
    response = requests.get(CATALOG_URL, timeout=60)
    response.raise_for_status()
    OUTPUT_PATH.write_bytes(response.content)
    print(f"Saved {len(response.content) / 1e6:.1f} MB to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
