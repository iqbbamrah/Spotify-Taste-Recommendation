"""CLI: pull the logged-in user's real Spotify listening history and save it
locally. Run this after setting up .env with your Spotify app credentials.

    python scripts/ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste_engine.spotify_client import fetch_all_listening_history, get_client  # noqa: E402

OUTPUT_PATH = Path("data/raw/user_tracks_raw.csv")


def main() -> None:
    load_dotenv()
    sp = get_client()
    me = sp.current_user()
    print(f"Authenticated as: {me['display_name']} ({me['id']})")

    history = fetch_all_listening_history(sp)
    print(f"Pulled {len(history)} track entries across sources:")
    print(history["source"].value_counts().to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
