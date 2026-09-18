"""Prints the Spotify authorization URL to open in a real browser."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taste_engine.spotify_client import get_auth_manager  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    auth_manager = get_auth_manager()
    print(auth_manager.get_authorize_url())
