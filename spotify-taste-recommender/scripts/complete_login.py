"""Blocks waiting for the Spotify OAuth redirect on 127.0.0.1:8888/callback,
exchanges the code for a token, and caches it to .spotify_token_cache so
later scripts (ingest.py, app.py) don't need to re-authenticate.

Run this in the background, then open the URL from print_auth_url.py in a
real browser and approve access.
"""
from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taste_engine.spotify_client import get_auth_manager  # noqa: E402

captured: dict[str, str] = {}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if "code" in qs:
            captured["code"] = qs["code"][0]
            self.wfile.write(b"<h2>Spotify auth complete. You can close this tab.</h2>")
        else:
            captured["error"] = qs.get("error", ["unknown_error"])[0]
            self.wfile.write(f"<h2>Auth failed: {captured['error']}</h2>".encode())

    def log_message(self, format, *args):  # silence default request logging
        pass


def main() -> None:
    load_dotenv()
    auth_manager = get_auth_manager()

    print("Waiting for Spotify redirect on http://127.0.0.1:8888/callback ...", flush=True)
    server = HTTPServer(("127.0.0.1", 8888), CallbackHandler)
    while "code" not in captured and "error" not in captured:
        server.handle_request()

    if "error" in captured:
        print(f"AUTH_ERROR: {captured['error']}", flush=True)
        sys.exit(1)

    auth_manager.get_access_token(captured["code"], as_dict=True, check_cache=False)
    print("AUTH_SUCCESS: token cached to .spotify_token_cache", flush=True)


if __name__ == "__main__":
    main()
