from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _import_google_oauth():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install with: "
            ".venv/bin/pip install google-auth google-auth-oauthlib google-auth-httplib2"
        ) from exc
    return Request, Credentials, InstalledAppFlow


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    Request, Credentials, InstalledAppFlow = _import_google_oauth()

    client_secret_path = (
        os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET_FILE")
        or str(root / "client_secret.json")
    )
    token_cache_path = (
        os.environ.get("GOOGLE_OAUTH_TOKEN_CACHE_FILE")
        or str(root / "google_oauth_token.json")
    )

    client_secret = Path(client_secret_path)
    token_cache = Path(token_cache_path)

    if not client_secret.is_file():
        print("FAIL: client secret file not found")
        print(f"Expected: {client_secret}")
        print(
            "Set GOOGLE_OAUTH_CLIENT_SECRET_FILE in .env or place OAuth client JSON at ./client_secret.json"
        )
        return 2

    creds = None
    if token_cache.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(token_cache), SCOPES)
        except Exception:
            creds = None

    oauth_port_raw = (os.environ.get("GOOGLE_OAUTH_LOCAL_PORT") or "8080").strip()
    try:
        oauth_port = int(oauth_port_raw)
    except ValueError:
        oauth_port = 8080

    redirect_hint = f"http://localhost:{oauth_port}/"
    print(f"OAuth redirect URI expected by this script: {redirect_hint}")

    if creds and creds.valid:
        pass
    elif creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        creds = flow.run_local_server(port=oauth_port)

    if not creds:
        print("FAIL: unable to obtain credentials")
        return 1

    token_cache.write_text(creds.to_json(), encoding="utf-8")

    print("PASS: OAuth token acquired")
    print(f"Token cache: {token_cache}")
    print("")
    print("Use these values in .env:")
    print(f"GOOGLE_CALENDAR_ACCESS_TOKEN={creds.token}")
    if creds.refresh_token:
        print(f"GOOGLE_CALENDAR_REFRESH_TOKEN={creds.refresh_token}")
    print("")
    payload = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    print("Token payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
