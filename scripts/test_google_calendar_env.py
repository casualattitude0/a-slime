from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv


def _mask(value: str, keep: int = 4) -> str:
    v = (value or "").strip()
    if len(v) <= keep * 2:
        return "*" * len(v)
    return f"{v[:keep]}...{v[-keep:]}"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    token = (os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN") or "").strip()
    calendar_id = (os.environ.get("GOOGLE_CALENDAR_ID") or "").strip()

    print("== Google Calendar ENV Test ==")
    print(f"GOOGLE_CALENDAR_ACCESS_TOKEN: {'set' if token else 'missing'} ({_mask(token) if token else ''})")
    print(f"GOOGLE_CALENDAR_ID: {'set' if calendar_id else 'missing'} ({calendar_id or ''})")

    if not token:
        print("FAIL: GOOGLE_CALENDAR_ACCESS_TOKEN is missing")
        return 2
    if not calendar_id:
        print("FAIL: GOOGLE_CALENDAR_ID is missing")
        return 2

    headers = {"Authorization": f"Bearer {token}"}
    base = "https://www.googleapis.com/calendar/v3"

    with httpx.Client(timeout=20.0) as client:
        try:
            print("\n[1/3] Testing calendar id format/access via events endpoint ...")
            encoded_id = quote(calendar_id, safe="")
            now = datetime.now(timezone.utc)
            start_dt = now + timedelta(minutes=1)
            end_dt = start_dt + timedelta(minutes=5)
            payload = {
                "summary": "Agent token/id test event",
                "description": "Auto-created by scripts/test_google_calendar_env.py",
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
            }
            create_resp = client.post(
                f"{base}/calendars/{encoded_id}/events",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
            )
            if create_resp.status_code not in (200, 201):
                print(f"FAIL: create event failed ({create_resp.status_code})")
                print(create_resp.text[:1200])
                print(
                    "\nHint: verify GOOGLE_CALENDAR_ID is a calendar id and token has calendar.events scope."
                )
                return 1
            created = create_resp.json()
            event_id = created.get("id", "")
            print("PASS: create event succeeded")
            print(
                json.dumps(
                    {
                        "event_id": event_id,
                        "htmlLink": created.get("htmlLink"),
                        "status": created.get("status"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print("\n[2/3] Verifying created event by GET ...")
            get_resp = client.get(
                f"{base}/calendars/{encoded_id}/events/{quote(event_id, safe='')}",
                headers=headers,
            )
            if get_resp.status_code != 200:
                print(f"FAIL: get created event failed ({get_resp.status_code})")
                print(get_resp.text[:1200])
                return 1
            fetched = get_resp.json()
            print("PASS: get event succeeded")
            print(
                json.dumps(
                    {
                        "event_id": fetched.get("id"),
                        "summary": fetched.get("summary"),
                        "status": fetched.get("status"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print("\n[3/3] Cleaning up test event by DELETE ...")
            del_resp = client.delete(
                f"{base}/calendars/{encoded_id}/events/{quote(event_id, safe='')}",
                headers=headers,
            )
            if del_resp.status_code not in (200, 204):
                print(f"WARN: delete test event failed ({del_resp.status_code})")
                print(del_resp.text[:1200])
                return 1
            print("PASS: delete event succeeded")
            print("\nALL PASS: token + calendar id are working for event operations.")
            return 0
        except Exception as exc:
            print(f"FAIL: request error: {exc}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
