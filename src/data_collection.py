from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_MAX_TEXT_LEN = 4000
_WRITE_LOCK = threading.Lock()


def _telemetry_dir(root: Path) -> Path:
    p = root / "data" / "telemetry"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_text(value: Any, *, max_len: int = _MAX_TEXT_LEN) -> str:
    s = value if isinstance(value, str) else str(value or "")
    s = s.replace("\x00", "")
    if len(s) > max_len:
        return s[:max_len]
    return s


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            kl = str(k).lower()
            if "authorization" in kl or "api_key" in kl or "apikey" in kl:
                out[str(k)] = "[REDACTED]"
            else:
                out[str(k)] = _sanitize_payload(v)
        return out
    if isinstance(value, list):
        return [_sanitize_payload(v) for v in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_agent_event(
    root: Path,
    *,
    event_type: str,
    session_id: str,
    version_id: str | None = None,
    request_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    record = {
        "event_id": event_id,
        "event_type": event_type,
        "session_id": session_id,
        "version_id": version_id or "",
        "request_id": request_id or "",
        "timestamp": time.time(),
        "payload": _sanitize_payload(payload or {}),
    }
    _append_jsonl(_telemetry_dir(root) / "agent_events.jsonl", record)
    return event_id


def write_feedback(
    root: Path,
    *,
    session_id: str,
    message_ref: str,
    rating: int,
    comment: str | None = None,
) -> str:
    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id": feedback_id,
        "session_id": session_id,
        "message_ref": message_ref,
        "rating": rating,
        "comment": _sanitize_text(comment or ""),
        "timestamp": time.time(),
    }
    _append_jsonl(_telemetry_dir(root) / "user_feedback.jsonl", record)
    return feedback_id


def read_recent_agent_events(root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    path = _telemetry_dir(root) / "agent_events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result: list[dict[str, Any]] = []
    for line in lines[-max(1, min(limit, 500)) :]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            result.append(obj)
    return result


def read_recent_feedback(root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    path = _telemetry_dir(root) / "user_feedback.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result: list[dict[str, Any]] = []
    for line in lines[-max(1, min(limit, 500)) :]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            result.append(obj)
    return result
