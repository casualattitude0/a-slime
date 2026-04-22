"""LLM provider error classification."""

from __future__ import annotations

import re
from typing import Any


class LLMErrorInfo:
    """Structured info extracted from an LLM provider exception."""

    __slots__ = ("is_llm_error", "error_type", "message", "retry_after_seconds")

    def __init__(
        self,
        *,
        is_llm_error: bool,
        error_type: str,
        message: str,
        retry_after_seconds: float | None,
    ) -> None:
        self.is_llm_error = is_llm_error
        self.error_type = error_type
        self.message = message
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_llm_error": self.is_llm_error,
            "error_type": self.error_type,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
        }


def parse_retry_delay(text: str) -> float | None:
    m = re.search(r"retry[_\s](?:in|after)[:\s]+([0-9]+(?:\.[0-9]+)?)\s*s", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay[\"']?\s*:\s*[\"']([0-9]+(?:\.[0-9]+)?)s[\"']", text)
    if m:
        return float(m.group(1))
    return None


_LLM_STATUS_CODES = {401, 403, 404, 429, 500, 502, 503, 504}
_LLM_ERROR_KEYWORDS = (
    "resource_exhausted",
    "rate_limit",
    "ratelimit",
    "quota",
    "too many requests",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "overloaded",
    "timeout",
)


def rich_exception_message(exc: BaseException) -> str:
    """Include HTTP response body when present."""
    lines: list[str] = []
    seen: set[int] = set()
    e: BaseException | None = exc
    depth = 0
    while e is not None and depth < 12:
        if id(e) in seen:
            break
        seen.add(id(e))
        lines.append(str(e))
        resp = getattr(e, "response", None)
        if resp is not None:
            body = getattr(resp, "text", "") or ""
            if isinstance(body, str) and body.strip():
                snippet = body.strip()
                if len(snippet) > 8000:
                    snippet = snippet[:8000] + "\n… [truncated]"
                lines.append(f"--- HTTP response ---\n{snippet}")
        e = e.__cause__
        depth += 1

    return "\n\n".join(lines)


def classify_llm_error(exc: BaseException) -> LLMErrorInfo:
    """Return LLMErrorInfo for any exception; ``is_llm_error`` is True only for provider errors."""
    raw = rich_exception_message(exc)
    text = raw.lower()

    is_llm = False
    error_type = "unknown"

    for kw in _LLM_ERROR_KEYWORDS:
        if kw in text:
            is_llm = True
            break

    for code in _LLM_STATUS_CODES:
        if str(code) in raw:
            is_llm = True
            break

    if not is_llm and any(
        k in text for k in ("not found", "invalid api", "nvidia", "nim", "client error", "server error")
    ):
        is_llm = True

    if isinstance(exc, TimeoutError):
        is_llm = True

    if is_llm:
        if "resource_exhausted" in text or "429" in raw or "quota" in text or "rate" in text:
            error_type = "quota_exceeded"
        elif isinstance(exc, TimeoutError) or "timeout" in text:
            error_type = "timeout"
        elif any(k in text for k in ("unavailable", "bad gateway", "gateway timeout", "overloaded")):
            error_type = "service_unavailable"
        elif "404" in raw or "not found" in text:
            error_type = "provider_error"
        else:
            error_type = "provider_error"

    retry_after = parse_retry_delay(raw) if is_llm else None

    return LLMErrorInfo(
        is_llm_error=is_llm,
        error_type=error_type,
        message=raw,
        retry_after_seconds=retry_after,
    )


__all__ = ["LLMErrorInfo", "classify_llm_error", "rich_exception_message"]
