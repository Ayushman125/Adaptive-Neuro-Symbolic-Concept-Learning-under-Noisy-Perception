import re


_REDACTION_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)(api[_-]?key|token|authorization)\s*[:=]\s*['\"]?([A-Za-z0-9._\-]{6,})"),
    re.compile(r"(?i)(api_key=)([^&\s]+)"),
    re.compile(r"(?i)(access_token=)([^&\s]+)"),
]


def redact_sensitive(text):
    if not text:
        return text
    redacted = str(text)
    for pattern in _REDACTION_PATTERNS:
        redacted = pattern.sub(r"\1***", redacted)
    return redacted


__all__ = ["redact_sensitive"]
