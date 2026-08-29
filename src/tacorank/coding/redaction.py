"""Credential-aware redaction for prompts, process output, and trajectories."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


REDACTED = "[REDACTED]"

_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|client[_-]?secret|"
    r"credential|passwd|password|private[_-]?key|secret|session[_-]?token)",
    re.IGNORECASE,
)
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization|client[_-]?secret|"
    r"password|private[_-]?key|secret|session[_-]?token)\b\s*[:=]\s*)"
    r"(\$\{[^}\r\n]+\}|\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}]+)"
)
_WELL_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,})\b"
)
_URL_CREDENTIAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:)([^\s/@]+)(@)")


class RedactionError(ValueError):
    """Raised when sensitive input cannot be safely redacted."""


class SecretRedactor:
    """Redact explicit credential values plus conservative credential shapes.

    Explicit values are sorted longest-first to avoid leaking a longer secret
    through replacement of one of its substrings.  They are intentionally not
    exposed by ``repr`` or public attributes.
    """

    def __init__(self, secret_values: Iterable[str] = ()) -> None:
        normalized = set()
        for value in secret_values:
            if not isinstance(value, str):
                raise RedactionError("secret values must be strings")
            if value:
                normalized.add(value)
        self._secrets: Tuple[str, ...] = tuple(
            sorted(normalized, key=lambda item: (-len(item), item))
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        credential_names: Sequence[str],
    ) -> "SecretRedactor":
        """Create a redactor from an explicit credential environment allowlist."""

        values = []
        for name in credential_names:
            if not isinstance(name, str) or not name or not _SENSITIVE_KEY_RE.search(name):
                raise RedactionError(f"credential environment name is not sensitive: {name!r}")
            value = environment.get(name)
            if value:
                values.append(value)
        return cls(values)

    def __repr__(self) -> str:
        return f"SecretRedactor(secret_count={len(self._secrets)})"

    def redact(self, text: str) -> str:
        """Return text with explicit and credential-shaped values removed."""

        if not isinstance(text, str):
            raise RedactionError("redaction input must be text")
        redacted = text
        for secret in self._secrets:
            redacted = redacted.replace(secret, REDACTED)
        redacted = _PEM_RE.sub(f"{REDACTED}:private-key", redacted)
        redacted = _BEARER_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", redacted)
        redacted = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
        redacted = _WELL_KNOWN_TOKEN_RE.sub(f"{REDACTED}:token", redacted)
        redacted = _URL_CREDENTIAL_RE.sub(
            lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}", redacted
        )
        return redacted

    def contains_known_secret(self, value: bytes) -> bool:
        """Return whether raw bytes contain any explicitly supplied credential."""

        return any(secret.encode("utf-8") in value for secret in self._secrets)

    def contains_credential_material(self, text: str) -> bool:
        """Detect known or credential-shaped material while allowing placeholders."""

        if self.contains_known_secret(text.encode("utf-8")):
            return True
        if (
            _PEM_RE.search(text)
            or _BEARER_RE.search(text)
            or _WELL_KNOWN_TOKEN_RE.search(text)
            or _URL_CREDENTIAL_RE.search(text)
        ):
            return True
        for match in _ASSIGNMENT_RE.finditer(text):
            raw_value = match.group(2).strip().strip("\"'")
            lowered = raw_value.lower()
            is_placeholder = (
                lowered in {"", "null", "none", "unset", "placeholder"}
                or lowered.startswith("your_")
                or (raw_value.startswith("${") and raw_value.endswith("}"))
                or (raw_value.startswith("$") and raw_value[1:].replace("_", "").isalnum())
            )
            if not is_placeholder:
                return True
        return False

    def redact_object(self, value: Any) -> Any:
        """Recursively redact sensitive JSON keys and all string leaves."""

        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                key_text = str(key)
                if _SENSITIVE_KEY_RE.search(key_text):
                    result[key_text] = REDACTED
                else:
                    result[key_text] = self.redact_object(item)
            return result
        if isinstance(value, list):
            return [self.redact_object(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact_object(item) for item in value]
        if isinstance(value, str):
            return self.redact(value)
        return value

    def redact_json_bytes(self, value: bytes, *, max_bytes: int = 50 * 1024 * 1024) -> bytes:
        """Validate a JSON document and return deterministic redacted JSON bytes."""

        if len(value) > max_bytes:
            raise RedactionError("JSON document exceeds the redaction size limit")
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RedactionError("JSON document is not UTF-8") from exc
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise RedactionError("malformed JSON document") from exc
        redacted = self.redact_object(parsed)
        rendered = json.dumps(
            redacted,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )
        return (rendered + "\n").encode("utf-8")


def redact_optional(text: Optional[str], redactor: SecretRedactor) -> Optional[str]:
    """Redact an optional text field without converting ``None`` to a string."""

    return None if text is None else redactor.redact(text)
