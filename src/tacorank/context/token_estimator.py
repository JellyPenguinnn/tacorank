"""Deterministic conservative token estimator.

Provider tokenizers are optional.  The control plane uses a stable upper-bound
estimate so the same source events always produce the same packing decision.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    # UTF-8 bytes / 3 is conservative for English/code and deterministic across
    # Python builds.  Always charge at least one token for non-empty text.
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 2) // 3)
