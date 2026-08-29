"""Deterministic, user-safe evaluation population construction."""

from collections import defaultdict
import hashlib
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def split_validation_indices(
    user_ids: Sequence[object],
    val_a_ratio: float = 0.8,
    salt: str = "ladder-2026",
) -> Tuple[List[int], List[int]]:
    """Split by user with a stable hash; a user never appears in both arms."""
    if not 0.0 < val_a_ratio < 1.0:
        raise ValueError("val_a_ratio must be between zero and one")
    threshold = int(round(val_a_ratio * 10_000))
    arm_cache: Dict[str, bool] = {}
    val_a: List[int] = []
    val_b: List[int] = []
    for index, raw_user_id in enumerate(user_ids):
        user_id = str(raw_user_id)
        if user_id not in arm_cache:
            arm_cache[user_id] = _hash_int(salt, user_id) % 10_000 < threshold
        (val_a if arm_cache[user_id] else val_b).append(index)
    return val_a, val_b


def build_internal_proxy(
    rows: Sequence[Mapping[str, object]],
    holdout_start: int,
    holdout_end: int,
    impressions_per_user: int = 5,
    date_key: str = "date",
    user_key: str = "user_id",
    salt: str = "internal-proxy-2026",
) -> List[Mapping[str, object]]:
    """Select a temporal holdout and deterministically cap each user's rows."""
    if impressions_per_user <= 0:
        raise ValueError("impressions_per_user must be positive")
    grouped: Dict[str, List[Tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for original_index, row in enumerate(rows):
        date = int(row[date_key])
        if holdout_start <= date <= holdout_end:
            grouped[str(row[user_key])].append((original_index, row))
    selected: List[Tuple[int, Mapping[str, object]]] = []
    for user_id, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda pair: (_hash_int(salt, user_id, str(pair[0])), pair[0]),
        )[:impressions_per_user]
        selected.extend(ranked)
    selected.sort(key=lambda pair: pair[0])
    return [row for _, row in selected]


def restrict_random_log(
    rows: Iterable[Mapping[str, object]],
    start: int = 20220422,
    end: int = 20220428,
    date_key: str = "date",
) -> List[Mapping[str, object]]:
    return [row for row in rows if start <= int(row[date_key]) <= end]


def _hash_int(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
