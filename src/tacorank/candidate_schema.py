"""Controller-owned schemas for candidate-visible KuaiRand CSV views."""

from __future__ import annotations


CANDIDATE_TRAIN_BASE_COLUMNS = (
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "long_view",
)

CANDIDATE_TRAIN_AUXILIARY_COLUMNS = (
    "time_ms",
    "hourmin",
    "is_click",
    "play_time_ms",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)

# Keep this order identical to the materialized CSV. It is intentionally not
# base-columns + auxiliary-columns because the temporal fields are placed next
# to the date before the entity identifiers.
CANDIDATE_TRAIN_COLUMNS = (
    "date",
    "time_ms",
    "hourmin",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "long_view",
    "is_click",
    "play_time_ms",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)

CANDIDATE_SCORE_COLUMNS = (
    "row_id",
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
)

# Deployments created before auxiliary training labels were exposed remain
# readable for deterministic resume and historical inspection.
APPROVED_CANDIDATE_TRAIN_SCHEMAS = (
    CANDIDATE_TRAIN_BASE_COLUMNS,
    CANDIDATE_TRAIN_COLUMNS,
)
