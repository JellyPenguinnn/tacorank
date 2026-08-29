"""Git lineage primitives for TacoRank experiments."""

from .patches import (
    NormalizedPatch,
    WrittenArtifact,
    capture_commit_range,
    capture_commit_patch,
    commit_staged_patch,
    stage_and_capture,
    validate_relative_path,
    write_artifact,
)
from .refs import (
    GitOperationError,
    best_ref,
    branch_tip,
    experiment_branch,
    is_ancestor,
    require_ancestor,
    resolve_commit,
    update_best_ref,
    validate_object_id,
    validated_repository,
)
from .worktrees import WorktreeLease, WorktreeManager, WorktreeRecord

__all__ = [
    "GitOperationError",
    "NormalizedPatch",
    "WorktreeManager",
    "WorktreeLease",
    "WorktreeRecord",
    "WrittenArtifact",
    "best_ref",
    "branch_tip",
    "capture_commit_range",
    "capture_commit_patch",
    "commit_staged_patch",
    "experiment_branch",
    "is_ancestor",
    "require_ancestor",
    "resolve_commit",
    "stage_and_capture",
    "update_best_ref",
    "validate_object_id",
    "validate_relative_path",
    "validated_repository",
    "write_artifact",
]
