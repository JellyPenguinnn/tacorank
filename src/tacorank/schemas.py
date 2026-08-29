"""Strict shared contracts used by execution, SRE, and recovery.

This module deliberately contains only the schema slice needed at those
boundaries.  Durable-event validation and the complete planning/evaluation
schemas belong to their respective components.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactKind(str, Enum):
    DIFF = "diff"
    TRAJECTORY = "trajectory"
    CONTEXT = "context"
    LOG = "log"
    CHECKPOINT = "checkpoint"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    VERIFICATION_RECEIPT = "verification_receipt"
    SUBMISSION = "submission"
    REPORT = "report"
    OTHER = "other"


class TokenMeasurement(str, Enum):
    PROVIDER = "provider"
    ESTIMATED = "estimated"
    NONE = "none"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class Fidelity(str, Enum):
    SMOKE = "smoke"
    PROXY = "proxy"
    FULL = "full"
    FINAL = "final"


class ExecutionOutcome(str, Enum):
    SUCCESS = "success"
    CODE_ERROR = "code_error"
    INTERFACE_ERROR = "interface_error"
    CONTRACT_ERROR = "contract_error"
    NUMERICAL_ERROR = "numerical_error"
    OOM = "oom"
    TIMEOUT = "timeout"
    HANG = "hang"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"


class MonitorAction(str, Enum):
    CONTINUE = "continue"
    TERMINATE = "terminate"


class RecoveryAction(str, Enum):
    TRAE_REPAIR = "trae_repair"
    RETRY_SAME_COMMIT = "retry_same_commit"
    ADJUST_APPROVED_RUNTIME_SETTING = "adjust_approved_runtime_setting"
    ROLLBACK = "rollback"
    ABANDON = "abandon"


class LessonOrigin(str, Enum):
    OPERATIONAL = "operational"
    RESEARCH = "research"


class LessonCategory(str, Enum):
    RESEARCH_RESULT = "research_result"
    RESOURCE_CONSTRAINT = "resource_constraint"
    IMPLEMENTATION_CONSTRAINT = "implementation_constraint"
    INTEGRITY_WARNING = "integrity_warning"
    PROCESS_RULE = "process_rule"


class EvaluationPopulation(str, Enum):
    INTERNAL_PROXY = "internal_proxy"
    PUBLIC_VALIDATION = "public_validation"
    HIDDEN_FINAL = "hidden_final"


class TrustVerdict(str, Enum):
    ACCEPTED = "accepted"
    INCONCLUSIVE = "inconclusive"
    NEGATIVE = "negative"
    NO_OP = "no_op"
    SUSPICIOUS = "suspicious"


class TrustStability(str, Enum):
    SINGLE_SEED = "single_seed"
    CONFIRMED = "confirmed"
    UNSTABLE = "unstable"
    NOT_APPLICABLE = "not_applicable"


class TrustIntegrity(str, Enum):
    CLEAN = "clean"
    COMPROMISED = "compromised"
    INCONCLUSIVE = "inconclusive"


def _validate_sha256(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError("must be 64 lowercase hexadecimal characters")
    return value


def _validate_repo_path(value: str) -> str:
    if not value or value == "." or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("must be a normalized repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ValueError("must be a normalized repository-relative path without '..'")
    if any(part in ("", ".") for part in path.parts):
        raise ValueError("must not contain empty or '.' path segments")
    return value


def _finite(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("must be finite")
    return value


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str | None

    _path_is_normalized = field_validator("path")(_validate_repo_path)
    _sha256_is_valid = field_validator("sha256")(_validate_sha256)


class ResourceDelta(StrictModel):
    llm_input_tokens: int = Field(ge=0)
    llm_output_tokens: int = Field(ge=0)
    token_measurement: TokenMeasurement
    wall_time_ms: int = Field(ge=0)
    cpu_time_ms: int = Field(ge=0)
    gpu_time_ms: int = Field(ge=0)
    gpu_count: int = Field(ge=0)
    peak_rss_mb: int | None = Field(ge=0)
    peak_gpu_memory_mb: int | None = Field(ge=0)
    manual_interventions: int = Field(ge=0)


class CheckResult(StrictModel):
    name: str = Field(min_length=1)
    status: CheckStatus
    details: str | None


class Violation(StrictModel):
    code: str = Field(min_length=1)
    path: str | None = None
    message: str = Field(min_length=1)

    _path_is_normalized = field_validator("path")(
        lambda value: _validate_repo_path(value) if value is not None else value
    )


class TelemetrySample(StrictModel):
    timestamp: datetime
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    elapsed_ms: int = Field(ge=0)
    process_alive: bool
    last_output_age_ms: int = Field(ge=0)
    cpu_percent: float = Field(ge=0)
    rss_mb: int = Field(ge=0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_memory_mb: int | None = Field(default=None, ge=0)
    loss: float | None = None
    gradient_norm: float | None = None
    disk_free_mb: int | None = Field(default=None, ge=0)
    recent_output_tail: str | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    # Loss/gradient are raw anomaly signals and intentionally permit NaN/Inf;
    # the observer must see them so it can issue NUMERICAL_NONFINITE.
    _finite_percentages = field_validator("cpu_percent", "gpu_utilization_percent")(_finite)


class MonitorDirective(StrictModel):
    action: MonitorAction
    reason_code: str | None
    summary: str | None

    @model_validator(mode="after")
    def fields_match_action(self) -> "MonitorDirective":
        if self.action is MonitorAction.CONTINUE and (
            self.reason_code is not None or self.summary is not None
        ):
            raise ValueError("continue directives cannot report a termination reason")
        if self.action is MonitorAction.TERMINATE and (
            not self.reason_code or not self.summary
        ):
            raise ValueError("terminate directives require reason_code and summary")
        return self


class PatchCheckResult(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    patch_commit_sha: str = Field(min_length=1)
    diff_sha256: str
    accepted: bool
    receipt_id: str | None
    receipt_artifact: ArtifactRef | None
    checks: list[CheckResult]
    violations: list[Violation]

    _diff_sha256_is_valid = field_validator("diff_sha256")(_validate_sha256)

    @model_validator(mode="after")
    def receipt_and_violations_match_acceptance(self) -> "PatchCheckResult":
        if self.accepted:
            if not self.receipt_id or self.receipt_artifact is None:
                raise ValueError("accepted patches require a receipt identity and artifact")
            if self.violations or any(check.status is CheckStatus.FAIL for check in self.checks):
                raise ValueError("accepted patches cannot contain failures or violations")
        elif not self.violations:
            raise ValueError("rejected patches require at least one violation")
        return self


class RunResult(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    fidelity: Fidelity
    patch_commit_sha: str = Field(min_length=1)
    outcome: ExecutionOutcome
    exit_code: int | None
    error_class: str | None
    error_fingerprint: str | None
    error_summary: str | None
    log_artifact: ArtifactRef | None
    telemetry_artifact: ArtifactRef | None
    checkpoint_artifact: ArtifactRef | None
    prediction_artifact: ArtifactRef | None
    resource_delta: ResourceDelta

    _fingerprint_is_valid = field_validator("error_fingerprint")(
        lambda value: _validate_sha256(value) if value is not None else value
    )

    @model_validator(mode="after")
    def errors_match_outcome(self) -> "RunResult":
        error_fields = (self.error_class, self.error_fingerprint, self.error_summary)
        if self.outcome is ExecutionOutcome.SUCCESS and any(v is not None for v in error_fields):
            raise ValueError("successful runs cannot contain error fields")
        if self.outcome is not ExecutionOutcome.SUCCESS and any(not v for v in error_fields):
            raise ValueError("non-success outcomes require all error fields")
        return self


class ScoreStats(StrictModel):
    rows: int = Field(ge=0)
    unique_scores: int = Field(ge=0)
    minimum: float
    maximum: float

    _finite_values = field_validator("minimum", "maximum")(_finite)

    @model_validator(mode="after")
    def stats_are_consistent(self) -> "ScoreStats":
        if self.unique_scores > self.rows:
            raise ValueError("unique_scores cannot exceed rows")
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class OutputCheckResult(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    prediction_artifact: ArtifactRef
    accepted: bool
    checks: dict[str, CheckStatus]
    score_stats: ScoreStats | None
    violations: list[Violation]

    @model_validator(mode="after")
    def checks_match_acceptance(self) -> "OutputCheckResult":
        has_failure = any(status is CheckStatus.FAIL for status in self.checks.values())
        if self.accepted and (has_failure or self.violations):
            raise ValueError("accepted output cannot contain failures or violations")
        if not self.accepted and not self.violations:
            raise ValueError("rejected output requires at least one violation")
        return self


class MetricSet(StrictModel):
    metrics: dict[str, float]
    primary_metric_name: str = Field(min_length=1)
    primary_score: float

    @field_validator("metrics")
    @classmethod
    def metrics_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not name or not math.isfinite(score) for name, score in value.items()):
            raise ValueError("metric names must be non-empty and values finite")
        return value

    _primary_score_is_finite = field_validator("primary_score")(_finite)

class PredictionChange(StrictModel):
    spearman_vs_parent: float | None
    changed_row_fraction: float | None = Field(ge=0, le=1)

    _finite_values = field_validator("spearman_vs_parent", "changed_row_fraction")(_finite)


class TrustAssessment(StrictModel):
    verdict: TrustVerdict
    stability: TrustStability
    integrity: TrustIntegrity
    flags: list[str]


class EvaluationResult(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    population: EvaluationPopulation
    fidelity: Fidelity
    seed: int
    public_query_index: int | None = Field(ge=0)
    evaluator_sha256: str
    contract_sha256: str
    metric_set: MetricSet
    baseline_delta: float
    parent_delta: float
    previous_best_delta: float
    prediction_change: PredictionChange
    trust: TrustAssessment

    _hashes_are_valid = field_validator("evaluator_sha256", "contract_sha256")(
        _validate_sha256
    )
    _deltas_are_finite = field_validator(
        "baseline_delta", "parent_delta", "previous_best_delta"
    )(_finite)


class LessonCandidate(StrictModel):
    origin: LessonOrigin
    category: LessonCategory
    tags: list[str]
    summary: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    avoid_when: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source_event_ids: list[str] = Field(min_length=1)
    source_commit_shas: list[str] = Field(min_length=1)

    _confidence_is_finite = field_validator("confidence")(_finite)

    @model_validator(mode="after")
    def category_matches_origin(self) -> "LessonCandidate":
        if self.origin is LessonOrigin.OPERATIONAL and self.category is LessonCategory.RESEARCH_RESULT:
            raise ValueError("operational lessons cannot claim a research result")
        if self.origin is LessonOrigin.RESEARCH and self.category in {
            LessonCategory.RESOURCE_CONSTRAINT,
            LessonCategory.IMPLEMENTATION_CONSTRAINT,
        }:
            raise ValueError("resource and implementation constraints are operational lessons")
        return self


class RecoveryPolicyContext(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    original_experiment_spec: dict[str, Any]
    current_patch_commit_sha: str = Field(min_length=1)
    failure_event_id: str = Field(min_length=1)
    attempt_history: list[dict[str, Any]]
    prior_error_fingerprints: list[str]
    repair_attempts_used: int = Field(ge=0)
    max_repair_attempts: int = Field(ge=0, le=2)
    same_commit_retries_used: int = Field(ge=0, le=1)
    # During migration the harness may pass a legacy attempt count; the durable
    # form is a named budget snapshot so additional dimensions are not conflated.
    remaining_run_budget: Annotated[int, Field(ge=0)] | dict[str, Any]
    allowed_runtime_adjustments: dict[str, Any]
    contract_summary: str = Field(min_length=1)

    @field_validator("prior_error_fingerprints")
    @classmethod
    def fingerprints_are_valid(cls, value: list[str]) -> list[str]:
        return [_validate_sha256(item) for item in value]

    @model_validator(mode="after")
    def repair_usage_is_bounded(self) -> "RecoveryPolicyContext":
        if self.repair_attempts_used > self.max_repair_attempts:
            raise ValueError("repair_attempts_used exceeds max_repair_attempts")
        return self


class RecoveryDecision(StrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    failure_event_id: str = Field(min_length=1)
    repair_attempt: int = Field(ge=0)
    action: RecoveryAction
    reason_code: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    same_error_count: int = Field(ge=1)
    remaining_repair_budget: int = Field(ge=0, le=2)
    lesson_candidate: LessonCandidate | None

    @model_validator(mode="after")
    def repair_attempt_is_bounded(self) -> "RecoveryDecision":
        if self.action is RecoveryAction.TRAE_REPAIR and not 1 <= self.repair_attempt <= 2:
            raise ValueError("Trae repair decisions are limited to attempts 1 and 2")
        return self


__all__ = [
    "ArtifactKind", "ArtifactRef", "CheckResult", "CheckStatus", "EvaluationPopulation",
    "EvaluationResult", "ExecutionOutcome", "Fidelity", "LessonCandidate", "LessonCategory",
    "LessonOrigin", "MetricSet", "MonitorAction", "MonitorDirective", "OutputCheckResult",
    "PatchCheckResult", "PredictionChange", "RecoveryAction", "RecoveryDecision",
    "RecoveryPolicyContext", "ResourceDelta", "RunResult", "ScoreStats", "TelemetrySample",
    "TokenMeasurement", "TrustAssessment", "TrustIntegrity", "TrustStability", "TrustVerdict",
    "Violation",
]
