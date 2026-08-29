"""Versioned schemas shared by every TacoRank component.

The harness is deliberately strict at component boundaries.  Adapters may use any
internal representation, but values crossing into the orchestrator must validate
against the models in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


SCHEMA_VERSION = "1.0"
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
EVENT_ID_RE = re.compile(r"^evt_[0-9]{6,}$")

COMPETITION_CONTRACT_HEADINGS: Tuple[str, ...] = (
    "# Competition Contract",
    "## Identity and source precedence",
    "## Required benchmark",
    "## Data and temporal boundary",
    "## Target label and permitted inputs",
    "## Metrics and primary aggregation",
    "## Official baseline",
    "## Convergence and resource limits",
    "## Editable and protected paths",
    "## Allowed commands",
    "## Evaluation isolation",
    "## Submission schema",
    "## Resolved ambiguities",
    "## Human approvals",
)

METHOD_CARD_HEADINGS: Tuple[str, ...] = (
    "## Mechanism",
    "## Preconditions",
    "## Allowed data",
    "## Expected effect",
    "## Falsification condition",
    "## Do not use when",
    "## Minimal implementation",
    "## Sources",
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_RUNTIME_ADJUSTMENT_KEYS = frozenset(
    {"batch_size", "num_workers", "mixed_precision", "timeout_profile"}
)


def _runtime_next_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("next_value", value.get("value"))
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _validate_runtime_mapping(values: Dict[str, Any]) -> Dict[str, Any]:
    if not set(values).issubset(_RUNTIME_ADJUSTMENT_KEYS):
        raise ValueError("runtime setting is not contract-approved")
    for name, raw in values.items():
        value = _runtime_next_value(raw)
        valid = (
            (name == "batch_size" and isinstance(value, int) and not isinstance(value, bool) and value > 0)
            or (name == "num_workers" and isinstance(value, int) and not isinstance(value, bool) and value >= 0)
            or (name == "mixed_precision" and isinstance(value, bool))
            or (name == "timeout_profile" and isinstance(value, str) and bool(value.strip()))
        )
        if not valid:
            raise ValueError("invalid value for runtime setting %s" % name)
    return values


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


def _validate_id(value: str, field_name: str) -> str:
    if not ID_RE.fullmatch(value):
        raise ValueError(
            "%s must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'" % field_name
        )
    return value


def normalize_relative_path(value: str) -> str:
    """Validate and return a canonical repository-relative POSIX path."""

    if not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("path must be normalized and repository-relative")
    normalized = str(path)
    if normalized != value or normalized.startswith("/"):
        raise ValueError("path must already be normalized")
    return normalized


class ArtifactKind(str, Enum):
    DIFF = "diff"
    TRAJECTORY = "trajectory"
    CONTEXT = "context"
    LOG = "log"
    CHECKPOINT = "checkpoint"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    DELTA_VECTOR = "delta_vector"
    VERIFICATION_RECEIPT = "verification_receipt"
    SUBMISSION = "submission"
    REPORT = "report"
    OTHER = "other"


class TokenMeasurement(str, Enum):
    PROVIDER = "provider"
    ESTIMATED = "estimated"
    NONE = "none"


class CostTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Fidelity(str, Enum):
    SMOKE = "smoke"
    PROXY = "proxy"
    FULL = "full"
    FINAL = "final"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class PlannerAction(str, Enum):
    PROPOSE = "propose"
    RECOMMEND_STOP = "recommend_stop"
    BLOCKED = "blocked"


class RunOutcome(str, Enum):
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


class Population(str, Enum):
    INTERNAL_PROXY = "internal_proxy"
    PUBLIC_VALIDATION = "public_validation"
    UNBIASED_AUDIT = "unbiased_audit"
    HIDDEN_FINAL = "hidden_final"


class TrustVerdict(str, Enum):
    ACCEPTED = "accepted"
    INCONCLUSIVE = "inconclusive"
    NEGATIVE = "negative"
    NO_OP = "no_op"
    SUSPICIOUS = "suspicious"
    REDUNDANT = "redundant"


class Stability(str, Enum):
    SINGLE_SEED = "single_seed"
    CONFIRMED = "confirmed"
    UNSTABLE = "unstable"
    NOT_APPLICABLE = "not_applicable"


class Integrity(str, Enum):
    CLEAN = "clean"
    COMPROMISED = "compromised"
    INCONCLUSIVE = "inconclusive"


class ExperimentDecisionKind(str, Enum):
    PROMOTE = "promote"
    ACCEPT = "accept"
    REJECT = "reject"
    PRUNE = "prune"
    INVALID = "invalid"


# Person 5's deterministic decision layer uses the shorter canonical name.
Decision = ExperimentDecisionKind


class ExperimentFamily(str, Enum):
    OBJECTIVE = "objective"
    SAMPLING = "sampling"
    TEMPORAL_HISTORY = "temporal_history"
    FEATURES = "features"
    MODEL = "model"
    MULTITASK = "multitask"
    DURATION_BIAS = "duration_bias"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    OTHER = "other"


class ExperimentKind(str, Enum):
    FRAME = "frame"
    CONTENT = "content"
    CAPACITY = "capacity"
    COMPOSITION = "composition"
    POLICY = "policy"
    OTHER = "other"


class MethodStatus(str, Enum):
    CANDIDATE = "candidate"
    BLOCKED = "blocked"
    KNOWN_NEGATIVE = "known_negative"
    FORBIDDEN = "forbidden"


FAMILY_KIND: Mapping[ExperimentFamily, ExperimentKind] = {
    ExperimentFamily.OBJECTIVE: ExperimentKind.FRAME,
    ExperimentFamily.SAMPLING: ExperimentKind.FRAME,
    ExperimentFamily.TEMPORAL_HISTORY: ExperimentKind.CONTENT,
    ExperimentFamily.FEATURES: ExperimentKind.CONTENT,
    ExperimentFamily.MULTITASK: ExperimentKind.CONTENT,
    ExperimentFamily.DURATION_BIAS: ExperimentKind.CONTENT,
    ExperimentFamily.MODEL: ExperimentKind.CAPACITY,
    ExperimentFamily.ENSEMBLE: ExperimentKind.COMPOSITION,
    ExperimentFamily.EVALUATION: ExperimentKind.POLICY,
    ExperimentFamily.OTHER: ExperimentKind.OTHER,
}


def family_kind(family: ExperimentFamily) -> ExperimentKind:
    return FAMILY_KIND[family]


class MethodCardMetadata(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    method_id: NonEmptyStr
    family: ExperimentFamily
    status: MethodStatus
    tags: List[NonEmptyStr]
    cost_tier: CostTier
    sources: List[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_method_card(self) -> "MethodCardMetadata":
        if not self.tags:
            raise ValueError("method card requires at least one tag")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("method card tags must be unique")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("method card sources must be unique")
        if any(
            not source.startswith(("https://", "http://"))
            for source in self.sources
        ):
            raise ValueError("method card sources must be HTTP(S) URLs")
        return self


class LessonOrigin(str, Enum):
    OPERATIONAL = "operational"
    RESEARCH = "research"


class LessonCategory(str, Enum):
    RESEARCH_RESULT = "research_result"
    RESOURCE_CONSTRAINT = "resource_constraint"
    IMPLEMENTATION_CONSTRAINT = "implementation_constraint"
    INTEGRITY_WARNING = "integrity_warning"
    PROCESS_RULE = "process_rule"


class LessonStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    STALE = "stale"
    REJECTED = "rejected"


class ArtifactRef(StrictModel):
    artifact_id: NonEmptyStr
    kind: ArtifactKind
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: Optional[str] = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return _validate_id(value, "artifact_id")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    def verify_file(
        self,
        repository_root: Union[str, Path],
        approved_roots: Sequence[str] = ("artifacts", "runs"),
    ) -> Path:
        """Verify location, size, and content hash under an approved root."""
        root = Path(repository_root).resolve()
        candidate = root.joinpath(*PurePosixPath(self.path).parts)
        if candidate.is_symlink():
            raise ValueError("artifact path must not be a symlink")
        resolved = candidate.resolve()
        _require_within(resolved, root, "artifact path escapes repository root")
        approved = [
            root.joinpath(*PurePosixPath(item).parts).resolve()
            for item in approved_roots
        ]
        if not any(_is_within(resolved, allowed) for allowed in approved):
            raise ValueError("artifact path is outside approved artifact roots")
        if not resolved.is_file():
            raise ValueError("artifact bytes are missing or not a regular file")
        if resolved.stat().st_size != self.size_bytes:
            raise ValueError("artifact size does not match ArtifactRef")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ValueError("artifact hash does not match ArtifactRef")
        return resolved


class ResourceDelta(StrictModel):
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    token_measurement: TokenMeasurement = TokenMeasurement.NONE
    wall_time_ms: int = Field(default=0, ge=0)
    cpu_time_ms: int = Field(default=0, ge=0)
    gpu_time_ms: int = Field(default=0, ge=0)
    gpu_count: int = Field(default=0, ge=0)
    peak_rss_mb: Optional[int] = Field(default=None, ge=0)
    peak_gpu_memory_mb: Optional[int] = Field(default=None, ge=0)
    manual_interventions: int = Field(default=0, ge=0)

    @property
    def gpu_hours(self) -> float:
        return (self.gpu_time_ms * self.gpu_count) / 3_600_000.0


class MetricSet(StrictModel):
    metrics: Dict[str, float]
    primary_metric_name: NonEmptyStr
    primary_score: float

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            raise ValueError("metrics must not be empty")
        for name, value in values.items():
            if not name.strip():
                raise ValueError("metric names must not be empty")
            if not math.isfinite(value):
                raise ValueError("metric values must be finite")
        return values

    @model_validator(mode="after")
    def validate_primary(self) -> "MetricSet":
        if not math.isfinite(self.primary_score):
            raise ValueError("primary_score must be finite")
        if (
            self.primary_metric_name in self.metrics
            and self.metrics[self.primary_metric_name] != self.primary_score
        ):
            raise ValueError("primary_score must equal the named primary metric")
        return self

    def validate_contract(
        self,
        required_metrics: Sequence[str],
        diagnostic_metrics: Sequence[str] = (),
        expected_primary: Optional[float] = None,
        tolerance: float = 1e-12,
    ) -> None:
        required = set(required_metrics)
        allowed = required | set(diagnostic_metrics)
        names = set(self.metrics)
        if not required.issubset(names):
            raise ValueError("MetricSet is missing required contract metrics")
        if not names.issubset(allowed):
            raise ValueError("MetricSet contains undeclared diagnostic metrics")
        if expected_primary is not None and not math.isclose(
            self.primary_score,
            float(expected_primary),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        ):
            raise ValueError("primary_score does not match contract aggregation")


class CostEstimate(StrictModel):
    llm_tokens_upper_bound: int = Field(ge=0)
    wall_time_seconds_upper_bound: int = Field(ge=0)
    gpu_seconds_upper_bound: int = Field(ge=0)
    cost_tier: CostTier


class CheckResult(StrictModel):
    name: NonEmptyStr
    status: CheckStatus
    summary: Optional[str] = None


class Violation(StrictModel):
    code: NonEmptyStr
    message: NonEmptyStr
    path: Optional[str] = None

    @field_validator("path")
    @classmethod
    def validate_optional_path(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else normalize_relative_path(value)


class LessonCandidate(StrictModel):
    origin: LessonOrigin
    category: LessonCategory
    tags: List[NonEmptyStr] = Field(default_factory=list)
    summary: NonEmptyStr
    applicability: NonEmptyStr
    avoid_when: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    source_commit_shas: List[NonEmptyStr] = Field(default_factory=list)


class ExperimentSpec(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    parent_experiment_id: Optional[str] = None
    parent_commit_sha: NonEmptyStr
    context_id: NonEmptyStr
    hypothesis: NonEmptyStr
    family: NonEmptyStr
    change_summary: NonEmptyStr
    target_stage: NonEmptyStr
    target_files: List[str]
    fidelity_plan: List[Fidelity]
    expected_mechanism: NonEmptyStr
    success_criteria: NonEmptyStr
    falsification_condition: NonEmptyStr
    estimated_cost: CostEstimate
    method_card_ids: List[NonEmptyStr] = Field(default_factory=list)
    evidence_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    duplicate_key: NonEmptyStr

    @field_validator("run_id", "experiment_id", "context_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_id(value, info.field_name)

    @field_validator("parent_experiment_id")
    @classmethod
    def validate_optional_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_id(value, "parent_experiment_id")

    @field_validator("target_files")
    @classmethod
    def validate_target_files(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("target_files must not be empty")
        normalized = [normalize_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("target_files must not contain duplicates")
        return normalized

    @field_validator("fidelity_plan")
    @classmethod
    def validate_fidelity_plan(cls, values: List[Fidelity]) -> List[Fidelity]:
        if not values:
            raise ValueError("fidelity_plan must not be empty")
        order = {Fidelity.SMOKE: 0, Fidelity.PROXY: 1, Fidelity.FULL: 2}
        for previous, current in zip(values, values[1:]):
            if order[current] <= order[previous]:
                raise ValueError("fidelity_plan must be strictly increasing without duplicates")
        return values


class PlannerOutput(StrictModel):
    action: PlannerAction
    spec: Optional[ExperimentSpec] = None
    reason_code: NonEmptyStr
    reason: NonEmptyStr
    supporting_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @model_validator(mode="after")
    def validate_action_spec(self) -> "PlannerOutput":
        if (self.action == PlannerAction.PROPOSE) != (self.spec is not None):
            raise ValueError("action=propose if and only if spec is present")
        return self


class PatchCandidate(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    experiment_spec_event_id: NonEmptyStr
    context_id: NonEmptyStr
    base_commit_sha: NonEmptyStr
    patch_commit_sha: NonEmptyStr
    diff_sha256: str
    changed_files: List[str]
    diff_artifact: ArtifactRef
    trajectory_artifact: ArtifactRef
    trae_version: NonEmptyStr
    model_id: NonEmptyStr
    steps_used: int = Field(ge=0)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @field_validator("diff_sha256")
    @classmethod
    def validate_diff_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("diff_sha256 must be lowercase sha256")
        return value

    @field_validator("changed_files")
    @classmethod
    def validate_changed_files(cls, values: List[str]) -> List[str]:
        normalized = [normalize_relative_path(value) for value in values]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("changed_files must be non-empty and unique")
        return normalized


class PatchCheckResult(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    patch_commit_sha: NonEmptyStr
    diff_sha256: str
    accepted: bool
    receipt_id: Optional[str] = None
    receipt_artifact: Optional[ArtifactRef] = None
    checks: List[CheckResult]
    violations: List[Violation] = Field(default_factory=list)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @model_validator(mode="after")
    def validate_receipt(self) -> "PatchCheckResult":
        if self.accepted and (not self.receipt_id or self.receipt_artifact is None):
            raise ValueError("accepted patch checks require a receipt and artifact")
        if not self.accepted and (self.receipt_id or self.receipt_artifact):
            raise ValueError("rejected patch checks cannot issue a receipt")
        if self.accepted and any(check.status == CheckStatus.FAIL for check in self.checks):
            raise ValueError("accepted patch checks cannot contain failed checks")
        return self


class RunRequest(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    fidelity: Fidelity
    command_id: NonEmptyStr
    patch_commit_sha: NonEmptyStr
    patch_receipt_id: NonEmptyStr
    seed: int
    data_manifest_sha256: str
    timeout_seconds: int = Field(gt=0)
    memory_limit_mb: int = Field(gt=0)
    gpu_memory_limit_mb: int = Field(ge=0)
    network_enabled: bool = False
    runtime_settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_manifest_sha256")
    @classmethod
    def validate_manifest_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("data_manifest_sha256 must be lowercase sha256")
        return value

    @field_validator("runtime_settings")
    @classmethod
    def validate_runtime_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_runtime_mapping(values)


class TelemetrySample(StrictModel):
    timestamp: datetime
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    elapsed_ms: int = Field(ge=0)
    process_alive: bool
    last_output_age_ms: int = Field(ge=0)
    cpu_percent: float = Field(ge=0)
    rss_mb: int = Field(ge=0)
    gpu_utilization_percent: Optional[float] = Field(default=None, ge=0, le=100)
    gpu_memory_mb: Optional[int] = Field(default=None, ge=0)
    loss: Optional[float] = None
    gradient_norm: Optional[float] = None
    disk_free_mb: int = Field(ge=0)
    recent_output_tail: str = ""


class MonitorDirective(StrictModel):
    action: MonitorAction
    reason_code: Optional[str] = None
    summary: Optional[str] = None

    @model_validator(mode="after")
    def validate_termination_reason(self) -> "MonitorDirective":
        if self.action == MonitorAction.TERMINATE and not self.reason_code:
            raise ValueError("terminate directives require a reason_code")
        return self


class RunResult(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    fidelity: Fidelity
    patch_commit_sha: NonEmptyStr
    outcome: RunOutcome
    exit_code: Optional[int] = None
    error_class: Optional[str] = None
    error_fingerprint: Optional[str] = None
    error_summary: Optional[str] = None
    log_artifact: ArtifactRef
    telemetry_artifact: ArtifactRef
    checkpoint_artifact: Optional[ArtifactRef] = None
    prediction_artifact: Optional[ArtifactRef] = None
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @model_validator(mode="after")
    def validate_outcome_artifacts(self) -> "RunResult":
        if self.outcome == RunOutcome.SUCCESS and self.prediction_artifact is None:
            raise ValueError("successful runs require a prediction artifact")
        if self.outcome != RunOutcome.SUCCESS and not self.error_class:
            raise ValueError("failed runs require an error_class")
        return self


class OutputCheckResult(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    prediction_artifact: ArtifactRef
    ordered_row_identity_sha256: str
    ordered_prediction_sha256: str
    accepted: bool
    checks: Dict[str, CheckStatus]
    score_stats: Dict[str, float] = Field(default_factory=dict)
    violations: List[Violation] = Field(default_factory=list)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @field_validator(
        "ordered_row_identity_sha256", "ordered_prediction_sha256"
    )
    @classmethod
    def validate_ordered_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("ordered identities must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def validate_acceptance(self) -> "OutputCheckResult":
        failed = any(status == CheckStatus.FAIL for status in self.checks.values())
        if self.accepted and (failed or self.violations):
            raise ValueError("accepted output checks must be clean")
        return self


class RecoveryDecision(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    failure_event_id: NonEmptyStr
    repair_attempt: int = Field(ge=1)
    action: RecoveryAction
    reason_code: NonEmptyStr
    instructions: NonEmptyStr
    same_error_count: int = Field(ge=0)
    remaining_repair_budget: int = Field(ge=0)
    runtime_adjustments: Dict[str, Any] = Field(default_factory=dict)
    lesson_candidate: Optional[LessonCandidate] = None
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @field_validator("runtime_adjustments")
    @classmethod
    def validate_runtime_adjustments(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_runtime_mapping(values)

    @model_validator(mode="after")
    def validate_action_adjustment(self) -> "RecoveryDecision":
        has_adjustment = bool(self.runtime_adjustments)
        if (self.action == RecoveryAction.ADJUST_APPROVED_RUNTIME_SETTING) != has_adjustment:
            raise ValueError("runtime adjustments must match the recovery action")
        if has_adjustment and len(self.runtime_adjustments) != 1:
            raise ValueError("exactly one runtime adjustment is allowed")
        return self


class TrustAssessment(StrictModel):
    verdict: TrustVerdict
    stability: Stability
    integrity: Integrity
    flags: List[NonEmptyStr] = Field(default_factory=list)
    eta_applied: Optional[float] = Field(default=None, ge=0.0)
    seed_mean: Optional[float] = None
    seed_stderr: Optional[float] = Field(default=None, ge=0.0)
    seed_count: int = Field(default=1, ge=1)

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("trust flags must be unique")
        return values

    @model_validator(mode="after")
    def validate_consistency(self) -> "TrustAssessment":
        if (
            self.integrity == Integrity.COMPROMISED
            and self.verdict != TrustVerdict.SUSPICIOUS
        ):
            raise ValueError("compromised integrity requires suspicious verdict")
        if (
            self.verdict == TrustVerdict.NO_OP
            and self.stability != Stability.NOT_APPLICABLE
        ):
            raise ValueError("no_op stability must be not_applicable")
        supplied = (self.seed_mean is not None, self.seed_stderr is not None)
        if supplied[0] != supplied[1]:
            raise ValueError("seed mean and standard error must be supplied together")
        if any(supplied) and self.seed_count < 3:
            raise ValueError("seed aggregates require at least three seed results")
        return self


class PredictionChange(StrictModel):
    spearman_vs_parent: float = Field(ge=-1.0, le=1.0)
    changed_row_fraction: float = Field(ge=0.0, le=1.0)


class EvaluationRequest(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    output_checked_event_id: NonEmptyStr
    prediction_artifact: ArtifactRef
    population: Population
    fidelity: Fidelity
    seed: int
    contract_sha256: str
    evaluator_sha256: str
    baseline_summary: Dict[str, float] = Field(default_factory=dict)
    parent_summary: Dict[str, float] = Field(default_factory=dict)
    previous_best_summary: Dict[str, float] = Field(default_factory=dict)
    public_query_index: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_public_query(self) -> "EvaluationRequest":
        if self.population == Population.PUBLIC_VALIDATION and self.public_query_index is None:
            raise ValueError("public validation requires public_query_index")
        if self.population != Population.PUBLIC_VALIDATION and self.public_query_index is not None:
            raise ValueError("public_query_index is only valid for public validation")
        return self


class EvaluationResult(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    attempt: int = Field(ge=1)
    population: Population
    fidelity: Fidelity
    seed: int
    public_query_index: Optional[int] = Field(default=None, ge=1)
    evaluator_sha256: str
    contract_sha256: str
    metric_set: MetricSet
    baseline_delta: float
    parent_delta: Optional[float] = None
    previous_best_delta: Optional[float] = None
    prediction_change: Union[float, PredictionChange]
    trust: TrustAssessment
    seed_evidence_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    metrics_artifact: Optional[ArtifactRef] = None
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @field_validator("evaluator_sha256", "contract_sha256")
    @classmethod
    def validate_evaluation_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("evaluation hashes must be lowercase sha256")
        return value

    @field_validator("prediction_change")
    @classmethod
    def validate_prediction_change(
        cls, value: Union[float, PredictionChange]
    ) -> Union[float, PredictionChange]:
        if isinstance(value, float) and value < 0:
            raise ValueError("prediction_change must be non-negative")
        return value

    @field_validator("seed_evidence_event_ids")
    @classmethod
    def validate_seed_events(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("seed_evidence_event_ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_evaluation_route(self) -> "EvaluationResult":
        if self.population == Population.PUBLIC_VALIDATION:
            if self.fidelity != Fidelity.FULL or self.public_query_index is None:
                raise ValueError("public validation requires full fidelity and query index")
        elif self.public_query_index is not None:
            raise ValueError("public_query_index is only valid for public validation")
        if (
            self.population == Population.INTERNAL_PROXY
            and self.fidelity != Fidelity.PROXY
        ):
            raise ValueError("internal proxy requires proxy fidelity")
        if (
            self.population == Population.UNBIASED_AUDIT
            and self.fidelity != Fidelity.FULL
        ):
            raise ValueError("unbiased audit requires full fidelity")
        if (
            self.population == Population.HIDDEN_FINAL
            and self.fidelity != Fidelity.FINAL
        ):
            raise ValueError("hidden final requires final fidelity")
        if self.experiment_id != "baseline":
            if self.trust.seed_count != len(self.seed_evidence_event_ids) + 1:
                raise ValueError(
                    "trust seed_count must include the current evaluation and all "
                    "seed evidence events"
                )
            if self.trust.stability in (Stability.CONFIRMED, Stability.UNSTABLE):
                if (
                    self.trust.seed_count < 3
                    or self.trust.seed_mean is None
                    or self.trust.seed_stderr is None
                ):
                    raise ValueError(
                        "confirmed or unstable trust requires verified seed aggregates"
                    )
        return self


class ExperimentDecision(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    evaluation_event_id: Optional[str] = None
    decision: ExperimentDecisionKind
    reason_code: NonEmptyStr
    fidelity_completed: Fidelity
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Optional[Fidelity] = None
    supporting_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    lesson_candidate: Optional[LessonCandidate] = None
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)

    @model_validator(mode="after")
    def validate_promotion(self) -> "ExperimentDecision":
        if (self.decision == ExperimentDecisionKind.PROMOTE) != (self.next_fidelity is not None):
            raise ValueError("promote decisions require next_fidelity and only promotes may set it")
        if self.best_eligible and self.fidelity_completed != Fidelity.FULL:
            raise ValueError("only full-fidelity decisions can be best eligible")
        return self


class PlannerContractSummary(StrictModel):
    """Machine-readable subset of the frozen contract used by Person 1."""

    resolved: bool = False
    allowed_families: List[NonEmptyStr] = Field(default_factory=list)
    protected_paths: List[str] = Field(default_factory=list)
    editable_paths: List[str] = Field(default_factory=list)
    epsilon: float = Field(default=0.0, ge=0.0)


class PlannerBudgetSummary(StrictModel):
    remaining_experiments: int = Field(default=0, ge=0)
    remaining_public_queries: Optional[int] = Field(default=None, ge=0)
    remaining_llm_tokens: Optional[int] = Field(default=None, ge=0)
    remaining_wall_time_seconds: int = Field(default=0, ge=0)
    remaining_gpu_seconds: Optional[int] = Field(default=None, ge=0)


class PlannerConvergenceSummary(StrictModel):
    patience: int = Field(gt=0)
    consecutive_non_improving_full_evaluations: int = Field(default=0, ge=0)
    full_evaluations_completed: int = Field(default=0, ge=0)


class PlannerMethodCardSummary(StrictModel):
    method_id: NonEmptyStr
    family: NonEmptyStr
    status: NonEmptyStr
    cost_tier: CostTier


class PlannerExperimentSummary(StrictModel):
    """Verified experiment projection; code remains authoritative in Git."""

    experiment_id: NonEmptyStr
    parent_experiment_id: Optional[str] = None
    commit_sha: NonEmptyStr
    family: Optional[str] = None
    hypothesis_summary: str = ""
    trust_verdict: Optional[TrustVerdict] = None
    stability: Optional[Stability] = None
    integrity: Optional[Integrity] = None
    decision: Optional[ExperimentDecisionKind] = None
    highest_completed_fidelity: Optional[Fidelity] = None
    primary_score: Optional[float] = None
    metric_set: Optional[MetricSet] = None
    metric_deltas: Dict[str, float] = Field(default_factory=dict)
    baseline_delta: Optional[float] = None
    parent_delta: Optional[float] = None
    previous_best_delta: Optional[float] = None
    prediction_change: Optional[float] = Field(default=None, ge=0.0)
    child_count: int = Field(default=0, ge=0)
    actual_cost: Optional[CostTier] = None
    parent_eligible: bool = False
    best_eligible: bool = False
    status: NonEmptyStr
    duplicate_key: str = ""
    method_card_ids: List[NonEmptyStr] = Field(default_factory=list)
    supporting_event_ids: List[NonEmptyStr] = Field(default_factory=list)


class ContextDocument(StrictModel):
    context_id: NonEmptyStr
    role: Literal["planner", "coder", "recovery"]
    run_id: NonEmptyStr
    experiment_id: Optional[str] = None
    snapshot_event_id: Optional[str] = None
    source_event_ids: List[NonEmptyStr] = Field(default_factory=list)
    excluded_source_ids: Dict[str, str] = Field(default_factory=dict)
    content: str
    estimated_tokens: int = Field(ge=0)
    artifact: ArtifactRef


class PlannerContext(ContextDocument):
    role: Literal["planner"] = "planner"
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    contract_sha256: str
    contract_summary: PlannerContractSummary
    baseline: PlannerExperimentSummary
    current_best: PlannerExperimentSummary
    # This collection is authoritative. An empty list means there is no legal
    # parent; consumers must not reconstruct eligibility from history.
    eligible_frontier: List[PlannerExperimentSummary] = Field(default_factory=list)
    family_history: List[PlannerExperimentSummary] = Field(default_factory=list)
    method_cards: List[PlannerMethodCardSummary] = Field(default_factory=list)
    remaining_budget: PlannerBudgetSummary
    convergence: PlannerConvergenceSummary


class CoderContext(ContextDocument):
    role: Literal["coder"] = "coder"


class RecoveryContext(ContextDocument):
    role: Literal["recovery"] = "recovery"


ContextValue = Annotated[
    Union[PlannerContext, CoderContext, RecoveryContext],
    Field(discriminator="role"),
]


class RecoveryPolicyContext(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    original_experiment_spec: ExperimentSpec
    current_patch_commit_sha: NonEmptyStr
    failure_event_id: NonEmptyStr
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list)
    repair_attempts_used: int = Field(ge=0)
    max_repair_attempts: int = Field(ge=0, le=2)
    same_commit_retries_used: int = Field(ge=0, le=1)
    remaining_repair_budget: int = Field(ge=0)
    previous_error_fingerprints: List[NonEmptyStr] = Field(default_factory=list)
    remaining_run_budget: Dict[str, int] = Field(default_factory=dict)
    allowed_runtime_adjustments: Dict[str, Any] = Field(default_factory=dict)
    current_runtime_settings: Dict[str, Any] = Field(default_factory=dict)
    contract_summary: NonEmptyStr

    @model_validator(mode="after")
    def validate_repair_budget(self) -> "RecoveryPolicyContext":
        if self.remaining_repair_budget != (
            self.max_repair_attempts - self.repair_attempts_used
        ):
            raise ValueError("remaining repair budget is inconsistent")
        _validate_runtime_mapping(self.allowed_runtime_adjustments)
        _validate_runtime_mapping(self.current_runtime_settings)
        return self


class EvaluationDecisionContext(StrictModel):
    run_id: NonEmptyStr
    experiment_id: NonEmptyStr
    baseline_score: float
    parent_score: Optional[float] = None
    previous_best_score: Optional[float] = None


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    CONTRACT_VERIFIED = "contract.verified"
    BASELINE_VERIFIED = "baseline.verified"
    CONTEXT_CREATED = "context.created"
    PLANNER_RECOMMENDED = "planner.recommended"
    EXPERIMENT_PROPOSED = "experiment.proposed"
    PATCH_CREATED = "patch.created"
    PATCH_CHECKED = "patch.checked"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_FINISHED = "execution.finished"
    RECOVERY_DECIDED = "recovery.decided"
    OUTPUT_CHECKED = "output.checked"
    EVALUATION_COMPLETED = "evaluation.completed"
    EXPERIMENT_DECIDED = "experiment.decided"
    BEST_UPDATED = "best.updated"
    LESSON_RECORDED = "lesson.recorded"
    LESSON_STATUS_CHANGED = "lesson.status_changed"
    MANUAL_INTERVENTION = "manual.intervention"
    RUN_STOPPED = "run.stopped"
    FINAL_SELECTED = "final.selected"
    SUBMISSION_CHECKED = "submission.checked"


class RunStartedPayload(StrictModel):
    type: Literal["run.started"] = "run.started"
    config_sha256: str
    contract_sha256: str
    protected_paths_sha256: str
    max_experiments: int = Field(gt=0)
    wall_time_limit_seconds: int = Field(gt=0)
    token_limit: Optional[int] = Field(default=None, gt=0)
    gpu_seconds_limit: Optional[int] = Field(default=None, gt=0)
    max_repairs_per_experiment: int = Field(default=2, ge=0)
    max_confirmation_attempts: int = Field(default=2, ge=0)
    seed_schedule: List[int]

    @field_validator("config_sha256", "contract_sha256", "protected_paths_sha256")
    @classmethod
    def validate_frozen_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("frozen hashes must be lowercase sha256")
        return value


class ContractVerifiedPayload(StrictModel):
    type: Literal["contract.verified"] = "contract.verified"
    contract_sha256: str
    protected_paths_sha256: str
    metric_names: List[NonEmptyStr]
    primary_metric_name: NonEmptyStr
    command_ids: List[NonEmptyStr]
    artifact_roots: List[str]
    evaluator_sha256: str

    @field_validator("contract_sha256", "protected_paths_sha256", "evaluator_sha256")
    @classmethod
    def validate_contract_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("contract hashes must be lowercase sha256")
        return value

    @field_validator("artifact_roots")
    @classmethod
    def validate_roots(cls, values: List[str]) -> List[str]:
        return [normalize_relative_path(value) for value in values]


class BaselineVerifiedPayload(StrictModel):
    type: Literal["baseline.verified"] = "baseline.verified"
    experiment_id: NonEmptyStr = "baseline"
    commit_sha: NonEmptyStr
    metric_set: MetricSet
    evaluation: EvaluationResult


class ContextCreatedPayload(StrictModel):
    type: Literal["context.created"] = "context.created"
    context: ContextValue


class PlannerRecommendedPayload(StrictModel):
    type: Literal["planner.recommended"] = "planner.recommended"
    output: PlannerOutput

    @model_validator(mode="after")
    def reject_proposal(self) -> "PlannerRecommendedPayload":
        if self.output.action == PlannerAction.PROPOSE:
            raise ValueError("planner.recommended cannot contain a proposal")
        return self


class ExperimentProposedPayload(StrictModel):
    type: Literal["experiment.proposed"] = "experiment.proposed"
    spec: ExperimentSpec


class PatchCreatedPayload(StrictModel):
    type: Literal["patch.created"] = "patch.created"
    candidate: PatchCandidate


class PatchCheckedPayload(StrictModel):
    type: Literal["patch.checked"] = "patch.checked"
    result: PatchCheckResult


class ExecutionStartedPayload(StrictModel):
    type: Literal["execution.started"] = "execution.started"
    request: RunRequest


class ExecutionFinishedPayload(StrictModel):
    type: Literal["execution.finished"] = "execution.finished"
    result: RunResult


class RecoveryDecidedPayload(StrictModel):
    type: Literal["recovery.decided"] = "recovery.decided"
    decision: RecoveryDecision


class OutputCheckedPayload(StrictModel):
    type: Literal["output.checked"] = "output.checked"
    result: OutputCheckResult


class EvaluationCompletedPayload(StrictModel):
    type: Literal["evaluation.completed"] = "evaluation.completed"
    result: EvaluationResult


class ExperimentDecidedPayload(StrictModel):
    type: Literal["experiment.decided"] = "experiment.decided"
    decision: ExperimentDecision


class BestUpdatedPayload(StrictModel):
    type: Literal["best.updated"] = "best.updated"
    experiment_id: NonEmptyStr
    commit_sha: NonEmptyStr
    primary_metric_name: NonEmptyStr
    primary_score: float
    decision_event_id: NonEmptyStr


class LessonRecordedPayload(StrictModel):
    type: Literal["lesson.recorded"] = "lesson.recorded"
    lesson_id: NonEmptyStr
    candidate: LessonCandidate


class LessonStatusChangedPayload(StrictModel):
    type: Literal["lesson.status_changed"] = "lesson.status_changed"
    lesson_id: NonEmptyStr
    status: LessonStatus
    reason: NonEmptyStr


class ManualInterventionPayload(StrictModel):
    type: Literal["manual.intervention"] = "manual.intervention"
    reason: NonEmptyStr
    actor: NonEmptyStr


class RunStoppedPayload(StrictModel):
    type: Literal["run.stopped"] = "run.stopped"
    reason_code: NonEmptyStr
    reason: NonEmptyStr


class FinalSelectedPayload(StrictModel):
    type: Literal["final.selected"] = "final.selected"
    experiment_id: NonEmptyStr
    commit_sha: NonEmptyStr
    reproduction_evaluation_event_id: NonEmptyStr


class SubmissionCheckedPayload(StrictModel):
    type: Literal["submission.checked"] = "submission.checked"
    accepted: bool
    submission_artifact: ArtifactRef
    checks: List[CheckResult]

    @model_validator(mode="after")
    def validate_acceptance(self) -> "SubmissionCheckedPayload":
        if self.accepted and any(check.status == CheckStatus.FAIL for check in self.checks):
            raise ValueError("accepted submissions cannot contain failed checks")
        return self


EventPayload = Annotated[
    Union[
        RunStartedPayload,
        ContractVerifiedPayload,
        BaselineVerifiedPayload,
        ContextCreatedPayload,
        PlannerRecommendedPayload,
        ExperimentProposedPayload,
        PatchCreatedPayload,
        PatchCheckedPayload,
        ExecutionStartedPayload,
        ExecutionFinishedPayload,
        RecoveryDecidedPayload,
        OutputCheckedPayload,
        EvaluationCompletedPayload,
        ExperimentDecidedPayload,
        BestUpdatedPayload,
        LessonRecordedPayload,
        LessonStatusChangedPayload,
        ManualInterventionPayload,
        RunStoppedPayload,
        FinalSelectedPayload,
        SubmissionCheckedPayload,
    ],
    Field(discriminator="type"),
]

EVENT_PAYLOAD_MODELS: Mapping[EventType, Type[StrictModel]] = {
    EventType.RUN_STARTED: RunStartedPayload,
    EventType.CONTRACT_VERIFIED: ContractVerifiedPayload,
    EventType.BASELINE_VERIFIED: BaselineVerifiedPayload,
    EventType.CONTEXT_CREATED: ContextCreatedPayload,
    EventType.PLANNER_RECOMMENDED: PlannerRecommendedPayload,
    EventType.EXPERIMENT_PROPOSED: ExperimentProposedPayload,
    EventType.PATCH_CREATED: PatchCreatedPayload,
    EventType.PATCH_CHECKED: PatchCheckedPayload,
    EventType.EXECUTION_STARTED: ExecutionStartedPayload,
    EventType.EXECUTION_FINISHED: ExecutionFinishedPayload,
    EventType.RECOVERY_DECIDED: RecoveryDecidedPayload,
    EventType.OUTPUT_CHECKED: OutputCheckedPayload,
    EventType.EVALUATION_COMPLETED: EvaluationCompletedPayload,
    EventType.EXPERIMENT_DECIDED: ExperimentDecidedPayload,
    EventType.BEST_UPDATED: BestUpdatedPayload,
    EventType.LESSON_RECORDED: LessonRecordedPayload,
    EventType.LESSON_STATUS_CHANGED: LessonStatusChangedPayload,
    EventType.MANUAL_INTERVENTION: ManualInterventionPayload,
    EventType.RUN_STOPPED: RunStoppedPayload,
    EventType.FINAL_SELECTED: FinalSelectedPayload,
    EventType.SUBMISSION_CHECKED: SubmissionCheckedPayload,
}


class Event(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: str
    seq: int = Field(ge=1)
    timestamp: datetime
    run_id: NonEmptyStr
    event_type: EventType
    idempotency_key: NonEmptyStr
    causation_event_id: Optional[str] = None
    payload: EventPayload
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)
    prev_event_hash: str
    event_hash: str

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        if not EVENT_ID_RE.fullmatch(value):
            raise ValueError("event_id must use evt_<zero-padded sequence>")
        return value

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("causation_event_id")
    @classmethod
    def validate_causation(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not EVENT_ID_RE.fullmatch(value):
            raise ValueError("causation_event_id must be an event ID")
        return value

    @field_validator("prev_event_hash", "event_hash")
    @classmethod
    def validate_event_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("event hashes must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def validate_envelope(self) -> "Event":
        if self.event_id != "evt_%06d" % self.seq:
            raise ValueError("event_id must match seq")
        if self.event_type.value != self.payload.type:
            raise ValueError("event_type must match payload discriminator")
        key_parts = self.idempotency_key.split(":")
        if (
            len(key_parts) != 5
            or key_parts[0] != self.run_id
            or not ID_RE.fullmatch(key_parts[1])
            or not ID_RE.fullmatch(key_parts[2])
            or not key_parts[3].isdigit()
            or str(int(key_parts[3])) != key_parts[3]
            or not SHA256_RE.fullmatch(key_parts[4])
        ):
            raise ValueError(
                "idempotency_key must be run:experiment:stage:attempt:input_sha256"
            )
        if self.causation_event_id and int(self.causation_event_id.split("_")[1]) >= self.seq:
            raise ValueError("causation_event_id must refer to an earlier event")
        expected_artifacts = payload_artifacts(self.payload)
        if self.artifact_refs != expected_artifacts:
            raise ValueError(
                "artifact_refs must exactly match the canonical artifacts nested in payload"
            )
        nested_deltas = payload_resource_deltas(self.payload)
        if any(delta != self.resource_delta for delta in nested_deltas):
            raise ValueError(
                "resource_delta must match every resource delta nested in payload"
            )
        return self

    def canonical_bytes(self) -> bytes:
        data = self.model_dump(mode="json", exclude_none=False)
        data.pop("event_hash", None)
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def computed_event_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def assert_hash_valid(self) -> None:
        if self.event_hash != self.computed_event_hash():
            raise ValueError("event_hash does not match canonical event bytes")

    @classmethod
    def create(cls, **values: Any) -> "Event":
        """Validate an event draft and fill its canonical event hash."""
        draft_values = dict(values)
        draft_values["event_hash"] = ZERO_SHA256
        draft = cls.model_validate(draft_values)
        return draft.model_copy(update={"event_hash": draft.computed_event_hash()})


def payload_artifacts(payload: EventPayload) -> List[ArtifactRef]:
    """Collect ArtifactRef values nested anywhere in a payload."""

    found: List[ArtifactRef] = []

    def visit(value: Any) -> None:
        if isinstance(value, ArtifactRef):
            found.append(value)
        elif isinstance(value, BaseModel):
            for item in value.__dict__.values():
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    unique: Dict[str, ArtifactRef] = {}
    for item in found:
        previous = unique.get(item.artifact_id)
        if previous is not None and previous != item:
            raise ValueError("conflicting payload artifacts share artifact_id %r" % item.artifact_id)
        unique[item.artifact_id] = item
    return [unique[key] for key in sorted(unique)]


def payload_resource_deltas(payload: EventPayload) -> List[ResourceDelta]:
    """Collect action-local resource measurements nested in a payload."""

    found: List[ResourceDelta] = []

    def visit(value: Any) -> None:
        if isinstance(value, ResourceDelta):
            found.append(value)
        elif isinstance(value, BaseModel):
            for item in value.__dict__.values():
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return found


def parse_event_json(line: Union[str, bytes], verify_hash: bool = True) -> Event:
    """Parse one JSON event and optionally verify its content hash."""
    event = Event.model_validate_json(line)
    if verify_hash:
        event.assert_hash_valid()
    return event


def validate_competition_contract_markdown(text: str) -> None:
    """Validate the canonical contract's required heading order."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("competition contract must not be empty")
    headings = [line.strip() for line in text.splitlines() if line.startswith("#")]
    if not headings or headings[0] != COMPETITION_CONTRACT_HEADINGS[0]:
        raise ValueError("competition contract has the wrong title")
    position = -1
    for required in COMPETITION_CONTRACT_HEADINGS:
        matches = [
            index for index, heading in enumerate(headings) if heading == required
        ]
        if len(matches) != 1 or matches[0] <= position:
            raise ValueError(
                "competition contract headings are missing, duplicated, or out of order"
            )
        position = matches[0]


def parse_method_card_markdown(text: str) -> MethodCardMetadata:
    """Parse and validate a method card's first JSON block and headings."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("method card must not be empty")
    match = re.search(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if match is None or "```" in text[: match.start()]:
        raise ValueError("method card's first fenced block must be JSON")
    metadata = MethodCardMetadata.model_validate_json(match.group(1))
    headings = [
        line.strip()
        for line in text[match.end() :].splitlines()
        if line.startswith("## ")
    ]
    position = -1
    for required in METHOD_CARD_HEADINGS:
        matches = [
            index for index, heading in enumerate(headings) if heading == required
        ]
        if len(matches) != 1 or matches[0] <= position:
            raise ValueError(
                "method card headings are missing, duplicated, or out of order"
            )
        position = matches[0]
    return metadata


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_within(path: Path, root: Path, message: str) -> None:
    if not _is_within(path, root):
        raise ValueError(message)
