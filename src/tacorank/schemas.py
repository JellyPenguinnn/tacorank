"""Canonical Pydantic models for TacoRank's event ledger and component ports.

The Markdown memory schema is the human-readable authority.  This module is the
strict executable representation imported by every subsystem.
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
    ClassVar,
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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
ZERO_SHA256 = "0" * 64

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

RUN_ID_PATTERN = r"^run_[0-9]{8}_[a-z0-9][a-z0-9_-]*$"
EVENT_ID_PATTERN = r"^evt_[0-9]{6}$"
EXPERIMENT_ID_PATTERN = r"^exp_[0-9]{4}$"
LESSON_ID_PATTERN = r"^lesson_[0-9]{4}$"
CONTEXT_ID_PATTERN = r"^ctx_(planner|coder|recovery|evaluator)_[0-9]{6}$"
ARTIFACT_ID_PATTERN = r"^art_[0-9a-f]{8}$"
METHOD_ID_PATTERN = r"^method_[a-z0-9][a-z0-9_-]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_SHA_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"

SchemaVersion = Literal["1.0"]
RunId = Annotated[str, Field(pattern=RUN_ID_PATTERN)]
EventId = Annotated[str, Field(pattern=EVENT_ID_PATTERN)]
ExperimentId = Annotated[str, Field(pattern=EXPERIMENT_ID_PATTERN)]
LessonId = Annotated[str, Field(pattern=LESSON_ID_PATTERN)]
ContextId = Annotated[str, Field(pattern=CONTEXT_ID_PATTERN)]
ArtifactId = Annotated[str, Field(pattern=ARTIFACT_ID_PATTERN)]
MethodId = Annotated[str, Field(pattern=METHOD_ID_PATTERN)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
CommitSha = Annotated[str, Field(pattern=COMMIT_SHA_PATTERN)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeFloat = Annotated[
    float, Field(strict=True, ge=0.0, allow_inf_nan=False)
]
UnitFloat = Annotated[
    float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
]


class SchemaModel(BaseModel):
    """Strict base for all canonical and cross-component values."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


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


class Population(str, Enum):
    INTERNAL_PROXY = "internal_proxy"
    PUBLIC_VALIDATION = "public_validation"
    UNBIASED_AUDIT = "unbiased_audit"
    HIDDEN_FINAL = "hidden_final"


class Fidelity(str, Enum):
    SMOKE = "smoke"
    PROXY = "proxy"
    FULL = "full"
    FINAL = "final"


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


class Decision(str, Enum):
    PROMOTE = "promote"
    ACCEPT = "accept"
    REJECT = "reject"
    PRUNE = "prune"
    INVALID = "invalid"


class Producer(str, Enum):
    HUMAN = "human"
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    TRAE = "trae"
    GATE_A = "gate_a"
    RUNNER = "runner"
    RECOVERY = "recovery"
    GATE_B = "gate_b"
    EVALUATOR = "evaluator"
    REPORTER = "reporter"


class EvidenceStatus(str, Enum):
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    INVALID = "invalid"


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


class ContextRole(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    RECOVERY = "recovery"
    EVALUATOR = "evaluator"


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


class MethodStatus(str, Enum):
    CANDIDATE = "candidate"
    BLOCKED = "blocked"
    KNOWN_NEGATIVE = "known_negative"
    FORBIDDEN = "forbidden"


class ExperimentKind(str, Enum):
    FRAME = "frame"
    CONTENT = "content"
    CAPACITY = "capacity"
    COMPOSITION = "composition"
    POLICY = "policy"
    OTHER = "other"


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


class RequiredMetricDirection(str, Enum):
    NON_DECREASING_ALL = "non_decreasing_all"
    PRIMARY_ONLY = "primary_only"
    CONTRACT_DEFINED = "contract_defined"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


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


class LessonStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class StopReason(str, Enum):
    CONVERGED = "converged"
    MAX_EXPERIMENTS = "max_experiments"
    MAX_FULL_EVALUATIONS = "max_full_evaluations"
    MAX_WALL_TIME = "max_wall_time"
    MAX_TOKENS = "max_tokens"
    MAX_GPU_TIME = "max_gpu_time"
    CONTRACT_CHANGED = "contract_changed"
    FATAL_INTEGRITY_FAILURE = "fatal_integrity_failure"
    NO_TRUSTED_CANDIDATE = "no_trusted_candidate"
    MANUAL_EMERGENCY_STOP = "manual_emergency_stop"


class RunStatus(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    FAILED = "failed"


class Phase(str, Enum):
    CONTRACT_VERIFICATION = "contract_verification"
    BASELINE_REPRODUCTION = "baseline_reproduction"
    PLANNING = "planning"
    CODING = "coding"
    PATCH_VERIFICATION = "patch_verification"
    EXECUTION_SMOKE = "execution_smoke"
    EXECUTION_PROXY = "execution_proxy"
    EXECUTION_FULL = "execution_full"
    OUTPUT_VERIFICATION = "output_verification"
    EVALUATION = "evaluation"
    DECISION = "decision"
    RECOVERY = "recovery"
    FINALIZATION = "finalization"
    COMPLETE = "complete"


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    PATCH_READY = "patch_ready"
    PATCH_REJECTED = "patch_rejected"
    READY_TO_RUN = "ready_to_run"
    RUNNING = "running"
    RECOVERING = "recovering"
    OUTPUT_READY = "output_ready"
    OUTPUT_VERIFIED = "output_verified"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PRUNED = "pruned"
    INVALID = "invalid"


class ArtifactRef(SchemaModel):
    artifact_id: ArtifactId
    kind: ArtifactKind
    path: str
    sha256: Sha256
    size_bytes: NonNegativeInt
    content_type: Optional[NonEmptyStr] = None

    @field_validator("path")
    @classmethod
    def validate_normalized_path(cls, value: str) -> str:
        return _normalized_relative_path(value)

    def verify_file(
        self,
        repository_root: Union[str, Path],
        approved_roots: Sequence[str] = ("artifacts", "runs"),
    ) -> Path:
        """Verify location, bytes, size, and hash against an explicit root."""
        root = Path(repository_root).resolve()
        candidate = root.joinpath(*PurePosixPath(self.path).parts)
        if candidate.is_symlink():
            raise ValueError("artifact path must not be a symlink")
        resolved = candidate.resolve()
        _require_within(resolved, root, "artifact path escapes repository root")
        approved = [root.joinpath(*PurePosixPath(item).parts).resolve() for item in approved_roots]
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


class ResourceDelta(SchemaModel):
    llm_input_tokens: NonNegativeInt = 0
    llm_output_tokens: NonNegativeInt = 0
    token_measurement: TokenMeasurement = TokenMeasurement.NONE
    wall_time_ms: NonNegativeInt = 0
    cpu_time_ms: NonNegativeInt = 0
    gpu_time_ms: NonNegativeInt = 0
    gpu_count: NonNegativeInt = 0
    peak_rss_mb: Optional[NonNegativeInt] = None
    peak_gpu_memory_mb: Optional[NonNegativeInt] = None
    manual_interventions: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_token_measurement(self) -> "ResourceDelta":
        if self.token_measurement == TokenMeasurement.NONE and (
            self.llm_input_tokens or self.llm_output_tokens
        ):
            raise ValueError("token_measurement=none requires zero token counts")
        return self

    @property
    def gpu_hours(self) -> float:
        return self.gpu_time_ms * self.gpu_count / 3_600_000.0


class MetricSet(SchemaModel):
    metrics: Dict[str, FiniteFloat]
    primary_metric_name: NonEmptyStr
    primary_score: FiniteFloat

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("metrics must not be empty")
        if any(not name or name.strip() != name for name in value):
            raise ValueError("metric names must be non-empty and normalized")
        return value

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


class CostEstimate(SchemaModel):
    llm_tokens_upper_bound: NonNegativeInt
    wall_time_seconds_upper_bound: NonNegativeInt
    gpu_seconds_upper_bound: NonNegativeInt
    cost_tier: CostTier


class CheckResult(SchemaModel):
    name: NonEmptyStr
    status: CheckStatus
    details: Optional[str] = None


class Violation(SchemaModel):
    code: NonEmptyStr
    path: Optional[str] = None
    message: NonEmptyStr

    @field_validator("path")
    @classmethod
    def validate_optional_path(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _normalized_relative_path(value)


class SuccessCriteria(SchemaModel):
    proxy_parent_delta_min: FiniteFloat
    full_parent_delta_min: FiniteFloat
    required_metric_direction: RequiredMetricDirection


class ExperimentSpec(SchemaModel):
    parent_experiment_id: ExperimentId
    parent_commit_sha: CommitSha
    context_id: ContextId
    hypothesis: NonEmptyStr
    family: ExperimentFamily
    change_summary: NonEmptyStr
    target_stage: NonEmptyStr
    target_files: List[str]
    fidelity_plan: List[Fidelity]
    expected_mechanism: NonEmptyStr
    success_criteria: SuccessCriteria
    falsification_condition: NonEmptyStr
    estimated_cost: CostEstimate
    method_card_ids: List[MethodId] = Field(default_factory=list)
    evidence_event_ids: List[EventId] = Field(default_factory=list)
    duplicate_key: Sha256

    @field_validator("target_files")
    @classmethod
    def validate_target_files(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("target_files must not be empty")
        normalized = [_normalized_relative_path(value) for value in values]
        _require_unique(normalized, "target_files")
        return normalized

    @field_validator("fidelity_plan")
    @classmethod
    def validate_fidelity_plan(cls, values: List[Fidelity]) -> List[Fidelity]:
        if not values:
            raise ValueError("fidelity_plan must not be empty")
        allowed = [Fidelity.SMOKE, Fidelity.PROXY, Fidelity.FULL]
        if any(value not in allowed for value in values):
            raise ValueError("fidelity_plan cannot include final")
        positions = [allowed.index(value) for value in values]
        if positions != sorted(set(positions)):
            raise ValueError("fidelity_plan must be an ordered subset without duplicates")
        return values

    @field_validator("method_card_ids", "evidence_event_ids")
    @classmethod
    def validate_unique_references(cls, values: List[str]) -> List[str]:
        _require_unique(values, "reference list")
        return values


class MethodCardMetadata(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    method_id: MethodId
    family: ExperimentFamily
    status: MethodStatus
    tags: List[NonEmptyStr]
    cost_tier: CostTier
    sources: List[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_method_card(self) -> "MethodCardMetadata":
        if not self.tags:
            raise ValueError("method card requires at least one tag")
        _require_unique(self.tags, "method card tags")
        _require_unique(self.sources, "method card sources")
        if any(not source.startswith(("https://", "http://")) for source in self.sources):
            raise ValueError("method card sources must be HTTP(S) URLs")
        return self


class RunBudgets(SchemaModel):
    max_experiments: PositiveInt
    max_full_evaluations: PositiveInt
    max_agent_wall_time_seconds: PositiveInt
    max_llm_tokens: Optional[NonNegativeInt] = None
    max_gpu_seconds: Optional[NonNegativeInt] = None


class ConvergenceRule(SchemaModel):
    epsilon: NonNegativeFloat
    patience: PositiveInt
    population: Population

    @field_validator("population")
    @classmethod
    def require_public_validation(cls, value: Population) -> Population:
        if value != Population.PUBLIC_VALIDATION:
            raise ValueError("convergence population must be public_validation")
        return value


class ResolvedTargetSignature(SchemaModel):
    label: NonEmptyStr
    metrics: List[NonEmptyStr]
    primary_formula: NonEmptyStr

    @field_validator("metrics")
    @classmethod
    def validate_metric_names(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("resolved target metrics must not be empty")
        _require_unique(values, "resolved target metrics")
        return values


class RunStartedPayload(SchemaModel):
    sequential: Literal[True]
    contract_path: str
    contract_sha256: Sha256
    protected_paths_sha256: Sha256
    source_commit: CommitSha
    budgets: RunBudgets
    convergence: ConvergenceRule

    @field_validator("contract_path")
    @classmethod
    def validate_contract_path(cls, value: str) -> str:
        return _normalized_relative_path(value)


class ContractVerifiedPayload(SchemaModel):
    contract_sha256: Sha256
    protected_paths_sha256: Sha256
    data_manifest_sha256: Sha256
    evaluator_sha256: Sha256
    submission_checker_sha256: Sha256
    resolved_target_signature: ResolvedTargetSignature


class BaselineVerifiedPayload(SchemaModel):
    baseline_experiment_id: ExperimentId
    commit_sha: CommitSha
    seed: NonNegativeInt
    metric_set: MetricSet
    published_primary_score: FiniteFloat
    absolute_error: NonNegativeFloat
    tolerance: NonNegativeFloat
    parity_passed: bool

    @model_validator(mode="after")
    def validate_parity(self) -> "BaselineVerifiedPayload":
        if self.baseline_experiment_id != "exp_0000":
            raise ValueError("baseline_experiment_id must be exp_0000")
        expected = self.absolute_error <= self.tolerance
        if self.parity_passed != expected:
            raise ValueError("parity_passed must agree with error and tolerance")
        return self


class ContextCreatedPayload(SchemaModel):
    context_id: ContextId
    role: ContextRole
    purpose: NonEmptyStr
    source_event_ids: List[EventId] = Field(default_factory=list)
    source_method_ids: List[MethodId] = Field(default_factory=list)
    source_commit_shas: List[CommitSha] = Field(default_factory=list)
    excluded_categories: List[NonEmptyStr] = Field(default_factory=list)
    input_token_budget: NonNegativeInt
    estimated_input_tokens: NonNegativeInt
    orthogonality: Optional[UnitFloat] = None

    @model_validator(mode="after")
    def validate_context(self) -> "ContextCreatedPayload":
        expected_prefix = "ctx_%s_" % self.role.value
        if not self.context_id.startswith(expected_prefix):
            raise ValueError("context_id role must match role")
        if self.estimated_input_tokens > self.input_token_budget:
            raise ValueError("estimated context tokens exceed input budget")
        for name, values in (
            ("source_event_ids", self.source_event_ids),
            ("source_method_ids", self.source_method_ids),
            ("source_commit_shas", self.source_commit_shas),
            ("excluded_categories", self.excluded_categories),
        ):
            _require_unique(values, name)
        if self.role == ContextRole.PLANNER and "inconclusive" not in self.excluded_categories:
            raise ValueError("planner contexts must exclude inconclusive evaluations")
        return self


class PlannerRecommendationAction(str, Enum):
    RECOMMEND_STOP = "recommend_stop"
    BLOCKED = "blocked"


class PlannerRecommendedPayload(SchemaModel):
    context_id: ContextId
    action: PlannerRecommendationAction
    reason_code: NonEmptyStr
    reason: NonEmptyStr
    supporting_event_ids: List[EventId]

    @field_validator("supporting_event_ids")
    @classmethod
    def validate_supporting_events(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("planner recommendation requires supporting evidence")
        _require_unique(values, "supporting_event_ids")
        return values


class PatchCreatedPayload(SchemaModel):
    experiment_spec_event_id: EventId
    context_id: ContextId
    base_commit_sha: CommitSha
    patch_commit_sha: CommitSha
    diff_sha256: Sha256
    changed_files: List[str]
    trae_version: NonEmptyStr
    model_id: NonEmptyStr
    must_patch: bool
    steps_used: NonNegativeInt

    @field_validator("changed_files")
    @classmethod
    def validate_changed_files(cls, values: List[str]) -> List[str]:
        normalized = [_normalized_relative_path(value) for value in values]
        _require_unique(normalized, "changed_files")
        return normalized

    @model_validator(mode="after")
    def validate_patch_change(self) -> "PatchCreatedPayload":
        if self.must_patch and not self.changed_files:
            raise ValueError("must_patch requires at least one changed file")
        if self.base_commit_sha == self.patch_commit_sha:
            raise ValueError("patch commit must differ from base commit")
        return self


class PatchCheckedPayload(SchemaModel):
    patch_event_id: EventId
    patch_commit_sha: CommitSha
    diff_sha256: Sha256
    accepted: bool
    receipt_id: Optional[NonEmptyStr]
    receipt_sha256: Optional[Sha256]
    checks: List[CheckResult]
    violations: List[Violation]

    @model_validator(mode="after")
    def validate_check_outcome(self) -> "PatchCheckedPayload":
        if not self.checks:
            raise ValueError("patch check must contain checks")
        _require_unique([check.name for check in self.checks], "patch check names")
        failed = any(check.status == CheckStatus.FAIL for check in self.checks)
        if self.accepted:
            if failed or self.violations:
                raise ValueError("accepted patch cannot have failed checks or violations")
            if self.receipt_id is None or self.receipt_sha256 is None:
                raise ValueError("accepted patch requires a receipt identity and hash")
        else:
            if not failed or not self.violations:
                raise ValueError("rejected patch requires a failed check and violation")
            if self.receipt_id is not None or self.receipt_sha256 is not None:
                raise ValueError("rejected patch cannot issue a receipt")
        return self


class ExecutionLimits(SchemaModel):
    timeout_seconds: PositiveInt
    memory_mb: PositiveInt
    gpu_memory_mb: Optional[PositiveInt] = None
    network_enabled: bool


class ExecutionStartedPayload(SchemaModel):
    patch_receipt_id: NonEmptyStr
    patch_commit_sha: CommitSha
    fidelity: Fidelity
    command_id: NonEmptyStr
    seed: NonNegativeInt
    data_manifest_sha256: Sha256
    limits: ExecutionLimits


class ExecutionFinishedPayload(SchemaModel):
    execution_started_event_id: EventId
    patch_commit_sha: CommitSha
    fidelity: Fidelity
    outcome: ExecutionOutcome
    exit_code: Optional[int]
    error_class: Optional[NonEmptyStr]
    error_fingerprint: Optional[Sha256]
    error_summary: Optional[NonEmptyStr]
    prediction_artifact_id: Optional[ArtifactId]
    checkpoint_artifact_id: Optional[ArtifactId]

    @model_validator(mode="after")
    def validate_execution_outcome(self) -> "ExecutionFinishedPayload":
        errors = (self.error_class, self.error_fingerprint, self.error_summary)
        if self.outcome == ExecutionOutcome.SUCCESS:
            if any(value is not None for value in errors):
                raise ValueError("successful execution cannot contain error fields")
            if self.exit_code not in (0, None):
                raise ValueError("successful execution exit_code must be zero or null")
        elif any(value is None for value in errors):
            raise ValueError("failed execution requires class, fingerprint, and summary")
        return self


class RecoveryDecidedPayload(SchemaModel):
    failure_event_id: EventId
    repair_attempt: PositiveInt
    action: RecoveryAction
    reason_code: NonEmptyStr
    instructions: NonEmptyStr
    same_error_count: PositiveInt
    remaining_repair_budget: NonNegativeInt

    @model_validator(mode="after")
    def validate_recovery_limits(self) -> "RecoveryDecidedPayload":
        if self.action == RecoveryAction.TRAE_REPAIR and self.repair_attempt > 2:
            raise ValueError("at most two Trae repairs are allowed")
        if self.same_error_count >= 2 and self.action != RecoveryAction.ABANDON:
            raise ValueError("the same error twice requires abandon")
        return self


class ScoreStats(SchemaModel):
    rows: PositiveInt
    unique_scores: PositiveInt
    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_score_stats(self) -> "ScoreStats":
        if self.unique_scores > self.rows:
            raise ValueError("unique_scores cannot exceed rows")
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class OutputCheckedPayload(SchemaModel):
    execution_finished_event_id: EventId
    prediction_artifact_id: ArtifactId
    ordered_row_identity_sha256: Sha256
    ordered_prediction_sha256: Sha256
    accepted: bool
    checks: Dict[str, CheckStatus]
    score_stats: ScoreStats
    violations: List[Violation]

    @model_validator(mode="after")
    def validate_output_outcome(self) -> "OutputCheckedPayload":
        if not self.checks:
            raise ValueError("output check must contain checks")
        failed = any(value == CheckStatus.FAIL for value in self.checks.values())
        if self.accepted and (failed or self.violations):
            raise ValueError("accepted output cannot have failed checks or violations")
        if not self.accepted and (not failed or not self.violations):
            raise ValueError("rejected output requires a failed check and violation")
        return self


class PredictionChange(SchemaModel):
    spearman_vs_parent: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    changed_row_fraction: UnitFloat


class TrustAssessment(SchemaModel):
    verdict: TrustVerdict
    stability: Stability
    integrity: Integrity
    flags: List[NonEmptyStr] = Field(default_factory=list)
    eta_applied: Optional[NonNegativeFloat] = None
    seed_mean: Optional[FiniteFloat] = None
    seed_stderr: Optional[NonNegativeFloat] = None
    seed_count: PositiveInt = 1

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, values: List[str]) -> List[str]:
        _require_unique(values, "trust flags")
        return values

    @model_validator(mode="after")
    def validate_trust_consistency(self) -> "TrustAssessment":
        if self.integrity == Integrity.COMPROMISED and self.verdict != TrustVerdict.SUSPICIOUS:
            raise ValueError("compromised integrity requires suspicious verdict")
        if self.verdict == TrustVerdict.NO_OP and self.stability != Stability.NOT_APPLICABLE:
            raise ValueError("no_op stability must be not_applicable")
        seed_values = (self.seed_mean, self.seed_stderr)
        if any(value is None for value in seed_values) != all(
            value is None for value in seed_values
        ):
            raise ValueError("seed mean and standard error must be supplied together")
        if self.stability in (Stability.CONFIRMED, Stability.UNSTABLE):
            if self.seed_count < 3 or any(value is None for value in seed_values):
                raise ValueError(
                    "confirmed or unstable trust requires at least three seed results"
                )
        return self


class EvaluationCompletedPayload(SchemaModel):
    output_checked_event_id: EventId
    prediction_artifact_id: ArtifactId
    population: Population
    fidelity: Fidelity
    seed: NonNegativeInt
    public_query_index: Optional[PositiveInt]
    evaluator_sha256: Sha256
    contract_sha256: Sha256
    metric_set: MetricSet
    baseline_delta: FiniteFloat
    parent_delta: FiniteFloat
    previous_best_delta: FiniteFloat
    prediction_change: PredictionChange
    trust: TrustAssessment
    seed_evidence_event_ids: List[EventId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_population_route(self) -> "EvaluationCompletedPayload":
        _validate_population_fidelity(
            self.population, self.fidelity, self.public_query_index
        )
        _require_unique(self.seed_evidence_event_ids, "seed_evidence_event_ids")
        if self.trust.seed_count != len(self.seed_evidence_event_ids) + 1:
            raise ValueError(
                "trust seed_count must include the current evaluation and all "
                "seed evidence events"
            )
        return self


class ExperimentDecidedPayload(SchemaModel):
    evaluation_event_id: Optional[EventId]
    decision: Decision
    reason_code: NonEmptyStr
    fidelity_completed: Fidelity
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Optional[Fidelity]
    supporting_event_ids: List[EventId]

    @model_validator(mode="after")
    def validate_decision(self) -> "ExperimentDecidedPayload":
        if not self.supporting_event_ids:
            raise ValueError("experiment decision requires supporting events")
        _require_unique(self.supporting_event_ids, "supporting_event_ids")
        if self.best_eligible and not self.parent_eligible:
            raise ValueError("best eligibility requires parent eligibility")
        if self.decision == Decision.PROMOTE:
            if self.next_fidelity is None:
                raise ValueError("promotion requires next_fidelity")
            if self.parent_eligible or self.best_eligible:
                raise ValueError("promotion cannot grant parent or best eligibility")
        elif self.next_fidelity is not None:
            raise ValueError("terminal decisions cannot set next_fidelity")
        if self.evaluation_event_id is None:
            if self.decision != Decision.PROMOTE or self.fidelity_completed != Fidelity.SMOKE:
                raise ValueError("only smoke promotion may omit evaluation_event_id")
            if len(self.supporting_event_ids) < 2:
                raise ValueError("smoke promotion requires execution and Gate-B evidence")
        if self.fidelity_completed == Fidelity.FINAL:
            raise ValueError("hidden-final results cannot produce experiment decisions")
        return self


class BestUpdatedPayload(SchemaModel):
    previous_best_experiment_id: ExperimentId
    previous_best_commit_sha: CommitSha
    previous_best_primary_score: FiniteFloat
    new_best_experiment_id: ExperimentId
    new_best_commit_sha: CommitSha
    new_best_primary_score: FiniteFloat
    evaluation_event_id: EventId

    @model_validator(mode="after")
    def validate_improvement(self) -> "BestUpdatedPayload":
        if self.new_best_primary_score <= self.previous_best_primary_score:
            raise ValueError("new best score must exceed previous best score")
        if self.new_best_experiment_id == self.previous_best_experiment_id:
            raise ValueError("best update must select a different experiment")
        return self


class LessonRecordedPayload(SchemaModel):
    lesson_id: LessonId
    category: LessonCategory
    status: LessonStatus
    tags: List[NonEmptyStr]
    summary: NonEmptyStr
    applicability: NonEmptyStr
    avoid_when: NonEmptyStr
    confidence: UnitFloat
    source_event_ids: List[EventId]
    source_commit_shas: List[CommitSha] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lesson(self) -> "LessonRecordedPayload":
        if self.status != LessonStatus.ACTIVE:
            raise ValueError("new lessons must start active")
        if not self.tags or not self.source_event_ids:
            raise ValueError("lesson requires tags and source events")
        _require_unique(self.tags, "lesson tags")
        _require_unique(self.source_event_ids, "lesson source events")
        _require_unique(self.source_commit_shas, "lesson source commits")
        return self


class LessonStatusChangedPayload(SchemaModel):
    lesson_id: LessonId
    new_status: LessonStatus
    reason: NonEmptyStr
    source_event_ids: List[EventId]

    @field_validator("source_event_ids")
    @classmethod
    def validate_source_events(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("lesson status change requires source events")
        _require_unique(values, "lesson status source events")
        return values


class ManualInterventionPayload(SchemaModel):
    actor: NonEmptyStr
    reason: NonEmptyStr
    action: NonEmptyStr
    affected_experiment_id: Optional[ExperimentId]
    code_changed: bool
    effect: NonEmptyStr


class BudgetSnapshot(SchemaModel):
    agent_wall_time_seconds: NonNegativeInt
    llm_input_tokens_provider: NonNegativeInt
    llm_output_tokens_provider: NonNegativeInt
    llm_input_tokens_estimated: NonNegativeInt
    llm_output_tokens_estimated: NonNegativeInt
    gpu_hours: NonNegativeFloat


class RunStoppedPayload(SchemaModel):
    reason: StopReason
    best_experiment_id: Optional[ExperimentId]
    best_commit_sha: Optional[CommitSha]
    best_primary_score: Optional[FiniteFloat]
    experiments_proposed: NonNegativeInt
    full_evaluations_completed: NonNegativeInt
    consecutive_non_improving_full_evaluations: NonNegativeInt
    total_manual_interventions: NonNegativeInt
    budget_snapshot: BudgetSnapshot

    @model_validator(mode="after")
    def validate_best_identity(self) -> "RunStoppedPayload":
        best_values = (
            self.best_experiment_id,
            self.best_commit_sha,
            self.best_primary_score,
        )
        if any(value is None for value in best_values) and any(
            value is not None for value in best_values
        ):
            raise ValueError("best experiment, commit, and score must be all set or all null")
        reasons_without_best = {
            StopReason.NO_TRUSTED_CANDIDATE,
            StopReason.CONTRACT_CHANGED,
            StopReason.FATAL_INTEGRITY_FAILURE,
            StopReason.MANUAL_EMERGENCY_STOP,
        }
        if self.reason not in reasons_without_best and all(value is None for value in best_values):
            raise ValueError("stop reason requires a trusted best candidate")
        return self


class FinalSelectedPayload(SchemaModel):
    experiment_id: ExperimentId
    commit_sha: CommitSha
    selection_evaluation_event_id: EventId
    clean_reproduction_passed: Literal[True]
    checkpoint_artifact_id: ArtifactId
    validation_predictions_artifact_id: ArtifactId
    selection_reason: NonEmptyStr


class SubmissionCheckedPayload(SchemaModel):
    final_selected_event_id: EventId
    submission_artifact_id: ArtifactId
    checker_sha256: Sha256
    accepted: bool
    violations: List[Violation]

    @model_validator(mode="after")
    def validate_submission(self) -> "SubmissionCheckedPayload":
        if self.accepted and self.violations:
            raise ValueError("accepted submission cannot have violations")
        if not self.accepted and not self.violations:
            raise ValueError("rejected submission requires violations")
        return self


# Cross-component values. They carry transport identity; event payloads above
# deliberately omit fields already present in the universal envelope.


class ExperimentSpecMessage(ExperimentSpec):
    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: RunId
    experiment_id: ExperimentId


class PlannerAction(str, Enum):
    PROPOSE = "propose"
    RECOMMEND_STOP = "recommend_stop"
    BLOCKED = "blocked"


class PlannerOutput(SchemaModel):
    action: PlannerAction
    spec: Optional[ExperimentSpecMessage]
    reason_code: NonEmptyStr
    reason: NonEmptyStr
    supporting_event_ids: List[EventId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_planner_output(self) -> "PlannerOutput":
        if (self.action == PlannerAction.PROPOSE) != (self.spec is not None):
            raise ValueError("planner propose action requires exactly one spec")
        _require_unique(self.supporting_event_ids, "supporting_event_ids")
        return self


class PatchCandidate(SchemaModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    experiment_spec_event_id: EventId
    context_id: ContextId
    base_commit_sha: CommitSha
    patch_commit_sha: CommitSha
    diff_sha256: Sha256
    changed_files: List[str]
    diff_artifact: ArtifactRef
    trajectory_artifact: ArtifactRef
    trae_version: NonEmptyStr
    model_id: NonEmptyStr
    steps_used: NonNegativeInt
    resource_delta: ResourceDelta

    @field_validator("changed_files")
    @classmethod
    def validate_candidate_files(cls, values: List[str]) -> List[str]:
        normalized = [_normalized_relative_path(value) for value in values]
        _require_unique(normalized, "changed_files")
        return normalized

    @model_validator(mode="after")
    def validate_candidate_artifacts(self) -> "PatchCandidate":
        if self.diff_artifact.kind != ArtifactKind.DIFF:
            raise ValueError("diff_artifact must have kind=diff")
        if self.trajectory_artifact.kind != ArtifactKind.TRAJECTORY:
            raise ValueError("trajectory_artifact must have kind=trajectory")
        if self.diff_artifact.sha256 != self.diff_sha256:
            raise ValueError("diff artifact hash must equal diff_sha256")
        return self


class PatchCheckResult(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    patch_commit_sha: CommitSha
    diff_sha256: Sha256
    accepted: bool
    receipt_id: Optional[NonEmptyStr]
    receipt_artifact: Optional[ArtifactRef]
    checks: List[CheckResult]
    violations: List[Violation]

    @model_validator(mode="after")
    def validate_patch_result(self) -> "PatchCheckResult":
        failed = any(check.status == CheckStatus.FAIL for check in self.checks)
        if self.accepted:
            if (
                failed
                or self.violations
                or self.receipt_id is None
                or self.receipt_artifact is None
            ):
                raise ValueError("accepted patch result requires clean checks and receipt")
            if self.receipt_artifact.kind != ArtifactKind.VERIFICATION_RECEIPT:
                raise ValueError("receipt artifact has the wrong kind")
        elif not failed or not self.violations:
            raise ValueError("rejected patch result requires failed checks and violations")
        return self


class RunRequest(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    fidelity: Fidelity
    command_id: NonEmptyStr
    patch_commit_sha: CommitSha
    patch_receipt_id: NonEmptyStr
    seed: NonNegativeInt
    data_manifest_sha256: Sha256
    timeout_seconds: PositiveInt
    memory_limit_mb: PositiveInt
    gpu_memory_limit_mb: Optional[PositiveInt] = None
    network_enabled: bool


class TelemetrySample(SchemaModel):
    timestamp: str
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    elapsed_ms: NonNegativeInt
    process_alive: bool
    last_output_age_ms: NonNegativeInt
    cpu_percent: Annotated[
        float, Field(strict=True, ge=0.0, le=100.0, allow_inf_nan=False)
    ]
    rss_mb: NonNegativeInt
    gpu_utilization_percent: Optional[
        Annotated[float, Field(strict=True, ge=0.0, le=100.0, allow_inf_nan=False)]
    ] = None
    gpu_memory_mb: Optional[NonNegativeInt] = None
    loss: Optional[FiniteFloat] = None
    gradient_norm: Optional[NonNegativeFloat] = None
    disk_free_mb: NonNegativeInt
    recent_output_tail: str

    @field_validator("timestamp")
    @classmethod
    def validate_sample_timestamp(cls, value: str) -> str:
        return _utc_timestamp(value)


class MonitorAction(str, Enum):
    CONTINUE = "continue"
    TERMINATE = "terminate"


class MonitorDirective(SchemaModel):
    action: MonitorAction
    reason_code: Optional[NonEmptyStr]
    summary: Optional[NonEmptyStr]

    @model_validator(mode="after")
    def validate_monitor_directive(self) -> "MonitorDirective":
        if self.action == MonitorAction.TERMINATE and (
            self.reason_code is None or self.summary is None
        ):
            raise ValueError("terminate directive requires reason and summary")
        if self.action == MonitorAction.CONTINUE and (
            self.reason_code is not None or self.summary is not None
        ):
            raise ValueError("continue directive cannot carry termination details")
        return self


class RunResult(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    fidelity: Fidelity
    patch_commit_sha: CommitSha
    outcome: ExecutionOutcome
    exit_code: Optional[int]
    error_class: Optional[NonEmptyStr]
    error_fingerprint: Optional[Sha256]
    error_summary: Optional[NonEmptyStr]
    log_artifact: ArtifactRef
    telemetry_artifact: Optional[ArtifactRef] = None
    checkpoint_artifact: Optional[ArtifactRef] = None
    prediction_artifact: Optional[ArtifactRef] = None
    resource_delta: ResourceDelta

    @model_validator(mode="after")
    def validate_run_result(self) -> "RunResult":
        if self.log_artifact.kind != ArtifactKind.LOG:
            raise ValueError("log_artifact must have kind=log")
        if (
            self.telemetry_artifact is not None
            and self.telemetry_artifact.kind != ArtifactKind.METRICS
        ):
            raise ValueError("telemetry_artifact must have kind=metrics")
        if (
            self.checkpoint_artifact is not None
            and self.checkpoint_artifact.kind != ArtifactKind.CHECKPOINT
        ):
            raise ValueError("checkpoint_artifact must have kind=checkpoint")
        if (
            self.prediction_artifact is not None
            and self.prediction_artifact.kind != ArtifactKind.PREDICTIONS
        ):
            raise ValueError("prediction_artifact must have kind=predictions")
        errors = (self.error_class, self.error_fingerprint, self.error_summary)
        if self.outcome == ExecutionOutcome.SUCCESS:
            if any(item is not None for item in errors):
                raise ValueError("successful run cannot contain error fields")
            if self.exit_code not in (0, None):
                raise ValueError("successful run exit_code must be zero or null")
        elif any(item is None for item in errors):
            raise ValueError("failed run requires complete normalized error fields")
        return self


class OutputCheckResult(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    prediction_artifact: ArtifactRef
    ordered_row_identity_sha256: Sha256
    ordered_prediction_sha256: Sha256
    accepted: bool
    checks: Dict[str, CheckStatus]
    score_stats: ScoreStats
    violations: List[Violation]

    @model_validator(mode="after")
    def validate_output_result(self) -> "OutputCheckResult":
        if self.prediction_artifact.kind != ArtifactKind.PREDICTIONS:
            raise ValueError("prediction_artifact must have kind=predictions")
        failed = any(status == CheckStatus.FAIL for status in self.checks.values())
        if self.accepted and (failed or self.violations):
            raise ValueError("accepted output result must be clean")
        if not self.accepted and (not failed or not self.violations):
            raise ValueError("rejected output result requires failure evidence")
        return self


class LessonCandidate(SchemaModel):
    origin: LessonOrigin
    category: LessonCategory
    tags: List[NonEmptyStr]
    summary: NonEmptyStr
    applicability: NonEmptyStr
    avoid_when: NonEmptyStr
    confidence: UnitFloat
    source_event_ids: List[EventId]
    source_commit_shas: List[CommitSha] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lesson_candidate(self) -> "LessonCandidate":
        if not self.tags or not self.source_event_ids:
            raise ValueError("lesson candidate requires tags and source events")
        _require_unique(self.tags, "lesson candidate tags")
        _require_unique(self.source_event_ids, "lesson candidate source events")
        _require_unique(self.source_commit_shas, "lesson candidate commits")
        return self


class RecoveryDecision(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    failure_event_id: EventId
    repair_attempt: PositiveInt
    action: RecoveryAction
    reason_code: NonEmptyStr
    instructions: NonEmptyStr
    same_error_count: PositiveInt
    remaining_repair_budget: NonNegativeInt
    lesson_candidate: Optional[LessonCandidate] = None

    @model_validator(mode="after")
    def validate_recovery_decision(self) -> "RecoveryDecision":
        if self.action == RecoveryAction.TRAE_REPAIR and self.repair_attempt > 2:
            raise ValueError("at most two Trae repair decisions are allowed")
        if self.same_error_count >= 2 and self.action != RecoveryAction.ABANDON:
            raise ValueError("repeated error requires abandon")
        return self


class EvaluationRequest(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    output_checked_event_id: EventId
    prediction_artifact: ArtifactRef
    population: Population
    fidelity: Fidelity
    seed: NonNegativeInt
    contract_sha256: Sha256
    evaluator_sha256: Sha256
    baseline_summary: MetricSet
    parent_summary: MetricSet
    previous_best_summary: MetricSet
    public_query_index: Optional[PositiveInt]

    @model_validator(mode="after")
    def validate_evaluation_request(self) -> "EvaluationRequest":
        if self.prediction_artifact.kind != ArtifactKind.PREDICTIONS:
            raise ValueError("evaluation requires a predictions artifact")
        _validate_population_fidelity(
            self.population, self.fidelity, self.public_query_index
        )
        return self


class EvaluationResult(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    attempt: PositiveInt
    population: Population
    fidelity: Fidelity
    seed: NonNegativeInt
    public_query_index: Optional[PositiveInt]
    evaluator_sha256: Sha256
    contract_sha256: Sha256
    metric_set: MetricSet
    baseline_delta: FiniteFloat
    parent_delta: FiniteFloat
    previous_best_delta: FiniteFloat
    prediction_change: PredictionChange
    trust: TrustAssessment
    seed_evidence_event_ids: List[EventId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evaluation_result(self) -> "EvaluationResult":
        _validate_population_fidelity(
            self.population, self.fidelity, self.public_query_index
        )
        _require_unique(self.seed_evidence_event_ids, "seed_evidence_event_ids")
        if self.trust.seed_count != len(self.seed_evidence_event_ids) + 1:
            raise ValueError(
                "trust seed_count must include the current evaluation and all "
                "seed evidence events"
            )
        return self


class ExperimentDecision(SchemaModel):
    run_id: RunId
    experiment_id: ExperimentId
    evaluation_event_id: Optional[EventId]
    decision: Decision
    reason_code: NonEmptyStr
    fidelity_completed: Fidelity
    parent_eligible: bool
    best_eligible: bool
    next_fidelity: Optional[Fidelity]
    supporting_event_ids: List[EventId]
    lesson_candidate: Optional[LessonCandidate] = None

    @model_validator(mode="after")
    def validate_experiment_decision(self) -> "ExperimentDecision":
        ExperimentDecidedPayload(
            evaluation_event_id=self.evaluation_event_id,
            decision=self.decision,
            reason_code=self.reason_code,
            fidelity_completed=self.fidelity_completed,
            parent_eligible=self.parent_eligible,
            best_eligible=self.best_eligible,
            next_fidelity=self.next_fidelity,
            supporting_event_ids=self.supporting_event_ids,
        )
        return self


class RemainingBudgets(SchemaModel):
    experiments: NonNegativeInt
    full_evaluations: NonNegativeInt
    agent_wall_time_seconds: NonNegativeInt
    llm_tokens: Optional[NonNegativeInt]
    gpu_seconds: Optional[NonNegativeInt]


class ResourceTotals(SchemaModel):
    llm_input_tokens_provider: NonNegativeInt = 0
    llm_output_tokens_provider: NonNegativeInt = 0
    llm_input_tokens_estimated: NonNegativeInt = 0
    llm_output_tokens_estimated: NonNegativeInt = 0
    cpu_time_ms: NonNegativeInt = 0
    gpu_weighted_time_ms: NonNegativeInt = 0
    manual_interventions: NonNegativeInt = 0


class RunState(SchemaModel):
    run_id: RunId
    status: RunStatus
    phase: Phase
    active_experiment_id: Optional[ExperimentId]
    active_attempt: Optional[NonNegativeInt]
    active_fidelity: Optional[Fidelity]
    best_experiment_id: Optional[ExperimentId]
    best_commit_sha: Optional[CommitSha]
    best_primary_score: Optional[FiniteFloat]
    experiments_proposed: NonNegativeInt
    full_evaluations_completed: NonNegativeInt
    public_validation_queries: NonNegativeInt
    consecutive_non_improving_full_evaluations: NonNegativeInt
    remaining_budgets: RemainingBudgets
    resource_totals: ResourceTotals

    @model_validator(mode="after")
    def validate_run_state(self) -> "RunState":
        active = (
            self.active_experiment_id,
            self.active_attempt,
            self.active_fidelity,
        )
        if any(value is None for value in active) and any(value is not None for value in active):
            raise ValueError("active experiment identity must be all set or all null")
        best = (
            self.best_experiment_id,
            self.best_commit_sha,
            self.best_primary_score,
        )
        if any(value is None for value in best) and any(value is not None for value in best):
            raise ValueError("best identity must be all set or all null")
        return self


class ExperimentNode(SchemaModel):
    experiment_id: ExperimentId
    parent_experiment_id: Optional[ExperimentId]
    hypothesis: NonEmptyStr
    family: ExperimentFamily
    base_commit_sha: CommitSha
    latest_patch_commit_sha: Optional[CommitSha]
    attempts: NonNegativeInt
    highest_fidelity_completed: Optional[Fidelity]
    status: ExperimentStatus
    metric_set: Optional[MetricSet]
    trust_verdict: Optional[TrustVerdict]
    parent_eligible: bool
    best_eligible: bool
    terminal_event_id: Optional[EventId]

    @model_validator(mode="after")
    def validate_node(self) -> "ExperimentNode":
        terminal = {
            ExperimentStatus.ACCEPTED,
            ExperimentStatus.REJECTED,
            ExperimentStatus.PRUNED,
            ExperimentStatus.INVALID,
        }
        if (self.status in terminal) != (self.terminal_event_id is not None):
            raise ValueError("terminal status and terminal_event_id must agree")
        if self.best_eligible and not self.parent_eligible:
            raise ValueError("best eligibility requires parent eligibility")
        return self


EventPayload = Union[
    RunStartedPayload,
    ContractVerifiedPayload,
    BaselineVerifiedPayload,
    ContextCreatedPayload,
    PlannerRecommendedPayload,
    ExperimentSpec,
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
]


EVENT_PAYLOAD_MODELS: Mapping[EventType, Type[SchemaModel]] = {
    EventType.RUN_STARTED: RunStartedPayload,
    EventType.CONTRACT_VERIFIED: ContractVerifiedPayload,
    EventType.BASELINE_VERIFIED: BaselineVerifiedPayload,
    EventType.CONTEXT_CREATED: ContextCreatedPayload,
    EventType.PLANNER_RECOMMENDED: PlannerRecommendedPayload,
    EventType.EXPERIMENT_PROPOSED: ExperimentSpec,
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


class Event(SchemaModel):
    """One canonical JSONL event with an event-type-discriminated payload."""

    payload_models: ClassVar[Mapping[EventType, Type[SchemaModel]]] = EVENT_PAYLOAD_MODELS

    schema_version: SchemaVersion = SCHEMA_VERSION
    seq: PositiveInt
    event_id: EventId
    timestamp: str
    run_id: RunId
    experiment_id: Optional[ExperimentId]
    attempt: Optional[NonNegativeInt]
    event_type: EventType
    producer: Producer
    evidence_status: EvidenceStatus
    causation_event_id: Optional[EventId]
    idempotency_key: NonEmptyStr
    payload: EventPayload
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    resource_delta: ResourceDelta = Field(default_factory=ResourceDelta)
    prev_event_hash: Sha256
    event_hash: Sha256

    @model_validator(mode="before")
    @classmethod
    def discriminate_payload(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        data = dict(raw)
        try:
            event_type = EventType(data.get("event_type"))
        except (TypeError, ValueError):
            return data
        payload_model = cls.payload_models[event_type]
        payload = data.get("payload")
        if not isinstance(payload, payload_model):
            data["payload"] = payload_model.model_validate(payload)
        return data

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _utc_timestamp(value)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        if any(character in value for character in ("\n", "\r", "\x00")):
            raise ValueError("idempotency_key contains a forbidden character")
        return value

    @field_validator("artifacts")
    @classmethod
    def validate_artifacts(cls, values: List[ArtifactRef]) -> List[ArtifactRef]:
        _require_unique([item.artifact_id for item in values], "artifact IDs")
        return values

    @model_validator(mode="after")
    def validate_envelope(self) -> "Event":
        expected_event_id = "evt_%06d" % self.seq
        if self.event_id != expected_event_id:
            raise ValueError("event_id must equal evt_{seq:06d}")
        if self.causation_event_id is not None:
            cause_seq = int(self.causation_event_id.removeprefix("evt_"))
            if cause_seq >= self.seq:
                raise ValueError("causation_event_id must precede this event")
        if self.experiment_id is None and self.attempt is not None:
            raise ValueError("run-level events must have null attempt")
        if self.experiment_id is not None and self.attempt is None:
            raise ValueError("experiment events require an attempt")

        run_level = {
            EventType.RUN_STARTED,
            EventType.CONTRACT_VERIFIED,
            EventType.PLANNER_RECOMMENDED,
            EventType.RUN_STOPPED,
        }
        if self.event_type in run_level and (
            self.experiment_id is not None or self.attempt is not None
        ):
            raise ValueError("run-level event has experiment identity")
        if self.event_type == EventType.RUN_STARTED:
            if self.seq != 1 or self.causation_event_id is not None:
                raise ValueError("run.started must be the first uncaused event")
            if self.prev_event_hash != ZERO_SHA256:
                raise ValueError("first event prev_event_hash must be zero")
            if self.producer != Producer.ORCHESTRATOR:
                raise ValueError("run.started producer must be orchestrator")
        elif self.seq > 1 and self.prev_event_hash == ZERO_SHA256:
            raise ValueError("non-first event cannot use the zero predecessor hash")

        if self.event_type == EventType.BASELINE_VERIFIED:
            if self.experiment_id != "exp_0000" or self.attempt != 1:
                raise ValueError("baseline.verified must use exp_0000 attempt 1")
        if self.event_type == EventType.EXPERIMENT_PROPOSED and self.attempt != 0:
            raise ValueError("experiment.proposed must use attempt 0")
        if self.event_type == EventType.PLANNER_RECOMMENDED and self.producer != Producer.PLANNER:
            raise ValueError("planner.recommended producer must be planner")
        if self.event_type == EventType.PATCH_CREATED:
            if self.producer != Producer.TRAE:
                raise ValueError("patch.created producer must be trae")
            if self.evidence_status != EvidenceStatus.PROVISIONAL:
                raise ValueError("patch.created evidence must be provisional")
        if self.event_type == EventType.PATCH_CHECKED and self.producer != Producer.GATE_A:
            raise ValueError("patch.checked producer must be gate_a")
        if self.event_type == EventType.OUTPUT_CHECKED and self.producer != Producer.GATE_B:
            raise ValueError("output.checked producer must be gate_b")
        if (
            self.event_type == EventType.EVALUATION_COMPLETED
            and self.producer != Producer.EVALUATOR
        ):
            raise ValueError("evaluation.completed producer must be evaluator")
        if self.event_type == EventType.MANUAL_INTERVENTION:
            if self.producer != Producer.HUMAN:
                raise ValueError("manual.intervention producer must be human")
            if self.resource_delta.manual_interventions != 1:
                raise ValueError("manual.intervention resource delta must count one")
        elif self.resource_delta.manual_interventions != 0:
            raise ValueError("only manual.intervention may increment intervention count")

        expected_payload = self.payload_models[self.event_type]
        if type(self.payload) is not expected_payload:
            raise ValueError("payload model does not match event_type")
        if isinstance(self.payload, FinalSelectedPayload):
            if self.experiment_id != self.payload.experiment_id:
                raise ValueError("final selection payload and envelope experiment differ")
        return self

    def canonical_bytes(self) -> bytes:
        data = self.model_dump(mode="json", exclude={"event_hash"})
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
        values = dict(values)
        values["event_hash"] = ZERO_SHA256
        draft = cls.model_validate(values)
        return draft.model_copy(update={"event_hash": draft.computed_event_hash()})


def parse_event_json(line: Union[str, bytes], verify_hash: bool = True) -> Event:
    """Parse one compact JSON object and optionally verify its content hash."""
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
        matches = [index for index, heading in enumerate(headings) if heading == required]
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
    headings = [line.strip() for line in text[match.end() :].splitlines() if line.startswith("## ")]
    position = -1
    for required in METHOD_CARD_HEADINGS:
        matches = [index for index, heading in enumerate(headings) if heading == required]
        if len(matches) != 1 or matches[0] <= position:
            raise ValueError("method card headings are missing, duplicated, or out of order")
        position = matches[0]
    return metadata


def family_kind(family: ExperimentFamily) -> ExperimentKind:
    return FAMILY_KIND[family]


def _validate_population_fidelity(
    population: Population,
    fidelity: Fidelity,
    public_query_index: Optional[int],
) -> None:
    if population != Population.PUBLIC_VALIDATION and public_query_index is not None:
        raise ValueError("only public validation may have a public query index")
    if population == Population.PUBLIC_VALIDATION and fidelity != Fidelity.FULL:
        raise ValueError("public_validation requires full fidelity")
    if population == Population.INTERNAL_PROXY and fidelity != Fidelity.PROXY:
        raise ValueError("internal_proxy requires proxy fidelity")
    if population == Population.UNBIASED_AUDIT and fidelity != Fidelity.FULL:
        raise ValueError("unbiased_audit requires full fidelity")
    if population == Population.HIDDEN_FINAL and fidelity != Fidelity.FINAL:
        raise ValueError("hidden_final requires final fidelity")
    if fidelity == Fidelity.FINAL and population != Population.HIDDEN_FINAL:
        raise ValueError("final fidelity is reserved for hidden_final")


def _normalized_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise ValueError("path must use normalized POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("path must be normalized and repository-relative")
    normalized = path.as_posix()
    if normalized != value or value.startswith("/"):
        raise ValueError("path must be normalized and repository-relative")
    return normalized


def _utc_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be RFC 3339 UTC ending in Z")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z",
        value,
    ):
        raise ValueError("timestamp must be RFC 3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("timestamp is not a valid date-time") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("timestamp must use UTC")
    return value


def _require_unique(values: Sequence[Any], name: str) -> None:
    serialized = [str(value) for value in values]
    if len(serialized) != len(set(serialized)):
        raise ValueError("%s must not contain duplicates" % name)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_within(path: Path, root: Path, message: str) -> None:
    if not _is_within(path, root):
        raise ValueError(message)


__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "BaselineVerifiedPayload",
    "BestUpdatedPayload",
    "CheckResult",
    "CheckStatus",
    "ContextCreatedPayload",
    "ContextRole",
    "ContractVerifiedPayload",
    "CostEstimate",
    "Decision",
    "EvaluationCompletedPayload",
    "EvaluationRequest",
    "EvaluationResult",
    "Event",
    "EventType",
    "EvidenceStatus",
    "ExecutionFinishedPayload",
    "ExecutionOutcome",
    "ExecutionStartedPayload",
    "ExperimentDecision",
    "ExperimentDecidedPayload",
    "ExperimentFamily",
    "ExperimentKind",
    "ExperimentNode",
    "ExperimentSpec",
    "ExperimentSpecMessage",
    "Fidelity",
    "FinalSelectedPayload",
    "Integrity",
    "LessonCandidate",
    "LessonCategory",
    "LessonOrigin",
    "LessonRecordedPayload",
    "LessonStatus",
    "LessonStatusChangedPayload",
    "ManualInterventionPayload",
    "MethodCardMetadata",
    "MethodStatus",
    "MetricSet",
    "MonitorDirective",
    "OutputCheckResult",
    "OutputCheckedPayload",
    "PatchCandidate",
    "PatchCheckResult",
    "PatchCheckedPayload",
    "PatchCreatedPayload",
    "Phase",
    "PlannerOutput",
    "Population",
    "PredictionChange",
    "Producer",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryDecidedPayload",
    "ResourceDelta",
    "ResourceTotals",
    "RunRequest",
    "RunResult",
    "RunStartedPayload",
    "RunState",
    "RunStatus",
    "RunStoppedPayload",
    "SCHEMA_VERSION",
    "ScoreStats",
    "Stability",
    "SubmissionCheckedPayload",
    "TelemetrySample",
    "TokenMeasurement",
    "TrustAssessment",
    "TrustVerdict",
    "Violation",
    "ZERO_SHA256",
    "family_kind",
    "parse_event_json",
    "parse_method_card_markdown",
    "validate_competition_contract_markdown",
]
