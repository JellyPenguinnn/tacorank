"""Pure derived state models; never persisted as mutable authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..accounting import ResourceTotals
from ..schemas import Fidelity, MetricSet, TrustAssessment


class RunStatus(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    PATCH_READY = "patch_ready"
    READY_TO_RUN = "ready_to_run"
    RUNNING = "running"
    OUTPUT_READY = "output_ready"
    OUTPUT_VERIFIED = "output_verified"
    EVALUATED = "evaluated"
    RECOVERING = "recovering"
    NO_OP = "no_op"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PRUNED = "pruned"
    INVALID = "invalid"


@dataclass
class ExperimentNode:
    experiment_id: str
    parent_experiment_id: Optional[str]
    hypothesis: str
    family: str
    base_commit_sha: str
    latest_commit_sha: Optional[str] = None
    attempt_count: int = 0
    repair_count: int = 0
    same_commit_retry_count: int = 0
    confirmation_count: int = 0
    highest_fidelity: Optional[Fidelity] = None
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    metric_set: Optional[MetricSet] = None
    trust: Optional[TrustAssessment] = None
    best_eligible: bool = False
    terminal_event_id: Optional[str] = None
    duplicate_key: str = ""
    last_error_fingerprint: Optional[str] = None
    evidence_event_ids: List[str] = field(default_factory=list)


@dataclass
class LessonView:
    lesson_id: str
    status: str
    summary: str
    tags: List[str]
    source_event_id: str


@dataclass
class RunState:
    run_id: Optional[str] = None
    status: RunStatus = RunStatus.INITIALIZING
    phase: str = "not_started"
    active_experiment_id: Optional[str] = None
    active_attempt: Optional[int] = None
    active_fidelity: Optional[Fidelity] = None
    best_experiment_id: Optional[str] = None
    best_commit_sha: Optional[str] = None
    best_primary_score: Optional[float] = None
    baseline_primary_score: Optional[float] = None
    experiments_proposed: int = 0
    full_evaluations_completed: int = 0
    public_validation_queries: int = 0
    consecutive_non_improving_full_evaluations: int = 0
    convergence_epsilon: float = 0.002
    convergence_patience: int = 3
    max_experiments: int = 0
    parallel_directions: int = 1
    research_agent_mode: str = "legacy"
    research_tool_step_limit: int = 4
    research_literature_max_queries: int = 2
    literature_required: bool = False
    parallel_schedule: List[int] = field(default_factory=list)
    synthesize_parallel_improvements: bool = True
    wall_time_limit_seconds: int = 0
    token_limit: Optional[int] = None
    gpu_seconds_limit: Optional[int] = None
    max_repairs_per_experiment: int = 2
    max_confirmation_attempts: int = 2
    seed_schedule: List[int] = field(default_factory=list)
    experiments: Dict[str, ExperimentNode] = field(default_factory=dict)
    lessons: Dict[str, LessonView] = field(default_factory=dict)
    resource_totals: ResourceTotals = field(default_factory=ResourceTotals)
    manual_intervention_count: int = 0
    started_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    elapsed_wall_time_seconds: float = 0.0
    last_event_id: Optional[str] = None
    last_event_hash: Optional[str] = None
    stop_reason_code: Optional[str] = None
    final_experiment_id: Optional[str] = None

    @property
    def remaining_experiments(self) -> int:
        return max(0, self.max_experiments - self.experiments_proposed)

    @property
    def current_experiment(self) -> Optional[ExperimentNode]:
        if self.active_experiment_id is None:
            return None
        return self.experiments.get(self.active_experiment_id)
