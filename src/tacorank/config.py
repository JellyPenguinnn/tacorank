"""Frozen run configuration and human-contract verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from .schemas import (
    MetricSet,
    NonEmptyStr,
    ResearchCampaign,
    SHA256_RE,
    StrictModel,
    _validate_runtime_mapping,
    normalize_relative_path,
)


class ContractError(RuntimeError):
    pass


DEFAULT_TARGET_INTERFACE_EXCERPTS = {
    "solution/official_fm.py": (
        "Editable adaptation of the frozen official five-field NumPy FM. "
        "Preserve controller-owned split selection, evaluation, and submission "
        "boundaries. Research trials may change model mathematics only when the "
        "ExperimentSpec authorizes this file."
    ),
    "solution/losses.py": (
        "Candidate-owned objective functions. Losses may consume training labels "
        "only and must not select checkpoints or read validation/test labels."
    ),
    "solution/features.py": (
        "Candidate-owned feature boundary. Fit on training data only; scoring "
        "rows never contain long_view and past-only features must not leak future "
        "interactions."
    ),
    "solution/model.py": (
        "Candidate-owned model components. Preserve deterministic seeds, finite "
        "unconstrained ranking scores, and non-zero trainable gradients."
    ),
    "solution/train.py": (
        "Candidate-owned training orchestration. Read train.csv only, respect "
        "fidelity and seed, and never select using validation or test labels."
    ),
    "solution/inference.py": (
        "Candidate-owned scoring helpers. Preserve row order and write exactly "
        "one finite output CSV exclusively."
    ),
    "solution/experiment_config.py": (
        "Edit only the scalar values in CONFIG. The stable scaffold supports "
        "formulation=official_fm|pointwise|bpr|listwise|temporal_history and validates: "
        "embedding_dim integer 2..32, learning_rate 1e-5..0.2, epochs 1..40, "
        "negative_count integer 1..16, l2 0..0.1, residual_scale 0..0.5, "
        "max_train_rows integer 1000..1141112, history_decay_days 1..180, "
        "history_shrinkage 0..1000, and listwise_strategy=full_observed. Set "
        "CONFIG family to the ExperimentSpec "
        "family and copy every approved variant_parameters value into its "
        "matching CONFIG key; do not edit executable code."
    ),
    "solution/research_scaffold.py": (
        "Implementation trials may add or repair one reviewed research capability. "
        "Preserve the candidate entrypoint, frozen FM parent, deterministic seeds, "
        "finite outputs, and training-diagnostics execution receipt. Configuration "
        "trials must not edit this file."
    ),
    "solution/candidate.py": (
        "Implementation trials may change the reviewed solution.candidate:run "
        "entrypoint only when the selected unverified method card authorizes it. "
        "Preserve the candidate output and data-boundary contracts."
    ),
}


# Production contains both the stable typed campaign scaffold and the modular
# implementation surface merged from main. Development fixtures may provide
# only the smaller default surface, so keep this expansion production-only.
PRODUCTION_TARGET_INTERFACE_EXCERPTS = {
    **DEFAULT_TARGET_INTERFACE_EXCERPTS,
    "solution/candidate.py": (
        "Required candidate entrypoint: def run(invocation: PipelineInvocation) "
        "-> None. Read only invocation.input_root and write exactly "
        "invocation.output_path as row_id,user_id,video_id,score CSV. Preserve "
        "the authenticated, row-aligned, unconstrained real-valued FM parent; "
        "never sigmoid, clip, normalize, or rescale it. Training dates strictly "
        "precede score dates, score rows never expose long_view, and duration_ms "
        "is video duration rather than watch time. Preserve deterministic finite "
        "scores, row order, duplicates, and exact parent fallback."
    ),
    "solution/features.py": (
        "Candidate-owned feature boundary. Fit on training data only. Any history "
        "aggregate must be deterministic and use only interactions earlier than "
        "the row it scores; scoring rows never contain long_view."
    ),
    "solution/model.py": (
        "Candidate-owned model components. Preserve deterministic seeds, finite "
        "unconstrained ranking scores, and non-zero trainable gradients."
    ),
    "solution/official_fm.py": DEFAULT_TARGET_INTERFACE_EXCERPTS[
        "solution/official_fm.py"
    ],
    "solution/losses.py": DEFAULT_TARGET_INTERFACE_EXCERPTS["solution/losses.py"],
    "solution/train.py": (
        "Candidate-owned training orchestration. Read train.csv only, respect "
        "fidelity and seed, and never early-stop or select using public-validation "
        "or score-population labels."
    ),
    "solution/inference.py": (
        "Candidate-owned scoring helpers. Add only the approved bounded residual "
        "on the original FM scale, preserve exact parent fallback, and create one "
        "ordered finite output CSV exclusively."
    ),
}


class RunConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: NonEmptyStr
    repository_root: Path
    contract_path: str = "contract/COMPETITION.md"
    protected_paths_path: str = "PROTECTED_PATHS.md"
    artifact_roots: List[str] = Field(default_factory=lambda: ["artifacts", "runs"])
    command_ids: List[NonEmptyStr]
    metric_names: List[NonEmptyStr]
    primary_metric_name: NonEmptyStr
    data_manifest_sha256: str
    evaluator_sha256: str
    baseline_commit_sha: NonEmptyStr
    max_experiments: int = Field(default=50, gt=0)
    run_mode: Literal["discovery", "submission"] = "submission"
    parallel_directions: int = Field(default=1, gt=0, le=7)
    synthesize_parallel_improvements: bool = True
    wall_time_limit_seconds: int = Field(default=21_600, gt=0)
    token_limit: Optional[int] = Field(default=None, gt=0)
    gpu_seconds_limit: Optional[int] = Field(default=None, gt=0)
    convergence_epsilon: float = Field(default=0.002, ge=0)
    convergence_patience: int = Field(default=3, gt=0)
    max_repairs_per_experiment: int = Field(default=2, ge=0, le=2)
    allowed_runtime_adjustments: Dict[str, Any] = Field(default_factory=dict)
    timeout_profiles: Dict[str, int] = Field(
        default_factory=lambda: {"standard": 600, "extended": 900}
    )
    max_confirmation_attempts: int = Field(default=2, ge=0)
    seed_schedule: List[int]
    context_token_limit: int = Field(default=6_000, gt=0)
    synthesis_context_token_limit: int = Field(default=16_000, gt=0)
    adapter_mode: Literal["live"] = "live"
    live_adapter_config_sha256: Optional[str] = None
    editable_roots: List[str] = Field(default_factory=lambda: ["solution"])
    allowed_research_families: List[NonEmptyStr] = Field(
        default_factory=lambda: [
            "objective",
            "temporal_history",
            "multitask",
            "duration_bias",
            "features",
            "model",
            "sampling",
            "ensemble",
            "evaluation",
            "other",
        ]
    )
    allowed_research_data: List[NonEmptyStr] = Field(
        default_factory=lambda: [
            "train_interactions",
            "public_validation",
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "date",
            "duration_ms",
            "long_view",
            "verified_predictions",
        ]
    )
    research_capabilities: List[NonEmptyStr] = Field(default_factory=list)
    active_research_prohibitions: List[NonEmptyStr] = Field(default_factory=list)
    research_campaign: Optional[ResearchCampaign] = None
    prediction_change_no_op_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    max_single_score_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    target_interface_excerpts: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_TARGET_INTERFACE_EXCERPTS)
    )
    coding_step_limit: int = Field(default=64, gt=0)
    # ``None`` explicitly disables TacoRank's cumulative coding-trajectory
    # token gate. Provider/model request limits remain independently enforced.
    coding_token_limit: Optional[int] = Field(default=None, gt=0)
    coding_wall_time_limit_seconds: int = Field(default=1800, gt=0)
    research_provider: Literal["deepseek"] = "deepseek"
    deepseek_model: NonEmptyStr = "deepseek-v4-flash"
    deepseek_base_url: NonEmptyStr = "https://api.deepseek.com"
    deepseek_api_key_env: NonEmptyStr = "DEEPSEEK_API_KEY"
    deepseek_timeout_seconds: int = Field(default=300, gt=0, le=600)
    research_planning_max_attempts: int = Field(default=2, gt=0, le=3)
    research_planning_retry_backoff_seconds: float = Field(
        default=1.0, ge=0.0, le=30.0
    )
    deepseek_max_output_tokens: int = Field(default=8_192, gt=0)
    deepseek_thinking_enabled: bool = True
    deepseek_reasoning_effort: Literal["low", "high", "max"] = "high"
    # Historical run configs omit these fields and therefore keep the old
    # offline-only planner behavior. New setup-live deployments explicitly
    # enable the hash-bound advisory paper bank.
    literature_research_enabled: bool = False
    literature_provider: Literal["openalex", "paper_bank"] = "openalex"
    literature_base_url: NonEmptyStr = "https://api.openalex.org"
    literature_timeout_seconds: int = Field(default=20, gt=0, le=120)
    literature_max_papers: int = Field(default=3, gt=0, le=8)
    literature_min_citation_count: int = Field(default=5, ge=0)
    literature_bank_path: str = "research/paper_bank.json"
    literature_bank_sha256: Optional[str] = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        from .schemas import _validate_id

        return _validate_id(value, "run_id")

    @field_validator(
        "contract_path", "protected_paths_path", "literature_bank_path"
    )
    @classmethod
    def validate_relative_config_paths(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("artifact_roots")
    @classmethod
    def validate_artifact_roots(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("artifact_roots must not be empty")
        roots = [normalize_relative_path(value) for value in values]
        if len(roots) != len(set(roots)):
            raise ValueError("artifact_roots must be unique")
        return roots

    @field_validator("editable_roots")
    @classmethod
    def validate_editable_roots(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("editable_roots must not be empty")
        roots = [normalize_relative_path(value) for value in values]
        if len(roots) != len(set(roots)):
            raise ValueError("editable_roots must be unique")
        return roots

    @field_validator(
        "allowed_research_families",
        "allowed_research_data",
        "research_capabilities",
        "active_research_prohibitions",
    )
    @classmethod
    def validate_unique_research_values(cls, values: List[str]) -> List[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("research contract values must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("research contract values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_research_contract(self) -> "RunConfig":
        if not self.allowed_research_families:
            raise ValueError("allowed_research_families must not be empty")
        if not self.allowed_research_data:
            raise ValueError("allowed_research_data must not be empty")
        if self.research_campaign is not None:
            campaign = self.research_campaign
            unknown = set(campaign.family_order) - set(self.allowed_research_families)
            if unknown:
                raise ValueError("campaign contains a research family not allowed by the run")
            if campaign.experiment_budget > self.max_experiments:
                raise ValueError("campaign experiment budget exceeds max_experiments")
        if (
            self.literature_research_enabled
            and self.literature_provider == "paper_bank"
            and self.literature_bank_sha256 is None
        ):
            raise ValueError(
                "enabled paper_bank literature requires literature_bank_sha256"
            )
        return self

    @field_validator("target_interface_excerpts")
    @classmethod
    def validate_target_interfaces(cls, values: Dict[str, str]) -> Dict[str, str]:
        if not values:
            raise ValueError("target_interface_excerpts must not be empty")
        normalized: Dict[str, str] = {}
        for path, excerpt in values.items():
            target = normalize_relative_path(path)
            if not excerpt.strip():
                raise ValueError("target interface excerpts must not be blank")
            if target in normalized:
                raise ValueError("target interface paths must be unique")
            normalized[target] = excerpt.strip()
        return normalized

    @field_validator("data_manifest_sha256", "evaluator_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("hashes must be lowercase sha256")
        return value

    @field_validator("live_adapter_config_sha256")
    @classmethod
    def validate_optional_live_config_hash(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("live_adapter_config_sha256 must be lowercase sha256")
        return value

    @field_validator("literature_bank_sha256")
    @classmethod
    def validate_optional_paper_bank_hash(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("literature_bank_sha256 must be lowercase sha256")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def validate_deepseek_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("deepseek_base_url must be a credential-free HTTPS origin")
        return value

    @field_validator("literature_base_url")
    @classmethod
    def validate_literature_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.openalex.org"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "literature_base_url must be the credential-free OpenAlex "
                "HTTPS origin"
            )
        return value

    @field_validator("deepseek_api_key_env")
    @classmethod
    def validate_deepseek_api_key_env(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", value):
            raise ValueError("deepseek_api_key_env must be an uppercase environment variable")
        return value

    @model_validator(mode="after")
    def validate_contract_fields(self) -> "RunConfig":
        if not self.metric_names:
            raise ValueError("metric_names must not be empty")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be unique")
        if not self.command_ids or len(set(self.command_ids)) != len(self.command_ids):
            raise ValueError("command_ids must be non-empty and unique")
        if not self.seed_schedule:
            raise ValueError("seed_schedule must not be empty")
        if len(self.seed_schedule) != len(set(self.seed_schedule)):
            raise ValueError("seed_schedule must contain distinct seeds")
        if len(self.seed_schedule) < self.max_confirmation_attempts + 1:
            raise ValueError(
                "seed_schedule must contain one initial seed plus every "
                "confirmation seed"
            )
        _validate_runtime_mapping(self.allowed_runtime_adjustments)
        if any(
            not isinstance(name, str) or not name.strip() or seconds <= 0
            for name, seconds in self.timeout_profiles.items()
        ):
            raise ValueError("timeout profiles must have positive named durations")
        return self

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = cls.model_validate(raw)
        root = config.repository_root
        if not root.is_absolute():
            config.repository_root = (path.parent / root).resolve()
        else:
            config.repository_root = root.resolve()
        return config

    def canonical_dict(self) -> dict:
        data = self.model_dump(mode="json")
        data["repository_root"] = str(self.repository_root.resolve())
        return data

    def validate_metric_set(self, metric_set: MetricSet) -> None:
        if set(metric_set.metrics) != set(self.metric_names):
            raise ContractError("metric set does not exactly match frozen metric names")
        if metric_set.primary_metric_name != self.primary_metric_name:
            raise ContractError("primary metric does not match frozen contract")


class VerifiedContract(StrictModel):
    contract_sha256: str
    protected_paths_sha256: str
    config_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contract_manifest_values(contract_text: str, label: str) -> List[str]:
    matches = re.findall(
        r"(?im)^\s*%s\s*:\s*([^\r\n]+?)\s*$" % re.escape(label),
        contract_text,
    )
    if len(matches) != 1:
        raise ContractError("contract must contain exactly one '%s:' line" % label)
    values = [item.strip().strip("`").strip() for item in matches[0].split(",")]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ContractError("contract %s must be a unique comma-separated list" % label)
    return values


def verify_contract(config: RunConfig) -> VerifiedContract:
    """Verify that humans supplied a resolved, explicitly frozen contract.

    The harness never edits these files.  It deliberately uses a small, visible
    marker instead of trying to infer intent from conflicting prose.
    """

    contract_path = config.repository_root / config.contract_path
    protected_path = config.repository_root / config.protected_paths_path
    if not contract_path.is_file() or not protected_path.is_file():
        raise ContractError("contract and protected-path files must exist")
    contract_bytes = contract_path.read_bytes()
    protected_bytes = protected_path.read_bytes()
    if not contract_bytes.strip():
        raise ContractError("contract is empty; a human must resolve and freeze it")
    if not protected_bytes.strip():
        raise ContractError("protected path manifest is empty")

    contract_text = contract_bytes.decode("utf-8", errors="strict")
    upper = contract_text.upper()
    if not re.search(r"(?im)^\s*Contract status\s*:\s*FROZEN\s*$", contract_text):
        raise ContractError("contract must contain the exact line 'Contract status: FROZEN'")
    unresolved_markers = ("TBD", "TODO", "UNRESOLVED", "<<<<<<<", ">>>>>>>")
    present = [marker for marker in unresolved_markers if marker in upper]
    if present:
        raise ContractError("contract contains unresolved markers: %s" % ", ".join(present))
    for metric_name in config.metric_names:
        if metric_name.lower() not in contract_text.lower():
            raise ContractError("frozen metric %r is not named in the contract" % metric_name)
    if config.primary_metric_name.lower() not in contract_text.lower():
        raise ContractError("frozen primary metric is not named in the contract")
    allowed_commands = _contract_manifest_values(contract_text, "Allowed command IDs")
    if allowed_commands != config.command_ids:
        raise ContractError("configured command_ids do not match the frozen contract")
    artifact_roots = _contract_manifest_values(contract_text, "Artifact roots")
    if artifact_roots != config.artifact_roots:
        raise ContractError("configured artifact_roots do not match the frozen contract")

    config_bytes = json.dumps(
        config.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return VerifiedContract(
        contract_sha256=_sha256(contract_bytes),
        protected_paths_sha256=_sha256(protected_bytes),
        config_sha256=_sha256(config_bytes),
    )
