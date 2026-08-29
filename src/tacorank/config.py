"""Frozen run configuration and human-contract verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .schemas import MetricSet, NonEmptyStr, SHA256_RE, StrictModel, normalize_relative_path


class ContractError(RuntimeError):
    pass


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
    wall_time_limit_seconds: int = Field(default=21_600, gt=0)
    token_limit: Optional[int] = Field(default=None, gt=0)
    gpu_seconds_limit: Optional[int] = Field(default=None, gt=0)
    convergence_epsilon: float = Field(default=0.002, ge=0)
    convergence_patience: int = Field(default=3, gt=0)
    max_repairs_per_experiment: int = Field(default=2, ge=0)
    max_confirmation_attempts: int = Field(default=2, ge=0)
    seed_schedule: List[int]
    context_token_limit: int = Field(default=6_000, gt=0)
    adapter_mode: Literal["fake"] = "fake"
    baseline_metrics: Optional[Dict[str, float]] = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        from .schemas import _validate_id

        return _validate_id(value, "run_id")

    @field_validator("contract_path", "protected_paths_path")
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

    @field_validator("data_manifest_sha256", "evaluator_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("hashes must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def validate_contract_fields(self) -> "RunConfig":
        if self.primary_metric_name not in self.metric_names:
            raise ValueError("primary_metric_name must be listed in metric_names")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be unique")
        if not self.command_ids or len(set(self.command_ids)) != len(self.command_ids):
            raise ValueError("command_ids must be non-empty and unique")
        if not self.seed_schedule:
            raise ValueError("seed_schedule must not be empty")
        if self.baseline_metrics is not None:
            if set(self.baseline_metrics) != set(self.metric_names):
                raise ValueError("baseline_metrics must exactly match metric_names")
            if self.primary_metric_name not in self.baseline_metrics:
                raise ValueError("baseline_metrics is missing the primary metric")
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
    if "CONTRACT STATUS: FROZEN" not in upper:
        raise ContractError("contract must contain 'Contract status: FROZEN'")
    unresolved_markers = ("TBD", "TODO", "UNRESOLVED", "<<<<<<<", ">>>>>>>")
    present = [marker for marker in unresolved_markers if marker in upper]
    if present:
        raise ContractError("contract contains unresolved markers: %s" % ", ".join(present))
    for metric_name in config.metric_names:
        if metric_name.lower() not in contract_text.lower():
            raise ContractError("frozen metric %r is not named in the contract" % metric_name)
    if config.primary_metric_name.lower() not in contract_text.lower():
        raise ContractError("frozen primary metric is not named in the contract")

    config_bytes = json.dumps(
        config.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return VerifiedContract(
        contract_sha256=_sha256(contract_bytes),
        protected_paths_sha256=_sha256(protected_bytes),
        config_sha256=_sha256(config_bytes),
    )
