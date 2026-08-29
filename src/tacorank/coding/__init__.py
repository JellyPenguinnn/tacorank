"""Bounded coding-worker primitives for TacoRank."""

from .output_parser import (
    ParsedTrajectory,
    TokenUsage,
    TrajectoryParseError,
    parse_trajectory_bytes,
    parse_trajectory_file,
)
from .prompts import PromptContractError, build_coding_prompt, build_repair_prompt
from .redaction import RedactionError, SecretRedactor
from .trae_adapter import (
    CandidateIdentity,
    CandidateIdentityResolver,
    CodingWorkerError,
    FakeCodingWorker,
    SchemaFactories,
    SchemaIntegrationError,
    TraeCodingWorker,
    TraeConfig,
    hash_trae_runtime_package,
)

__all__ = [
    "CandidateIdentity",
    "CandidateIdentityResolver",
    "CodingWorkerError",
    "FakeCodingWorker",
    "ParsedTrajectory",
    "PromptContractError",
    "RedactionError",
    "SchemaFactories",
    "SchemaIntegrationError",
    "SecretRedactor",
    "TokenUsage",
    "TraeCodingWorker",
    "TraeConfig",
    "TrajectoryParseError",
    "build_coding_prompt",
    "build_repair_prompt",
    "hash_trae_runtime_package",
    "parse_trajectory_bytes",
    "parse_trajectory_file",
]
