"""Bounded coding-worker primitives for TacoRank."""

from .output_parser import (
    ParsedTrajectory,
    TokenUsage,
    TrajectoryParseError,
    parse_trajectory_bytes,
    parse_trajectory_file,
)
from .prompts import (
    PromptContractError,
    build_coding_prompt,
    build_repair_prompt,
    build_solution_revision_prompt,
)
from .solution_verifier import (
    AcceptingSolutionVerifier,
    DeepSeekSolutionVerifier,
    SolutionFinding,
    SolutionVerificationResult,
    SolutionVerifier,
    SolutionVerifierError,
)
from .redaction import RedactionError, SecretRedactor
from .trae_adapter import (
    CandidateIdentity,
    CandidateIdentityResolver,
    CodingWorkerError,
    FakeCodingWorker,
    SchemaFactories,
    SchemaIntegrationError,
    TRAE_DEEPSEEK_REASONING_MARKER,
    TRAE_DEEPSEEK_TOOL_JSON_MARKER,
    TRAE_DOCKER_EDIT_TOOL_MARKER,
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
    "TRAE_DEEPSEEK_REASONING_MARKER",
    "TRAE_DEEPSEEK_TOOL_JSON_MARKER",
    "TRAE_DOCKER_EDIT_TOOL_MARKER",
    "TraeCodingWorker",
    "TraeConfig",
    "TrajectoryParseError",
    "build_coding_prompt",
    "build_repair_prompt",
    "build_solution_revision_prompt",
    "AcceptingSolutionVerifier",
    "DeepSeekSolutionVerifier",
    "SolutionFinding",
    "SolutionVerificationResult",
    "SolutionVerifier",
    "SolutionVerifierError",
    "hash_trae_runtime_package",
    "parse_trajectory_bytes",
    "parse_trajectory_file",
]
