"""Deterministic, bounded prompts for initial and repair coding tasks."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from tacorank.git.patches import validate_relative_path

from .redaction import SecretRedactor


MAX_PROMPT_BYTES = 512 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GATE_A_CHECK_NAMES = frozenset(
    {
        "diff_parse",
        "changed_file_match",
        "editable_path",
        "protected_path",
        "path_escape",
        "contract_hash",
        "syntax_import",
        "interface_contract",
        "command_policy",
        "data_boundary",
        "network_policy",
        "secret_scan",
        "dependency_policy",
        "smoke_test",
    }
)


# The bounded code-recovery actions that share the repair prompt contract.
# Kept in sync with the orchestrator's _CODE_RECOVERY_ACTIONS.
_REPAIR_PROMPT_ACTIONS = frozenset({"trae_repair", "restart_from_trusted_parent"})


class PromptContractError(ValueError):
    """Raised before invocation when a coding context violates its contract."""


class CoderContextLike(Protocol):
    context_id: str
    run_id: str
    experiment_id: str
    contract_sha256: str
    experiment_spec: Any
    parent_commit_sha: str
    target_interface_excerpts: Any
    editable_roots: Sequence[str]
    protected_paths: Sequence[str]
    allowed_command_ids: Sequence[str]
    selected_method_cards: Sequence[Any]
    active_lessons: Sequence[Any]
    coding_invariants: Sequence[str]
    prior_result_summaries: Sequence[Any]
    step_limit: int
    token_limit: Optional[int]
    wall_time_limit_seconds: int
    context_artifact: Any


class RecoveryContextLike(Protocol):
    context_id: str
    run_id: str
    experiment_id: str
    repair_attempt: int
    original_experiment_spec: Any
    current_patch_commit_sha: str
    accepted_patch_receipt_id: Optional[str]
    failure_class: str
    error_fingerprint: str
    error_summary: str
    relevant_trace_tail: str
    failed_checks: Sequence[Any]
    previous_repair_fingerprints: Sequence[str]
    recovery_instructions: str
    remaining_repair_budget: int
    editable_roots: Sequence[str]
    protected_paths: Sequence[str]


def build_coding_prompt(
    context: CoderContextLike,
    spec: Any,
    *,
    redactor: Optional[SecretRedactor] = None,
) -> str:
    """Build the initial Trae task for exactly one approved experiment spec."""

    safe_redactor = redactor or SecretRedactor()
    context_spec = _required_attribute(context, "experiment_spec")
    spec_document = _json_document(spec, "experiment_spec")
    if _canonical_json(context_spec) != _canonical_json(spec_document):
        raise PromptContractError("context experiment_spec differs from the approved spec")

    run_id = _required_text(context, "run_id")
    experiment_id = _required_text(context, "experiment_id")
    if spec_document.get("run_id") != run_id or spec_document.get("experiment_id") != experiment_id:
        raise PromptContractError("spec identity does not match coding context")
    parent = _validated_object_id(_required_text(context, "parent_commit_sha"))
    if spec_document.get("parent_commit_sha") != parent:
        raise PromptContractError("spec parent commit does not match coding context")
    contract_sha = _validated_sha256(_required_text(context, "contract_sha256"))
    editable_roots = _validated_paths(_required_attribute(context, "editable_roots"), "editable_roots")
    protected_paths = _validated_paths(
        _required_attribute(context, "protected_paths"), "protected_paths"
    )
    commands = _validated_commands(_required_attribute(context, "allowed_command_ids"))
    step_limit = _positive_int(context, "step_limit")
    token_limit = _optional_positive_int(context, "token_limit")
    wall_limit = _positive_int(context, "wall_time_limit_seconds")
    target_files = _validated_paths(spec_document.get("target_files"), "target_files")
    coding_invariants = _json_value(getattr(context, "coding_invariants", ()))
    prior_result_summaries = _json_value(
        getattr(context, "prior_result_summaries", ())
    )
    owner_retry_summary = getattr(context, "owner_retry_error_summary", None)
    owner_retry_instructions = getattr(context, "owner_retry_instructions", None)
    if (owner_retry_summary is None) != (owner_retry_instructions is None):
        raise PromptContractError(
            "owner retry requires both an error summary and instructions"
        )

    sections = [
        "# TacoRank bounded coding task",
        "",
        "Implement exactly the approved ExperimentSpec below and return a code patch plus a concise explanation.",
        "Do not choose or reinterpret the hypothesis. Do not run full-data training, compute official metrics, access protected labels, modify memory, or declare research success.",
        "Do not use network access, install packages, or run commands other than the symbolic lightweight capabilities listed below.",
        "",
        "## Immutable identity",
        _bullet("run_id", run_id),
        _bullet("experiment_id", experiment_id),
        _bullet("context_id", _required_text(context, "context_id")),
        _bullet("parent_commit_sha", parent),
        _bullet("contract_sha256", contract_sha),
        "",
        "## Context provenance",
        _json_block(_json_value(_required_attribute(context, "context_artifact"))),
        "",
        "## Hard bounds",
        _bullet("max_steps", step_limit),
        (
            _bullet("max_provider_tokens", token_limit)
            if token_limit is not None
            else _json_bullet("max_provider_tokens", None)
        ),
        "A null max_provider_tokens value means TacoRank does not impose a cumulative trajectory token limit.",
        _bullet("wall_time_limit_seconds", wall_limit),
        "",
        "## File and tool policy",
        _json_block(
            {
                "editable_roots": editable_roots,
                "protected_paths": protected_paths,
                "allowed_command_ids": commands,
            }
        ),
        "Only edit paths under editable_roots. Protected paths always win over editable roots.",
        "Dependency files may be inspected but not changed unless the ExperimentSpec explicitly names a reviewed dependency change.",
        "",
        "## Tool-use discipline",
        _json_block({"authoritative_target_files": target_files}),
        "Begin by viewing the authoritative target files directly; do not list the repository root or survey unrelated directories.",
        "Modify only authoritative_target_files. Do not add ad-hoc smoke, test, helper, or alternate entrypoint files unless each path is explicitly present in authoritative_target_files.",
        "The interface excerpts and method cards below are the supplied integration context. Inspect one non-target file only when a concrete missing symbol or schema blocks the edit.",
        "The production entrypoint is loaded as solution.candidate:run. Prefer one self-contained candidate.py; use sibling imports only when the approved target files and interface explicitly authorize them.",
        "When the interface supplies setup-verified FM scores, preserve them as the parent and implement the approved mechanism as a bounded residual unless the ExperimentSpec explicitly requires replacement or a selected method card is tagged replacement_capable; for those, output the new model's own score directly and use the FM score as an input feature or fallback for unseen categories.",
        "The supplied FM scores are unconstrained real-valued ranking scores, not probabilities. Never sigmoid, clip to [0,1], normalize, or rescale the FM parent or a parent-plus-residual result. Bound only the residual on the parent's original scale.",
        "Prior-result summaries are mandatory implementation constraints. Use them to avoid repeating score collapse, excessive parent divergence, missing personalization, or loss of within-user rankability.",
        "Before task_done, review the edited score path for full/representative training coverage, non-zero trainable gradients, user-conditioned score variation, correct feature semantics, deterministic seeds, and finite fallback scores.",
        "The symbolic allowed_command_ids are controller-owned post-patch checks, not shell tools available in this coding action. Do not search for or invoke them.",
        "Use the next editing-capable tool call after the target view to make the smallest coherent edit. Once the required edit and one bounded recheck are complete, then call task_done immediately; do not spend remaining steps browsing or making unrelated improvements.",
        "",
        "## Required target interfaces",
        _json_block(_json_value(_required_attribute(context, "target_interface_excerpts"))),
        "",
        "## Approved ExperimentSpec (exact)",
        _json_block(spec_document),
        "",
        "## Selected method cards",
        _json_block(_json_value(_required_attribute(context, "selected_method_cards"))),
        "",
        "## Score-scale and implementation invariants",
        _json_block(coding_invariants),
        "",
        "## Approved prior-result constraints",
        _json_block(prior_result_summaries),
        "",
    ]
    if owner_retry_summary is not None:
        sections.extend(
            [
                "## Bounded owner retry",
                _json_block(
                    {
                        "exact_error_summary": str(owner_retry_summary),
                        "proposed_correction": str(owner_retry_instructions),
                    }
                ),
                "This is a retry of the same approved coding assignment. Correct the reported protocol/tool failure without changing the hypothesis, mechanism, target files, or scope. Treat proposed_correction as a bounded diagnostic, verify it against exact_error_summary, and do not invent an unrelated code change.",
                "",
            ]
        )
    sections.extend(
        [
            "## Applicable lessons",
            _json_block(_json_value(_required_attribute(context, "active_lessons"))),
            "",
            "## Completion contract",
            "Make the smallest coherent implementation of this exact hypothesis. TacoRank runs Gate A and controller-owned checks after this action. Finish with a non-empty patch and a concise account of files changed; do not claim checks that this tool session could not run.",
        ]
    )
    return _finalize("\n".join(sections), safe_redactor)


def build_repair_prompt(
    context: RecoveryContextLike,
    decision: Any,
    *,
    step_limit: int,
    token_limit: Optional[int],
    wall_time_limit_seconds: int,
    allowed_command_ids: Sequence[str],
    redactor: Optional[SecretRedactor] = None,
) -> str:
    """Build a repair task that preserves the original research hypothesis."""

    safe_redactor = redactor or SecretRedactor()
    spec_document = _json_document(
        _required_attribute(context, "original_experiment_spec"),
        "original_experiment_spec",
    )
    run_id = _required_text(context, "run_id")
    experiment_id = _required_text(context, "experiment_id")
    if spec_document.get("run_id") != run_id or spec_document.get("experiment_id") != experiment_id:
        raise PromptContractError("original spec identity does not match recovery context")
    repair_attempt = _nonnegative_int(context, "repair_attempt")
    remaining_budget = _nonnegative_int(context, "remaining_repair_budget")
    if remaining_budget < 1:
        raise PromptContractError("repair budget is exhausted")
    current_commit = _validated_object_id(_required_text(context, "current_patch_commit_sha"))
    failed_checks = _json_value(_required_attribute(context, "failed_checks"))
    if not isinstance(failed_checks, list):
        raise PromptContractError("failed_checks must serialize to a sequence")
    gate_a_rejection = _contains_gate_a_failure(failed_checks)
    receipt_id = getattr(context, "accepted_patch_receipt_id", None)
    if gate_a_rejection:
        if receipt_id is not None:
            raise PromptContractError(
                "Gate-A rejection recovery cannot identify an accepted patch receipt"
            )
    elif not isinstance(receipt_id, str) or not receipt_id.strip():
        raise PromptContractError(
            "post-acceptance recovery requires accepted_patch_receipt_id"
        )
    limits = {
        "max_steps": _standalone_positive_int(step_limit, "step_limit"),
        "max_provider_tokens": _standalone_optional_positive_int(
            token_limit, "token_limit"
        ),
        "wall_time_limit_seconds": _standalone_positive_int(
            wall_time_limit_seconds, "wall_time_limit_seconds"
        ),
    }
    commands = _validated_commands(allowed_command_ids)
    editable_roots = _validated_paths(_required_attribute(context, "editable_roots"), "editable_roots")
    protected_paths = _validated_paths(
        _required_attribute(context, "protected_paths"), "protected_paths"
    )
    target_files = _validated_paths(spec_document.get("target_files"), "target_files")
    decision_document = _json_document(decision, "recovery_decision")
    # Both bounded code-recovery actions build their prompt here:
    # restart_from_trusted_parent recodes from the parent commit rather than
    # editing the rejected candidate, but it is the same bounded repair task
    # under the same hypothesis. Accepting only trae_repair made every
    # restart raise PromptContractError, which surfaced as
    # CODING_WORKER_FAILURE and abandoned the experiment.
    if decision_document.get("action") not in _REPAIR_PROMPT_ACTIONS:
        raise PromptContractError("recovery decision is not a code repair action")
    if decision_document.get("run_id", run_id) != run_id or decision_document.get(
        "experiment_id", experiment_id
    ) != experiment_id:
        raise PromptContractError("recovery decision identity does not match context")
    if decision_document.get("repair_attempt", repair_attempt) != repair_attempt:
        raise PromptContractError("recovery decision attempt does not match context")

    sections = [
        "# TacoRank bounded repair task",
        "",
        "Repair the current patch using only the supplied failure evidence. Preserve the original hypothesis and mechanism exactly; do not substitute a different experiment.",
        "Do not run full-data training, compute official metrics, access protected labels, modify memory, use network access, install packages, or declare research success.",
        "",
        "## Immutable identity",
        _bullet("run_id", run_id),
        _bullet("experiment_id", experiment_id),
        _bullet("context_id", _required_text(context, "context_id")),
        _bullet("repair_attempt", repair_attempt),
        _bullet("current_patch_commit_sha", current_commit),
        _json_bullet("accepted_patch_receipt_id", receipt_id),
        _bullet("remaining_repair_budget", remaining_budget),
        "",
        "## Hard bounds",
        _json_block({**limits, "allowed_command_ids": commands}),
        "A null max_provider_tokens value means TacoRank does not impose a cumulative trajectory token limit.",
        "",
        "## File policy",
        _json_block(
            {
                "editable_roots": editable_roots,
                "protected_paths": protected_paths,
                "authoritative_target_files": target_files,
            }
        ),
        "Only edit paths under editable_roots. Protected paths always win.",
        "Modify only authoritative_target_files; do not add ad-hoc smoke, test, helper, or alternate entrypoint files.",
        "",
        "## Original ExperimentSpec (must remain unchanged)",
        _json_block(spec_document),
        "",
        "## Validated recovery decision",
        _json_block(decision_document),
        "",
        "## Failure evidence",
        _json_block(
            {
                "failure_class": _required_text(context, "failure_class"),
                "error_fingerprint": _required_text(context, "error_fingerprint"),
                "error_summary": _required_text(context, "error_summary"),
                "relevant_trace_tail": _required_text(context, "relevant_trace_tail"),
                "failed_checks": failed_checks,
                "previous_repair_fingerprints": _json_value(
                    _required_attribute(context, "previous_repair_fingerprints")
                ),
                "recovery_instructions": _required_text(
                    context, "recovery_instructions"
                ),
            }
        ),
        "The error summary and trace are authoritative observations. The recovery instructions are a bounded proposed fix, not a proven diagnosis. Before editing, confirm that the proposed fix explains the supplied evidence; if it does not, state the narrower evidence-backed diagnosis and make only the smallest repair that satisfies the same success check.",
        "",
        "## Completion contract",
        "Make only the smallest repair justified by this evidence. After the edit and one bounded recheck, call task_done immediately. Finish with a non-empty repair patch plus a concise account of files changed and checks actually run.",
    ]
    return _finalize("\n".join(sections), safe_redactor)


def build_solution_revision_prompt(
    context: Any,
    experiment_spec: Any,
    verification: Any,
    *,
    review_attempt: int,
    max_review_attempts: int,
    step_limit: int,
    wall_time_limit_seconds: int,
    redactor: Optional[SecretRedactor] = None,
) -> str:
    """Build one verifier-grounded Trae revision without changing the hypothesis."""

    safe_redactor = redactor or SecretRedactor()
    spec_document = _json_document(experiment_spec, "experiment_spec")
    target_files = _validated_paths(spec_document.get("target_files"), "target_files")
    verification_document = _json_document(verification, "solution_verification")
    if verification_document.get("accepted") is not False:
        raise PromptContractError("solution revision requires a rejected verification")
    required = verification_document.get("required_changes")
    if not isinstance(required, list) or not required:
        raise PromptContractError("solution revision requires bounded required changes")
    if (
        isinstance(review_attempt, bool)
        or not isinstance(review_attempt, int)
        or review_attempt < 1
        or review_attempt >= max_review_attempts
    ):
        raise PromptContractError("solution review attempt is outside its bounded range")

    sections = [
        "# TacoRank bounded implementation-fidelity revision",
        "",
        "Revise the existing candidate only to address the grounded verifier findings below. Preserve the approved hypothesis, mechanism, data boundary, target stage, and target files exactly.",
        "The verifier is not an evaluator. Do not optimize imagined metrics, access labels, run full training, install packages, use network access, or broaden the experiment.",
        "",
        "## Hard bounds",
        _json_block(
            {
                "completed_review_attempt": review_attempt,
                "max_review_attempts": max_review_attempts,
                "max_steps": _standalone_positive_int(step_limit, "step_limit"),
                "wall_time_limit_seconds": _standalone_positive_int(
                    wall_time_limit_seconds, "wall_time_limit_seconds"
                ),
                "authoritative_target_files": target_files,
            }
        ),
        "Modify only authoritative_target_files. Do not add smoke, test, helper, or alternate entrypoint files.",
        "Begin with the named file and exact finding; do not survey the repository.",
        "",
        "## Approved ExperimentSpec (unchanged)",
        _json_block(spec_document),
        "",
        "## Grounded implementation review",
        _json_block(verification_document),
        "",
        "## Target interface",
        _json_block(_json_value(getattr(context, "target_interface_excerpts", {}))),
        "",
        "## Completion contract",
        "Make the smallest coherent correction for every required change. Once the edit and one bounded recheck are complete, call task_done immediately and report only what changed.",
    ]
    return _finalize("\n".join(sections), safe_redactor)


def prompt_sha256(prompt: str) -> str:
    """Return the hash recorded in adapter configuration evidence."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _finalize(prompt: str, redactor: SecretRedactor) -> str:
    redacted = redactor.redact(prompt)
    encoded = redacted.encode("utf-8")
    if len(encoded) > MAX_PROMPT_BYTES:
        raise PromptContractError(f"coding prompt exceeds {MAX_PROMPT_BYTES} bytes")
    if redactor.contains_known_secret(encoded):
        raise PromptContractError("known credential remained in coding prompt")
    return redacted


def _json_document(value: Any, name: str) -> Dict[str, Any]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise PromptContractError(f"{name} must serialize to an object")
    return normalized


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = _json_value(value.model_dump(mode="json"))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = _json_value(dataclasses.asdict(value))
    elif isinstance(value, enum.Enum):
        value = _json_value(value.value)
    elif isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PromptContractError("JSON object keys must be strings")
            normalized[key] = _json_value(item)
        value = normalized
    elif isinstance(value, (list, tuple)):
        value = [_json_value(item) for item in value]
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(serialized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PromptContractError("context value is not finite JSON data") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n```"


def _required_attribute(value: Any, field: str) -> Any:
    try:
        result = getattr(value, field)
    except AttributeError as exc:
        raise PromptContractError(f"coding context is missing {field}") from exc
    if result is None:
        raise PromptContractError(f"coding context {field} cannot be null")
    return result


def _required_text(value: Any, field: str) -> str:
    result = _required_attribute(value, field)
    if not isinstance(result, str) or not result.strip():
        raise PromptContractError(f"coding context {field} must be non-empty text")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = _required_attribute(value, field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise PromptContractError(f"coding context {field} must be a positive integer")
    return result


def _optional_positive_int(value: Any, field: str) -> Optional[int]:
    try:
        result = getattr(value, field)
    except AttributeError as exc:
        raise PromptContractError(f"coding context is missing {field}") from exc
    if result is None:
        return None
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise PromptContractError(
            f"coding context {field} must be null or a positive integer"
        )
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    result = _required_attribute(value, field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise PromptContractError(
            f"coding context {field} must be a non-negative integer"
        )
    return result


def _standalone_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PromptContractError(f"{field} must be a positive integer")
    return value


def _standalone_optional_positive_int(
    value: Any, field: str
) -> Optional[int]:
    if value is None:
        return None
    return _standalone_positive_int(value, field)


def _validated_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise PromptContractError("contract_sha256 must be lowercase 64-hex")
    return value


def _validated_object_id(value: str) -> str:
    if not _OBJECT_ID_RE.fullmatch(value):
        raise PromptContractError("commit SHA must be a full lowercase object ID")
    return value


def _validated_paths(value: Any, field: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PromptContractError(f"{field} must be a path sequence")
    try:
        paths = tuple(validate_relative_path(path) for path in value)
    except (TypeError, ValueError) as exc:
        raise PromptContractError(f"{field} contains an invalid path") from exc
    if not paths or paths != tuple(dict.fromkeys(paths)):
        raise PromptContractError(f"{field} must be non-empty and duplicate-free")
    return paths


def _validated_commands(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PromptContractError("allowed_command_ids must be a sequence")
    commands = tuple(value)
    if not commands:
        raise PromptContractError("allowed_command_ids cannot be empty")
    if any(not isinstance(item, str) or not _COMMAND_ID_RE.fullmatch(item) for item in commands):
        raise PromptContractError("allowed_command_ids contains an invalid identifier")
    if commands != tuple(dict.fromkeys(commands)):
        raise PromptContractError("allowed_command_ids contains duplicates")
    return commands


def _contains_gate_a_failure(failed_checks: Sequence[Any]) -> bool:
    """Recognize a pre-acceptance Gate-A failure without inventing a receipt."""

    for check in failed_checks:
        if isinstance(check, str):
            name = check
        elif isinstance(check, Mapping):
            name = None
            for field in ("name", "check_id", "check"):
                candidate = check.get(field)
                if isinstance(candidate, str):
                    name = candidate
                    break
        else:
            name = None
        if isinstance(name, str) and name in _GATE_A_CHECK_NAMES:
            return True
    return False


def _bullet(name: str, value: Any) -> str:
    return f"- {name}: `{value}`"


def _json_bullet(name: str, value: Any) -> str:
    return f"- {name}: {json.dumps(value, ensure_ascii=False, allow_nan=False)}"
