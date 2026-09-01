"""Focused, hypothesis-preserving self-debug instructions."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .classifier import FailureClassification


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _spec_summary(spec: Any) -> str:
    hypothesis = _get(spec, "hypothesis", "title", "summary", "description")
    mechanism = _get(spec, "expected_mechanism", "mechanism", "rationale")
    if hypothesis is None:
        hypothesis = str(spec) if spec is not None else "the accepted experiment specification"
    return f"{hypothesis}" + (f"; expected mechanism: {mechanism}" if mechanism else "")


def build_self_debug_instructions(
    classification: FailureClassification,
    context: Any,
    repair_attempt: int,
    remaining_budget: int,
    *,
    target_files: Iterable[str] = (),
    success_check: Optional[str] = None,
) -> str:
    """Build diagnostic instructions that constrain repair scope and objective drift."""
    spec = _get(context, "original_experiment_spec")
    commit = _get(context, "current_patch_commit_sha", default="unknown accepted commit")
    history = _get(context, "attempt_history", default=()) or ()
    if not target_files:
        target_files = _get(spec, "target_files", default=()) or ()
    targets = ", ".join(target_files) or "only the candidate files implicated by the failure"
    check = success_check or {
        "output_contract": "the existing Gate B output-contract checks",
        "no_op": "the existing wiring smoke test and a non-identical prediction check",
        "contract_error": "Gate A",
    }.get(classification.failure_class, "the previously failing smoke or execution check")
    previous = f" Previous failed outcomes: {len(history)}." if history else ""
    if classification.failure_class == "no_op":
        return (
            f"The original hypothesis remains: {_spec_summary(spec)}. Do not change "
            f"that hypothesis or its expected mechanism. The accepted candidate is "
            f"commit {commit}, and verified evaluation found unchanged predictions "
            f"({classification.evidence or 'NO_PREDICTION_CHANGE'}). Treat this as a "
            f"bounded implementation/wiring check, not proof that the research "
            f"hypothesis is false.{previous} Inspect only {targets}, starting with the "
            f"current diff and the path from the intended mechanism to the emitted "
            f"score. Do not survey setup files, packaging, documentation, or unrelated "
            f"modules. Before editing, emit `DIAGNOSIS:`, `REPAIR_PLAN:`, and "
            f"`VERIFICATION:` lines. Make the smallest justified edit, then call "
            f"task_done. You cannot execute code: your tools are the editor and "
            f"task_done, and TacoRank runs {check} for you after you finish. State "
            f"under `VERIFICATION:` what that run should show; do not keep editing "
            f"in search of confirmation you have no way to obtain, and do not spend "
            f"the step budget without calling task_done, which discards the attempt "
            f"entirely. Preserve the supplied contract and protected paths. This is "
            f"repair attempt {repair_attempt} of "
            f"{_get(context, 'max_repair_attempts', default=2)}; {remaining_budget} "
            f"repair attempt(s) remain after this decision."
        )
    return (
        f"The original hypothesis remains: {_spec_summary(spec)}. Do not change that hypothesis or its "
        f"expected mechanism. The exact accepted patch is commit {commit}. Failure class "
        f"{classification.failure_class} has fingerprint {classification.fingerprint}; evidence: "
        f"{classification.evidence or 'no additional safe trace text'}.{previous} First explain the fault "
        f"briefly. Before invoking any edit tool, emit `DIAGNOSIS:`, `REPAIR_PLAN:`, and `VERIFICATION:` lines in the trajectory, then patch {targets}. Preserve the supplied contract and protected paths; do not edit protected evaluators, data "
        f"loaders, contracts, or command configuration, then call task_done. You "
        f"cannot execute code: your tools are the editor and task_done, and TacoRank "
        f"runs {check} for you after you finish, so make the smallest edit that should "
        f"make it pass and stop. Spending the step budget without calling task_done "
        f"discards the attempt entirely. This is repair attempt "
        f"{repair_attempt} of {_get(context, 'max_repair_attempts', default=2)}; "
        f"{remaining_budget} repair attempt(s) remain after this decision."
    )
