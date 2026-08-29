"""Phase-aware deterministic context compiler (Ring 0 + Ring 1)."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

from ..artifacts import ArtifactStore
from ..config import RunConfig, VerifiedContract
from ..memory.projections import project
from ..memory.retrieval import (
    active_lessons,
    failure_chain,
    verified_experiment_history,
    visible_development_events,
)
from ..recovery.classifier import classify_failure
from ..recovery.fingerprints import fingerprint_result
from ..schemas import (
    ArtifactKind,
    CoderContext,
    ContextDocument,
    CostTier,
    Event,
    EventType,
    ExperimentDecisionKind,
    ExperimentSpec,
    PlannerBudgetSummary,
    PlannerContractSummary,
    PlannerContext,
    PlannerConvergenceSummary,
    PlannerExperimentSummary,
    PlannerMethodCardSummary,
    RecoveryContext,
)
from ..research.portfolio import ALL_FAMILIES, load_method_cards
from .redaction import redact
from .templates import compact_json, render_context
from .token_estimator import estimate_tokens


def _prediction_change_fraction(value: object) -> float:
    changed_fraction = getattr(value, "changed_row_fraction", value)
    return float(changed_fraction)


class ContextBuildError(RuntimeError):
    pass


ContextT = TypeVar("ContextT", bound=ContextDocument)


class ContextBuilder:
    def __init__(
        self,
        config: RunConfig,
        verified_contract: VerifiedContract,
        artifact_store: ArtifactStore,
    ):
        self.config = config
        self.verified_contract = verified_contract
        self.artifact_store = artifact_store

    def _contract_digest(self) -> str:
        contract = (self.config.repository_root / self.config.contract_path).read_text(
            encoding="utf-8"
        )
        lines = [line.rstrip() for line in contract.splitlines() if line.strip()]
        # Contract is mandatory. It may be compacted only by deterministic removal
        # of blank lines; substantive text is never summarized or truncated.
        return "\n".join(lines)

    def _protected_digest(self) -> str:
        return (
            self.config.repository_root / self.config.protected_paths_path
        ).read_text(encoding="utf-8").strip()

    def _identity(
        self,
        role: str,
        events: Sequence[Event],
        experiment_id: Optional[str],
        build_inputs: object,
    ) -> str:
        source = compact_json(
            {
                "run_id": self.config.run_id,
                "role": role,
                "experiment_id": experiment_id,
                "snapshot_hash": events[-1].event_hash if events else "genesis",
                "build_inputs": build_inputs,
            }
        )
        return "ctx_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def _event_card(self, event: Event) -> str:
        return compact_json(
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "seq": event.seq,
                "timestamp": event.timestamp.isoformat(),
                "payload": event.payload.model_dump(mode="json", exclude_none=False),
            }
        )

    def _pack(
        self,
        *,
        context_id: str,
        role: str,
        mandatory: Sequence[Tuple[str, str]],
        optional: Sequence[Tuple[str, str, str]],
        max_tokens: int,
    ) -> Tuple[str, List[str], Dict[str, str]]:
        instruction = list(mandatory)
        content = render_context(
            context_id=context_id,
            role=role,
            instruction_sections=instruction,
            evidence_sections=[],
        )
        redacted, _ = redact(content)
        if estimate_tokens(redacted) > max_tokens:
            raise ContextBuildError("mandatory context exceeds the hard token budget")

        included: List[str] = []
        excluded: Dict[str, str] = {}
        evidence: List[Tuple[str, str]] = []
        for source_id, title, body in optional:
            candidate_evidence = evidence + [(title, body)]
            candidate = render_context(
                context_id=context_id,
                role=role,
                instruction_sections=instruction,
                evidence_sections=candidate_evidence,
            )
            candidate, _ = redact(candidate)
            if estimate_tokens(candidate) <= max_tokens:
                evidence = candidate_evidence
                included.append(source_id)
            else:
                excluded[source_id] = "token_budget"
        final = render_context(
            context_id=context_id,
            role=role,
            instruction_sections=instruction,
            evidence_sections=evidence,
        )
        final, _ = redact(final)
        return final, included, excluded

    def _persist(
        self,
        context_type: Type[ContextT],
        *,
        context_id: str,
        role: str,
        experiment_id: Optional[str],
        events: Sequence[Event],
        content: str,
        included: List[str],
        excluded: Dict[str, str],
        context_fields: Optional[Dict[str, object]] = None,
    ) -> ContextT:
        relative_path = "runs/%s/contexts/%s.md" % (self.config.run_id, context_id)
        artifact = self.artifact_store.write(
            artifact_id="artifact_" + context_id,
            kind=ArtifactKind.CONTEXT,
            relative_path=relative_path,
            content=content.encode("utf-8"),
            content_type="text/markdown",
        )
        known_event_ids = {event.event_id for event in events}
        included_event_ids = [
            source_id for source_id in included if source_id in known_event_ids
        ]
        return context_type(
            context_id=context_id,
            role=role,
            run_id=self.config.run_id,
            experiment_id=experiment_id,
            snapshot_event_id=events[-1].event_id if events else None,
            source_event_ids=list(
                dict.fromkeys(
                    ([events[-1].event_id] if events else [])
                    + included_event_ids
                )
            ),
            excluded_source_ids=excluded,
            content=content,
            estimated_tokens=estimate_tokens(content),
            artifact=artifact,
            **(context_fields or {}),
        )

    def _protected_paths(self) -> List[str]:
        """Return normalized path entries from the frozen Markdown manifest."""

        values: List[str] = []
        manifest = self.config.repository_root / self.config.protected_paths_path
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.lstrip("-* ").strip().strip("`").rstrip("/")
            if line:
                values.append(line)
        return list(dict.fromkeys(values))

    def _trace_tail(self, failed_value: object, fallback: str) -> str:
        """Read only a bounded tail from a hash-verified failure log."""

        artifact = getattr(failed_value, "log_artifact", None)
        if artifact is None:
            return fallback[-4_000:]
        try:
            self.artifact_store.verify(artifact)
            path = self.config.repository_root / artifact.path
            with path.open("rb") as handle:
                handle.seek(max(0, artifact.size_bytes - 8_000))
                raw = handle.read(8_000)
            return raw.decode("utf-8", errors="replace")[-4_000:]
        except (OSError, ValueError):
            return fallback[-4_000:]

    def _planner_experiments(
        self, events: Sequence[Event]
    ) -> Tuple[
        PlannerExperimentSummary,
        PlannerExperimentSummary,
        List[PlannerExperimentSummary],
        List[PlannerExperimentSummary],
    ]:
        """Build the authoritative typed projection consumed by Person 1."""

        state = project(events)
        baseline_event = next(
            event for event in events if event.event_type == EventType.BASELINE_VERIFIED
        )
        baseline_payload = baseline_event.payload
        baseline_evaluation = baseline_payload.evaluation
        baseline = PlannerExperimentSummary(
            experiment_id=baseline_payload.experiment_id,
            parent_experiment_id=None,
            commit_sha=baseline_payload.commit_sha,
            family=None,
            hypothesis_summary="Frozen verified baseline",
            trust_verdict=baseline_evaluation.trust.verdict,
            stability=baseline_evaluation.trust.stability,
            integrity=baseline_evaluation.trust.integrity,
            decision=ExperimentDecisionKind.ACCEPT,
            highest_completed_fidelity=baseline_evaluation.fidelity,
            primary_score=baseline_payload.metric_set.primary_score,
            metric_set=baseline_payload.metric_set,
            metric_deltas={name: 0.0 for name in baseline_payload.metric_set.metrics},
            baseline_delta=0.0,
            parent_delta=0.0,
            previous_best_delta=0.0,
            prediction_change=_prediction_change_fraction(
                baseline_evaluation.prediction_change
            ),
            child_count=0,
            actual_cost=CostTier.LOW,
            parent_eligible=True,
            best_eligible=state.best_experiment_id == baseline_payload.experiment_id,
            status="accepted",
            supporting_event_ids=[baseline_event.event_id],
        )

        spec_events = [
            event for event in events if event.event_type == EventType.EXPERIMENT_PROPOSED
        ]
        evaluation_by_experiment = {}
        decision_by_experiment = {}
        for event in events:
            if event.event_type == EventType.EVALUATION_COMPLETED:
                evaluation_by_experiment[event.payload.result.experiment_id] = event
            elif event.event_type == EventType.EXPERIMENT_DECIDED:
                decision_by_experiment[event.payload.decision.experiment_id] = event

        children = Counter(
            event.payload.spec.parent_experiment_id
            for event in spec_events
            if event.payload.spec.parent_experiment_id is not None
        )
        baseline.child_count = children[baseline.experiment_id]
        metrics_by_experiment = {baseline.experiment_id: baseline_payload.metric_set}
        summaries: List[PlannerExperimentSummary] = []

        for proposal_event in spec_events:
            spec = proposal_event.payload.spec
            node = state.experiments[spec.experiment_id]
            evaluation_event = evaluation_by_experiment.get(spec.experiment_id)
            decision_event = decision_by_experiment.get(spec.experiment_id)
            evaluation = evaluation_event.payload.result if evaluation_event else None
            decision = decision_event.payload.decision if decision_event else None
            metric_set = evaluation.metric_set if evaluation else node.metric_set
            if metric_set is not None:
                metrics_by_experiment[spec.experiment_id] = metric_set
            parent_metrics = metrics_by_experiment.get(spec.parent_experiment_id)
            metric_deltas = {}
            if metric_set is not None and parent_metrics is not None:
                for name, value in metric_set.metrics.items():
                    if name in parent_metrics.metrics:
                        metric_deltas[name] = value - parent_metrics.metrics[name]

            support = [proposal_event.event_id]
            if evaluation_event is not None:
                support.append(evaluation_event.event_id)
            if decision_event is not None:
                support.append(decision_event.event_id)
            summaries.append(
                PlannerExperimentSummary(
                    experiment_id=spec.experiment_id,
                    parent_experiment_id=spec.parent_experiment_id,
                    commit_sha=node.latest_commit_sha or spec.parent_commit_sha,
                    family=spec.family,
                    hypothesis_summary=spec.hypothesis,
                    trust_verdict=evaluation.trust.verdict if evaluation else None,
                    stability=evaluation.trust.stability if evaluation else None,
                    integrity=evaluation.trust.integrity if evaluation else None,
                    decision=decision.decision if decision else None,
                    highest_completed_fidelity=(
                        evaluation.fidelity if evaluation else node.highest_fidelity
                    ),
                    primary_score=(metric_set.primary_score if metric_set else None),
                    metric_set=metric_set,
                    metric_deltas=metric_deltas,
                    baseline_delta=evaluation.baseline_delta if evaluation else None,
                    parent_delta=evaluation.parent_delta if evaluation else None,
                    previous_best_delta=(
                        evaluation.previous_best_delta if evaluation else None
                    ),
                    prediction_change=(
                        _prediction_change_fraction(evaluation.prediction_change)
                        if evaluation
                        else None
                    ),
                    child_count=children[spec.experiment_id],
                    actual_cost=spec.estimated_cost.cost_tier,
                    parent_eligible=bool(decision and decision.parent_eligible),
                    best_eligible=bool(decision and decision.best_eligible),
                    status=node.status.value,
                    duplicate_key=spec.duplicate_key,
                    method_card_ids=list(spec.method_card_ids),
                    supporting_event_ids=support,
                )
            )

        by_id = {baseline.experiment_id: baseline}
        by_id.update({summary.experiment_id: summary for summary in summaries})
        current_best = by_id.get(state.best_experiment_id, baseline)
        eligible_frontier = [baseline] + [
            summary for summary in summaries if summary.parent_eligible
        ]
        return baseline, current_best, eligible_frontier, summaries

    def _planner_context_fields(self, events: Sequence[Event]) -> Dict[str, object]:
        state = project(events)
        baseline, current_best, eligible_frontier, family_history = (
            self._planner_experiments(events)
        )
        card_directory = self.config.repository_root / "research/methods"
        method_cards = [
            PlannerMethodCardSummary(
                method_id=card.method_id,
                family=card.family,
                status=card.status,
                cost_tier=card.cost_tier,
            )
            for card in load_method_cards(card_directory).cards
        ]
        totals = state.resource_totals
        remaining_tokens = (
            None
            if self.config.token_limit is None
            else max(0, self.config.token_limit - totals.total_reported_tokens)
        )
        remaining_gpu = (
            None
            if self.config.gpu_seconds_limit is None
            else max(
                0,
                self.config.gpu_seconds_limit
                - int(totals.gpu_weighted_time_ms / 1000),
            )
        )
        return {
            "contract_sha256": self.verified_contract.contract_sha256,
            "contract_summary": PlannerContractSummary(
                resolved=True,
                allowed_families=list(ALL_FAMILIES),
                protected_paths=self._protected_paths(),
                editable_paths=list(self.config.editable_roots),
                epsilon=self.config.convergence_epsilon,
            ),
            "baseline": baseline,
            "current_best": current_best,
            "eligible_frontier": eligible_frontier,
            "family_history": family_history,
            "method_cards": method_cards,
            "remaining_budget": PlannerBudgetSummary(
                remaining_experiments=state.remaining_experiments,
                remaining_public_queries=None,
                remaining_llm_tokens=remaining_tokens,
                remaining_wall_time_seconds=max(
                    0,
                    self.config.wall_time_limit_seconds
                    - int(state.elapsed_wall_time_seconds),
                ),
                remaining_gpu_seconds=remaining_gpu,
            ),
            "convergence": PlannerConvergenceSummary(
                patience=self.config.convergence_patience,
                consecutive_non_improving_full_evaluations=(
                    state.consecutive_non_improving_full_evaluations
                ),
                full_evaluations_completed=state.full_evaluations_completed,
            ),
        }

    def build_planner(
        self,
        events: Sequence[Event],
        *,
        family: Optional[str] = None,
        tags: Iterable[str] = (),
        max_tokens: Optional[int] = None,
    ) -> PlannerContext:
        visible = visible_development_events(events)
        state = project(visible)
        normalized_tags = sorted({tag.lower() for tag in tags})
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.context_token_limit
        )
        mandatory = [
            ("Frozen contract", self._contract_digest()),
            (
                "Current objective and budget",
                compact_json(
                    {
                        "run_id": state.run_id,
                        "phase": state.phase,
                        "baseline_primary_score": state.baseline_primary_score,
                        "best_experiment_id": state.best_experiment_id,
                        "best_primary_score": state.best_primary_score,
                        "remaining_experiments": state.remaining_experiments,
                        "public_validation_queries": state.public_validation_queries,
                        "consecutive_non_improving_full_evaluations": state.consecutive_non_improving_full_evaluations,
                        "allowed_action": "return PlannerOutput only",
                    }
                ),
            ),
        ]
        optional: List[Tuple[str, str, str]] = []
        if state.best_experiment_id and state.best_experiment_id in state.experiments:
            node = state.experiments[state.best_experiment_id]
            optional.append(
                (
                    "experiment:%s" % node.experiment_id,
                    "Current verified best",
                    compact_json(
                        {
                            "experiment_id": node.experiment_id,
                            "parent_experiment_id": node.parent_experiment_id,
                            "hypothesis": node.hypothesis,
                            "family": node.family,
                            "latest_commit_sha": node.latest_commit_sha,
                            "status": node.status.value,
                            "metric_set": (
                                node.metric_set.model_dump(mode="json")
                                if node.metric_set is not None
                                else None
                            ),
                        }
                    ),
                )
            )
        history = verified_experiment_history(visible, family=family, limit=10)
        for event in history:
            optional.append((event.event_id, "Verified experiment history", self._event_card(event)))
        for event in active_lessons(visible, tags=normalized_tags, limit=5):
            optional.append((event.event_id, "Applicable active lesson", self._event_card(event)))
        for path in sorted((self.config.repository_root / "research/methods").glob("*.md")):
            relative = path.relative_to(self.config.repository_root).as_posix()
            optional.append((relative, "Method card", path.read_text(encoding="utf-8")))

        context_id = self._identity(
            "planner",
            visible,
            None,
            {
                "family": family,
                "tags": normalized_tags,
                "max_tokens": effective_max_tokens,
                "mandatory": mandatory,
                "optional": optional,
            },
        )

        content, included, excluded = self._pack(
            context_id=context_id,
            role="planner",
            mandatory=mandatory,
            optional=optional,
            max_tokens=effective_max_tokens,
        )
        # Explicitly account for sources removed by visibility policy.
        for event in events:
            if event not in visible:
                excluded[event.event_id] = "hidden_final"
        return self._persist(
            PlannerContext,
            context_id=context_id,
            role="planner",
            experiment_id=None,
            events=visible,
            content=content,
            included=included,
            excluded=excluded,
            context_fields=self._planner_context_fields(visible),
        )

    def build_coder(
        self,
        events: Sequence[Event],
        spec: ExperimentSpec,
        *,
        max_tokens: int = 2_500,
    ) -> CoderContext:
        visible = visible_development_events(events)
        mandatory = [
            (
                "Coding assignment",
                compact_json(
                    {
                        "spec": spec.model_dump(mode="json"),
                        "protected_paths": self._protected_digest(),
                        "target_interface_excerpts": (
                            self.config.target_interface_excerpts
                        ),
                        "allowed_output": "PatchCandidate",
                        "hypothesis_drift": "forbidden",
                    }
                ),
            ),
            (
                "Output and budget contract",
                compact_json(
                    {
                        "artifact_roots": self.config.artifact_roots,
                        "network_enabled": False,
                        "estimated_cost": spec.estimated_cost.model_dump(mode="json"),
                    }
                ),
            ),
        ]
        optional: List[Tuple[str, str, str]] = []
        selected_method_cards: List[Dict[str, object]] = []
        wanted = set(spec.method_card_ids)
        for path in sorted((self.config.repository_root / "research/methods").glob("*.md")):
            if path.stem in wanted:
                relative = path.relative_to(self.config.repository_root).as_posix()
                body = path.read_text(encoding="utf-8")
                selected_method_cards.append(
                    {"method_id": path.stem, "path": relative, "content": body}
                )
                optional.append((relative, "Selected method card", body))
        lesson_events = active_lessons(
            visible, tags=[spec.family, spec.target_stage], limit=5
        )
        for event in lesson_events:
            optional.append((event.event_id, "Applicable lesson", self._event_card(event)))
        context_id = self._identity(
            "coder",
            visible,
            spec.experiment_id,
            {
                "max_tokens": max_tokens,
                "mandatory": mandatory,
                "optional": optional,
            },
        )
        content, included, excluded = self._pack(
            context_id=context_id,
            role="coder",
            mandatory=mandatory,
            optional=optional,
            max_tokens=max_tokens,
        )
        return self._persist(
            CoderContext,
            context_id=context_id,
            role="coder",
            experiment_id=spec.experiment_id,
            events=visible,
            content=content,
            included=included,
            excluded=excluded,
            context_fields={
                "contract_sha256": self.verified_contract.contract_sha256,
                "experiment_spec": spec,
                "parent_commit_sha": spec.parent_commit_sha,
                "target_interface_excerpts": self.config.target_interface_excerpts,
                "editable_roots": self.config.editable_roots,
                "protected_paths": self._protected_paths(),
                "allowed_command_ids": self.config.command_ids,
                "selected_method_cards": [
                    card for card in selected_method_cards if card["path"] in included
                ],
                "active_lessons": [
                    {
                        "event_id": event.event_id,
                        "payload": event.payload.model_dump(
                            mode="json", exclude_none=False
                        ),
                    }
                    for event in lesson_events
                    if event.event_id in included
                ],
                "step_limit": self.config.coding_step_limit,
                "token_limit": self.config.coding_token_limit,
                "wall_time_limit_seconds": (
                    self.config.coding_wall_time_limit_seconds
                ),
            },
        )

    def build_recovery(
        self,
        events: Sequence[Event],
        experiment_id: str,
        *,
        remaining_repair_budget: int,
        max_tokens: int = 3_000,
    ) -> RecoveryContext:
        visible = visible_development_events(events)
        state = project(visible)
        if experiment_id not in state.experiments:
            raise ContextBuildError("unknown experiment")
        node = state.experiments[experiment_id]
        chain = failure_chain(visible, experiment_id)
        if not chain:
            raise ContextBuildError("no failure evidence exists for recovery")
        spec_event = next(
            (
                event
                for event in visible
                if event.event_type == EventType.EXPERIMENT_PROPOSED
                and event.payload.spec.experiment_id == experiment_id
            ),
            None,
        )
        if spec_event is None:
            raise ContextBuildError("recovery experiment specification is missing")
        failure_event = chain[-1]
        failed_value = getattr(failure_event.payload, "result", None)
        if failed_value is None:
            raise ContextBuildError("latest recovery evidence has no typed result")
        classification = classify_failure(failed_value)
        failed_checks = getattr(failed_value, "checks", ())
        if isinstance(failed_checks, dict):
            normalized_checks = [
                {"name": name, "status": getattr(status, "value", status)}
                for name, status in failed_checks.items()
            ]
        else:
            normalized_checks = [
                (
                    check.model_dump(mode="json", exclude_none=False)
                    if hasattr(check, "model_dump")
                    else dict(check)
                )
                for check in failed_checks
            ]
        accepted_patch = next(
            (
                event.payload.result
                for event in reversed(visible)
                if event.event_type == EventType.PATCH_CHECKED
                and event.payload.result.experiment_id == experiment_id
                and event.payload.result.accepted
            ),
            None,
        )
        if failure_event.event_type == EventType.PATCH_CHECKED:
            # A Gate-A rejection has no accepted receipt for the rejected
            # commit.  Supplying an older receipt would authorize the wrong
            # bytes and the real repair prompt correctly rejects it.
            accepted_patch = None
        prior_fingerprints = [
            fingerprint_result(event.payload.result)
            for event in chain[:-1]
            if getattr(event.payload, "result", None) is not None
        ]
        repair_attempt = node.repair_count + 1
        mandatory = [
            (
                "Recovery authority",
                compact_json(
                    {
                        "experiment_id": experiment_id,
                        "original_hypothesis": node.hypothesis,
                        "accepted_patch_commit": node.latest_commit_sha,
                        "remaining_repair_budget": remaining_repair_budget,
                        "hypothesis_drift": "forbidden",
                        "allowed_output": "RecoveryDecision or repaired PatchCandidate",
                    }
                ),
            )
        ]
        optional = [
            (event.event_id, "Exact failure chain", self._event_card(event)) for event in reversed(chain)
        ]
        context_id = self._identity(
            "recovery",
            visible,
            experiment_id,
            {
                "remaining_repair_budget": remaining_repair_budget,
                "repair_attempt": repair_attempt,
                "max_tokens": max_tokens,
                "mandatory": mandatory,
                "optional": optional,
            },
        )
        content, included, excluded = self._pack(
            context_id=context_id,
            role="recovery",
            mandatory=mandatory,
            optional=optional,
            max_tokens=max_tokens,
        )
        return self._persist(
            RecoveryContext,
            context_id=context_id,
            role="recovery",
            experiment_id=experiment_id,
            events=visible,
            content=content,
            included=included,
            excluded=excluded,
            context_fields={
                "repair_attempt": repair_attempt,
                "original_experiment_spec": spec_event.payload.spec,
                "current_patch_commit_sha": node.latest_commit_sha,
                "accepted_patch_receipt_id": (
                    accepted_patch.receipt_id if accepted_patch is not None else None
                ),
                "failure_class": classification.failure_class,
                "error_fingerprint": classification.fingerprint,
                "error_summary": (
                    getattr(failed_value, "error_summary", None)
                    or classification.evidence
                ),
                "relevant_trace_tail": self._trace_tail(
                    failed_value,
                    getattr(failed_value, "error_summary", None)
                    or classification.evidence,
                ),
                "failed_checks": normalized_checks,
                "previous_repair_fingerprints": prior_fingerprints,
                "recovery_instructions": "Await the deterministic recovery decision.",
                "remaining_repair_budget": remaining_repair_budget,
                "editable_roots": self.config.editable_roots,
                "protected_paths": self._protected_paths(),
            },
        )
