"""Phase-aware deterministic context compiler (Ring 0 + Ring 1)."""

from __future__ import annotations

import hashlib
from pathlib import Path
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
from ..schemas import (
    ArtifactKind,
    CoderContext,
    ContextDocument,
    Event,
    EventType,
    ExperimentSpec,
    PlannerContext,
    RecoveryContext,
)
from .redaction import redact
from .templates import compact_json, render_context
from .token_estimator import estimate_tokens


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
    ) -> ContextT:
        relative_path = "runs/%s/contexts/%s.md" % (self.config.run_id, context_id)
        artifact = self.artifact_store.write(
            artifact_id="artifact_" + context_id,
            kind=ArtifactKind.CONTEXT,
            relative_path=relative_path,
            content=content.encode("utf-8"),
            content_type="text/markdown",
        )
        return context_type(
            context_id=context_id,
            role=role,
            run_id=self.config.run_id,
            experiment_id=experiment_id,
            snapshot_event_id=events[-1].event_id if events else None,
            source_event_ids=list(
                dict.fromkeys(
                    ([events[-1].event_id] if events else []) + included
                )
            ),
            excluded_source_ids=excluded,
            content=content,
            estimated_tokens=estimate_tokens(content),
            artifact=artifact,
        )

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
        wanted = set(spec.method_card_ids)
        for path in sorted((self.config.repository_root / "research/methods").glob("*.md")):
            if path.stem in wanted:
                relative = path.relative_to(self.config.repository_root).as_posix()
                optional.append((relative, "Selected method card", path.read_text(encoding="utf-8")))
        for event in active_lessons(visible, tags=[spec.family, spec.target_stage], limit=5):
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
        )
