"""Evidence-backed research lessons and frame-staleness recommendations."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from tacorank.evaluation.types import EvaluationResult, Integrity, Stability, Verdict

from .lesson_rules import lesson_allowed


@dataclass(frozen=True)
class LessonCandidate:
    origin: str
    category: str
    tags: Tuple[str, ...]
    summary: str
    applicability: str
    avoid_when: str
    confidence: float
    source_event_ids: Tuple[str, ...]
    source_commit_shas: Tuple[str, ...]
    measured_under_frame_experiment_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.origin != "research":
            raise ValueError("research lessons must have origin='research'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("lesson confidence must be in [0, 1]")
        if not self.summary or not self.applicability or not self.avoid_when:
            raise ValueError("lesson narrative fields must not be empty")
        if not self.source_event_ids:
            raise ValueError("research lessons require source events")


@dataclass(frozen=True)
class ActiveLesson:
    lesson_id: str
    category: str
    tags: Tuple[str, ...]
    measured_under_frame_experiment_id: Optional[str]


@dataclass(frozen=True)
class StalenessRecommendation:
    lesson_id: str
    new_status: str
    reason: str
    source_event_ids: Tuple[str, ...]


def build_research_lesson(
    result: EvaluationResult,
    evaluation_event_ids: Sequence[str],
    commit_shas: Sequence[str],
    family: str,
    hypothesis: str,
    expected_mechanism: str,
    applicability: str,
    avoid_when: str,
    measured_under_frame_experiment_id: Optional[str] = None,
) -> Optional[LessonCandidate]:
    trust = result.trust
    if not lesson_allowed(
        trust.verdict, result.fidelity, result.population, trust.stability
    ):
        return None

    metric_text = _metric_delta_text(result.parent_delta.metrics)
    if trust.verdict == Verdict.SUSPICIOUS:
        summary = (
            "Integrity warning: %s. The measured score must not be used as research reward."
            % ", ".join(trust.flags or ("unverified evaluation",))
        )
        category = "integrity_warning"
        confidence = 0.95 if trust.integrity == Integrity.COMPROMISED else 0.75
        tags = (family, "integrity")
    elif trust.verdict == Verdict.REDUNDANT:
        summary = (
            "Observation: %s produced a delta pattern already represented by an accepted node (%s)."
            % (hypothesis, metric_text)
        )
        category = "research_result"
        confidence = 0.85
        tags = (family, "saturated", "redundant")
    elif trust.verdict == Verdict.NEGATIVE:
        summary = (
            "Observation: a verified implementation of '%s' did not improve the parent beyond noise (%s). "
            "Causal hypothesis: %s was insufficient under the current evaluation frame."
            % (hypothesis, metric_text, expected_mechanism)
        )
        category = "research_result"
        confidence = 0.80
        tags = (family, "falsified_under_frame")
    else:
        conflict = ""
        if "METRIC_DIRECTION_CONFLICT" in trust.flags:
            conflict = " The contract metrics moved in opposite directions."
        summary = (
            "Observation: '%s' produced a confirmed clean result versus its parent (%s). "
            "Causal hypothesis: %s.%s"
            % (hypothesis, metric_text, expected_mechanism, conflict)
        )
        category = "research_result"
        confidence = 0.90
        tags = (family, "confirmed")
    return LessonCandidate(
        origin="research",
        category=category,
        tags=tags,
        summary=summary,
        applicability=applicability,
        avoid_when=avoid_when,
        confidence=confidence,
        source_event_ids=tuple(evaluation_event_ids),
        source_commit_shas=tuple(commit_shas),
        measured_under_frame_experiment_id=measured_under_frame_experiment_id,
    )


def recommend_frame_staleness(
    accepted_frame_experiment_id: str,
    accepted_decision_event_id: str,
    active_lessons: Sequence[ActiveLesson],
    content_tags: Sequence[str] = (
        "feature",
        "features",
        "sequence",
        "temporal_history",
        "multitask",
        "duration_bias",
        "temporal",
        "content",
    ),
) -> Tuple[StalenessRecommendation, ...]:
    content = set(content_tags)
    recommendations = []
    for lesson in active_lessons:
        if lesson.category != "research_result":
            continue
        if not content.intersection(lesson.tags):
            continue
        old_frame = lesson.measured_under_frame_experiment_id
        if old_frame is None or old_frame == accepted_frame_experiment_id:
            continue
        recommendations.append(
            StalenessRecommendation(
                lesson_id=lesson.lesson_id,
                new_status="stale",
                reason=(
                    "Measured under objective frame %s, superseded by accepted frame %s."
                    % (old_frame, accepted_frame_experiment_id)
                ),
                source_event_ids=(accepted_decision_event_id,),
            )
        )
    return tuple(recommendations)


def _metric_delta_text(deltas: Mapping[str, float]) -> str:
    return ", ".join(
        "%s %+.6f" % (name, deltas[name]) for name in sorted(deltas)
    )
