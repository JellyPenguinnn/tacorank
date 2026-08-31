"""Experiment-family portfolio used by the deterministic search policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable


HIGH_VALUE_FAMILIES: tuple[str, ...] = (
    "objective",
    "temporal_history",
    "multitask",
    "duration_bias",
    "features",
    "model",
)

ALL_FAMILIES: tuple[str, ...] = HIGH_VALUE_FAMILIES + (
    "composition",
    "sampling",
    "ensemble",
    "evaluation",
    "other",
)
METHOD_STATUSES = {"candidate", "blocked", "known_negative", "forbidden"}
METHOD_COST_TIERS = {"low", "medium", "high"}

# Composition is deliberately expressed as compatibility slots instead of a
# Cartesian product.  Methods in one alternative slot replace one another;
# methods in different additive slots may be carried into one candidate when
# their prerequisites are satisfied.  The planner and validator both consume
# this catalog so the policy cannot silently turn an incompatible collection of
# cards into one edit.
COMPOSITION_METHOD_GROUPS: dict[str, tuple[str, ...]] = {
    "primary_objective": (
        "objective_pairwise_bpr",
        "objective_weighted_cross_entropy",
        "objective_listwise_user_softmax",
    ),
    "distillation_objective": ("objective_distill_softmax",),
    "loss_aligned_feature_refinement": ("objective_loss_aligned_features",),
    "feature_residual": (
        "features_general_bounded_engineering",
        "features_tab_context_residual",
        "features_author_affinity_past_only",
        "temporal_drift_past_only",
    ),
    "interest_encoder": (
        "temporal_causal_history_features",
        "temporal_search_interest_model",
        "temporal_deep_interest_network",
        "temporal_time_series_interest",
        "temporal_history_compact",
    ),
    "single_task_backbone": (
        "model_deep_cross_network",
        "model_compact_ranker",
        "model_field_aware_fm",
    ),
    "hidden_unit_adapter": ("model_lhuc",),
    "multitask_backbone": (
        "multitask_ple",
        "multitask_mmoe",
        "multitask_esu",
        "multitask_gsu",
        "multitask_shared_bottom",
        "multitask_single_auxiliary",
    ),
    "duration_residual": ("duration_bias_censored_watch_time",),
    "training_sampler": ("sampling_deterministic_coverage",),
    "posthoc_ensemble": (
        "ensemble_parallel_round_synthesis",
        "ensemble_diverse_residual_candidate",
        "ensemble_confirmed_members",
    ),
    "diagnostic_only": ("evaluation_random_exposure_robustness",),
}

COMPOSITION_PRIMARY_OBJECTIVES = frozenset(
    COMPOSITION_METHOD_GROUPS["primary_objective"]
)
COMPOSITION_DISTILLATION_OBJECTIVES = frozenset(
    COMPOSITION_METHOD_GROUPS["distillation_objective"]
)
COMPOSITION_LOSS_ALIGNED_REFINEMENTS = frozenset(
    COMPOSITION_METHOD_GROUPS["loss_aligned_feature_refinement"]
)
COMPOSITION_FEATURE_METHODS = frozenset(
    COMPOSITION_METHOD_GROUPS["feature_residual"]
)
COMPOSITION_INTEREST_METHODS = frozenset(
    COMPOSITION_METHOD_GROUPS["interest_encoder"]
)
COMPOSITION_SINGLE_TASK_BACKBONES = frozenset(
    COMPOSITION_METHOD_GROUPS["single_task_backbone"]
)
COMPOSITION_HIDDEN_UNIT_ADAPTERS = frozenset(
    COMPOSITION_METHOD_GROUPS["hidden_unit_adapter"]
)
COMPOSITION_MULTITASK_BACKBONES = frozenset(
    COMPOSITION_METHOD_GROUPS["multitask_backbone"]
)
COMPOSITION_OPTIONAL_ADDONS = frozenset(
    COMPOSITION_METHOD_GROUPS["duration_residual"]
    + COMPOSITION_METHOD_GROUPS["training_sampler"]
)
COMPOSITION_MAX_METHODS = 12


@dataclass(frozen=True)
class MethodCard:
    method_id: str
    family: str
    summary: str
    schema_version: str = "1.0"
    status: str = "candidate"
    tags: tuple[str, ...] = ()
    cost_tier: str = "medium"
    mechanism: str = ""
    prerequisites: tuple[str, ...] = ()
    allowed_data: tuple[str, ...] = ()
    expected_effect: str = ""
    falsifier: str = ""
    prohibition_conditions: tuple[str, ...] = ()
    implementation_targets: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    source_path: str | None = None


@dataclass
class ExperimentPortfolio:
    cards: list[MethodCard] = field(default_factory=list)

    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(card.family for card in self.cards))

    def for_family(self, family: str) -> tuple[MethodCard, ...]:
        return tuple(card for card in self.cards if card.family == family)

    def ids(self) -> set[str]:
        return {card.method_id for card in self.cards}

    def legal_families(self, allowed: Iterable[str] | None = None) -> tuple[str, ...]:
        if allowed is None:
            return ALL_FAMILIES
        allowed_set = {str(item) for item in allowed}
        return tuple(family for family in ALL_FAMILIES if family in allowed_set)


def default_portfolio() -> ExperimentPortfolio:
    """Return the documented seed portfolio without touching the repository."""

    return ExperimentPortfolio(
        cards=[
            MethodCard(
                method_id="objective_pairwise_bpr",
                family="objective",
                summary="Align training with within-user ranking metrics using pairwise BPR.",
                tags=("ranking", "bpr", "loss"),
                cost_tier="medium",
                mechanism="Optimize relative positive-versus-negative ordering within users.",
                prerequisites=("baseline_parity", "within_user_positive_negative_pairs"),
                allowed_data=("train_interactions", "user_id", "long_view"),
                expected_effect="Improve GAUC and nDCG-aligned ordering.",
                falsifier="No stable primary-score improvement over the pointwise parent.",
                prohibition_conditions=("evaluator_or_split_change_required",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="temporal_causal_history_features",
                family="temporal_history",
                summary=(
                    "Explore date-strict causal history signals over the verified "
                    "FM parent."
                ),
                tags=("temporal", "causal", "history"),
                cost_tier="medium",
                mechanism=(
                    "Implement and test a causal-history score path using earlier "
                    "interactions without using the scored row or future data."
                ),
                prerequisites=("baseline_parity", "strict_temporal_cutoff"),
                allowed_data=(
                    "train_interactions",
                    "date",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "duration_ms",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect=(
                    "Improve within-user ordering under chronological drift while "
                    "preserving the FM score path for unsupported rows."
                ),
                falsifier=(
                    "No trusted full-fidelity gain, a constant within-user residual, "
                    "or any evidence of current-row or future-label leakage."
                ),
                prohibition_conditions=(
                    "future_aggregate_required",
                    "ambiguous_within_date_order",
                    "unsupported_input_required",
                    "validation_tuned_weights",
                ),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="temporal_history_compact",
                family="temporal_history",
                summary="Add strictly past user history with deterministic truncation and padding.",
                tags=("sequence", "history", "temporal"),
                cost_tier="medium",
                mechanism="Represent recent user interest without using future interactions.",
                prerequisites=("strict_temporal_cutoff",),
                allowed_data=(
                    "train_interactions",
                    "date",
                    "user_id",
                    "video_id",
                    "author_id",
                ),
                expected_effect="Improve preference modeling for users with useful history.",
                falsifier="No gain over a no-history control or evidence of temporal leakage.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="model_compact_ranker",
                family="model",
                summary="Try a compact alternative ranker after objective/data-frame validation.",
                tags=("deepfm", "dcn", "model"),
                cost_tier="high",
                mechanism="Capture interactions not represented by the baseline FM.",
                prerequisites=("baseline_parity", "objective_data_frame_verified"),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "date",
                    "duration_ms",
                    "long_view",
                ),
                expected_effect="Improve ranking through additional interactions.",
                falsifier="No improvement after a bounded, mechanism-driven trial.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="multitask_single_auxiliary",
                family="multitask",
                summary="Add one legal auxiliary engagement target while preserving the primary head.",
                tags=("multitask", "auxiliary"),
                cost_tier="medium",
                mechanism="Use related engagement supervision to regularize long-view prediction.",
                prerequisites=("legal_auxiliary_label",),
                allowed_data=(
                    "train_interactions",
                    "long_view",
                    "auxiliary_engagement_labels",
                ),
                expected_effect="Improve generalization of the primary ranking head.",
                falsifier="Auxiliary task degrades primary validation or violates the contract.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="duration_bias_censored_watch_time",
                family="duration_bias",
                summary="Model censored watch duration only when the frozen contract permits it.",
                tags=("cwm", "duration", "censoring"),
                cost_tier="high",
                mechanism="Use one-sided duration supervision to address watch-time censoring.",
                prerequisites=("duration_features_legal",),
                allowed_data=("train_interactions", "duration_ms", "long_view"),
                expected_effect="Improve long-view ranking through duration bias correction.",
                falsifier="No primary improvement or mismatch with the competition definition.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="features_author_affinity_past_only",
                family="features",
                summary="Add a strictly past-only author affinity residual.",
                tags=("features", "author", "temporal", "residual"),
                cost_tier="medium",
                mechanism=(
                    "Capture creator-level preference not fully represented by "
                    "the supplied FM score."
                ),
                prerequisites=("baseline_parity", "strict_temporal_cutoff"),
                allowed_data=("train_interactions", "date", "author_id", "long_view"),
                expected_effect="Improve ranking when recent author preference has residual signal.",
                falsifier="No reproducible primary gain from the bounded residual.",
                prohibition_conditions=("future_aggregate_required",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="features_tab_context_residual",
                family="features",
                summary="Add one train-only tab-context long-view residual.",
                tags=("features", "tab", "context", "residual"),
                cost_tier="medium",
                mechanism=(
                    "Model systematic long-view-rate differences across feed "
                    "contexts missing from the FM score."
                ),
                prerequisites=("baseline_parity",),
                allowed_data=("train_interactions", "tab", "long_view"),
                expected_effect="Improve relative scores across feed contexts.",
                falsifier="No primary gain or a constant residual within user candidates.",
                prohibition_conditions=("evaluator_or_split_change_required",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="sampling_deterministic_coverage",
                family="sampling",
                summary="Use deterministic per-user coverage-preserving training samples.",
                tags=("sampling", "coverage", "within_user"),
                cost_tier="medium",
                mechanism=(
                    "Prevent high-volume users from dominating the bounded "
                    "residual learner."
                ),
                prerequisites=("baseline_parity",),
                allowed_data=("train_interactions", "user_id", "long_view"),
                expected_effect="Improve generalization through representative user coverage.",
                falsifier="No primary gain or loss of useful positive/negative evidence.",
                prohibition_conditions=("adaptive_validation_sampling",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="ensemble_parallel_round_synthesis",
                family="ensemble",
                summary="Align all independently accepted parallel-round members.",
                tags=("ensemble", "parallel_round", "alignment", "composition"),
                cost_tier="medium",
                mechanism=(
                    "Preserve compatible improvements on the strongest accepted "
                    "member and resolve overlapping score-path changes explicitly."
                ),
                prerequisites=("two_confirmed_clean_members",),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "date",
                    "duration_ms",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Retain complementary gains in one reproducible candidate.",
                falsifier="The aligned candidate fails a gate or does not improve the best member.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="ensemble_diverse_residual_candidate",
                family="ensemble",
                summary=(
                    "Test one fixed blend of the trusted parent and one clean, "
                    "diverse soft-pruned mechanism."
                ),
                tags=("ensemble", "residual", "rank_average", "soft_prune"),
                cost_tier="low",
                mechanism=(
                    "Retain the trusted parent score path and add one bounded, "
                    "explicitly identified complementary scoring path."
                ),
                prerequisites=(
                    "verified_best_prediction",
                    "diverse_clean_proxy_member",
                ),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "date",
                    "duration_ms",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect=(
                    "Improve the trusted parent only when the weaker mechanism "
                    "contributes complementary within-user ordering."
                ),
                falsifier="No fixed blend improves over the trusted parent.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="ensemble_confirmed_members",
                family="ensemble",
                summary="Rank-average two or three already confirmed complementary candidates.",
                tags=("ensemble", "rank_average"),
                cost_tier="low",
                mechanism="Reduce variance by combining complementary trusted rankers.",
                prerequisites=("two_confirmed_clean_members",),
                allowed_data=("verified_predictions",),
                expected_effect="Improve stability or small residual headroom.",
                falsifier="No gain over the best member or incompatible score behavior.",
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/inference.py",
                ),
            ),
        ]
    )


def load_method_cards(directory: str | Path) -> ExperimentPortfolio:
    """Load schema-v1 cards with a JSON machine-readable first fenced block.

    A small YAML-like front-matter fallback is retained for migration of older
    local cards, but new cards should use the attached TacoRank memory schema.
    Malformed cards are skipped rather than becoming planner knowledge.
    """

    directory = Path(directory)
    cards: list[MethodCard] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        values: dict[str, object] = {}
        match = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.DOTALL)
        body_text = text
        if match:
            try:
                decoded = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, dict):
                continue
            values = decoded
            body_text = text[match.end() :]
        else:
            # Migration fallback for the initial local scaffold.
            body: list[str] = []
            in_front_matter = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped == "---":
                    in_front_matter = not in_front_matter
                    continue
                if in_front_matter and ":" in line:
                    key, value = line.split(":", 1)
                    values[key.strip()] = value.strip()
                else:
                    body.append(line)
            body_text = "\n".join(body)

        required_headings = (
            "Mechanism",
            "Preconditions",
            "Allowed data",
            "Expected effect",
            "Falsification condition",
            "Do not use when",
            "Minimal implementation",
            "Sources",
        )
        if match and not all(re.search(rf"^##\s+{re.escape(heading)}\s*$", body_text, re.MULTILINE) for heading in required_headings):
            continue

        def text_value(name: str, default: str = "") -> str:
            value = values.get(name, default)
            return str(value) if value is not None else default

        def list_value(name: str) -> tuple[str, ...]:
            value = values.get(name, ())
            if isinstance(value, str):
                return tuple(item.strip() for item in value.split(",") if item.strip())
            if isinstance(value, (list, tuple, set)):
                return tuple(str(item) for item in value)
            return ()

        def section_value(heading: str) -> str:
            section = re.search(
                rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
                body_text,
                flags=re.MULTILINE,
            )
            return " ".join(line.strip() for line in section.group(1).splitlines() if line.strip()) if section else ""

        mechanism = text_value("mechanism") or section_value("Mechanism")
        expected_effect = text_value("expected_effect") or section_value("Expected effect")
        falsifier = text_value("falsifier") or text_value("falsification_condition") or section_value("Falsification condition")
        prerequisites = list_value("prerequisites") or ((section_value("Preconditions"),) if section_value("Preconditions") else ())
        allowed_data = list_value("allowed_data") or ((section_value("Allowed data"),) if section_value("Allowed data") else ())
        prohibition = list_value("prohibition_conditions") or ((section_value("Do not use when"),) if section_value("Do not use when") else ())
        implementation_targets = list_value("implementation_targets")
        sources = list_value("sources")
        if not sources and section_value("Sources"):
            sources = (section_value("Sources"),)

        method_id = text_value("method_id", path.stem)
        family = text_value("family", "other")
        summary = text_value("summary") or mechanism
        schema_version = text_value("schema_version", "1.0")
        status = text_value("status", "candidate")
        cost_tier = text_value("cost_tier", "medium")
        if (
            not method_id
            or schema_version != "1.0"
            or family not in set(ALL_FAMILIES)
            or status not in METHOD_STATUSES
            or cost_tier not in METHOD_COST_TIERS
            or not mechanism
            or not allowed_data
            or not expected_effect
            or not falsifier
        ):
            continue
        cards.append(
            MethodCard(
                method_id=method_id,
                family=family,
                summary=summary,
                schema_version=schema_version,
                status=status,
                tags=list_value("tags"),
                cost_tier=cost_tier,
                mechanism=mechanism,
                prerequisites=prerequisites,
                allowed_data=allowed_data,
                expected_effect=expected_effect,
                falsifier=falsifier,
                prohibition_conditions=prohibition,
                implementation_targets=implementation_targets,
                sources=sources,
                source_path=str(path),
            )
        )
    return ExperimentPortfolio(cards=cards)
