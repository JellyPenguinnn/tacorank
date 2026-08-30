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
    "sampling",
    "ensemble",
    "evaluation",
    "other",
)
METHOD_STATUSES = {"candidate", "blocked", "known_negative", "forbidden"}
METHOD_COST_TIERS = {"low", "medium", "high"}


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
