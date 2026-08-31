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
                method_id="objective_direct_within_user_ranker",
                family="objective",
                summary=(
                    "Replace FM with a direct within-user pairwise/listwise ranker."
                ),
                tags=("pairwise", "listwise", "within_user", "parent_replacement"),
                cost_tier="medium",
                mechanism=(
                    "Replace the FM score path with a directly trained within-user "
                    "ranker optimized by pairwise BPR or a bounded pairwise-listwise "
                    "objective."
                ),
                prerequisites=(
                    "baseline_parity",
                    "within_user_positive_negative_pairs",
                    "user_impression_groups",
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
                ),
                expected_effect=(
                    "Learn user-conditioned relative ordering without an FM residual."
                ),
                falsifier=(
                    "No meaningful within-user rank change or trusted metric gain."
                ),
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
                method_id="objective_pairwise_hinge_margin",
                family="objective",
                summary="Train a direct within-user ranker with a fixed hinge margin.",
                tags=("pairwise", "margin", "within_user", "parent_replacement"),
                cost_tier="medium",
                mechanism=(
                    "Replace FM with a compact direct ranker trained by bounded "
                    "within-user pairwise hinge loss."
                ),
                prerequisites=("baseline_parity", "within_user_positive_negative_pairs"),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "duration_ms",
                    "long_view",
                ),
                expected_effect="Create useful positive-negative score separation.",
                falsifier="The margin collapses scores or does not improve within-user ranking.",
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
                method_id="objective_lambda_ndcg_surrogate",
                family="objective",
                summary="Weight direct-ranker pair gradients by bounded top-five impact.",
                tags=("listwise", "ndcg", "within_user", "parent_replacement"),
                cost_tier="high",
                mechanism=(
                    "Replace FM with a compact ranker using clipped delta-nDCG "
                    "weights on deterministic within-user pairs."
                ),
                prerequisites=("baseline_parity", "user_impression_groups"),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "duration_ms",
                    "long_view",
                ),
                expected_effect="Focus training on swaps that affect nDCG@5.",
                falsifier="Top-five ranking does not improve or GAUC materially regresses.",
                prohibition_conditions=("uninformative_lists_unhandled",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="objective_loss_aligned_features",
                family="objective",
                summary="Add one bounded representation while holding ranking loss fixed.",
                tags=("features", "pairwise", "loss_alignment", "within_user"),
                cost_tier="medium",
                mechanism=(
                    "Add training-only features that vary inside the tested loss's "
                    "within-user comparison group."
                ),
                prerequisites=("pairwise_tested",),
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
                expected_effect="Give a tested ranking loss discriminative legal signals.",
                falsifier="Features do not change user ranks or improve trusted evaluation.",
                prohibition_conditions=(
                    "simultaneous_loss_change",
                    "future_or_validation_aggregate_required",
                ),
                implementation_targets=("solution/candidate.py",),
            ),
            MethodCard(
                method_id="objective_listwise_user_softmax",
                family="objective",
                summary="Optimize a bounded softmax over each observed user list.",
                tags=("listwise", "within_user", "top_k"),
                cost_tier="medium",
                mechanism="Train against deterministic within-user list distributions.",
                prerequisites=("pairwise_tested", "user_impression_groups"),
                allowed_data=("train_interactions", "user_id", "long_view"),
                expected_effect="Improve top-five placement while retaining broad ordering.",
                falsifier="nDCG@5 does not improve or GAUC materially regresses.",
                prohibition_conditions=("uninformative_lists_unhandled",),
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
                method_id="temporal_recency_weighted_ranker",
                family="temporal_history",
                summary="Weight a bounded residual ranker by fixed historical recency.",
                tags=("recency", "temporal", "within_user", "residual"),
                cost_tier="medium",
                mechanism=(
                    "Use one fixed exponential decay over strictly historical "
                    "interactions when fitting a parent-scale residual."
                ),
                prerequisites=("baseline_parity", "strict_temporal_cutoff"),
                allowed_data=(
                    "train_interactions",
                    "date",
                    "user_id",
                    "video_id",
                    "author_id",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Reduce stale-preference bias without a long sequence model.",
                falsifier="Recency weighting fails to improve ranking or overfits one date.",
                prohibition_conditions=("future_aggregate_required",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="temporal_hour_context",
                family="temporal_history",
                summary="Add a smoothed train-only hour-of-day and tab residual.",
                tags=("hour", "context", "temporal", "residual"),
                cost_tier="low",
                mechanism="Fit fixed hour-bucket and tab interactions on past training rows.",
                prerequisites=("baseline_parity", "strict_temporal_cutoff"),
                allowed_data=(
                    "train_interactions",
                    "date",
                    "hourmin",
                    "user_id",
                    "tab",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Capture stable consumption-time context omitted by FM.",
                falsifier="The residual is rank-wise inert or fails to generalize by date.",
                prohibition_conditions=("future_aggregate_required",),
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
                method_id="multitask_watch_time_auxiliary",
                family="multitask",
                summary="Regularize long-view ranking with clipped watch-time supervision.",
                tags=("multitask", "watch_time", "auxiliary"),
                cost_tier="medium",
                mechanism="Add one fixed-weight clipped play-time auxiliary task.",
                prerequisites=("baseline_parity", "legal_auxiliary_label"),
                allowed_data=(
                    "train_interactions",
                    "long_view",
                    "play_time_ms",
                    "duration_ms",
                    "auxiliary_engagement_labels",
                    "verified_predictions",
                ),
                expected_effect="Provide graded engagement signal beyond a binary target.",
                falsifier="The auxiliary task hurts ranking or only predicts video duration.",
                prohibition_conditions=("auxiliary_label_not_permitted",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="multitask_negative_feedback_auxiliary",
                family="multitask",
                summary="Use explicit negative feedback as one auxiliary penalty.",
                tags=("multitask", "negative_feedback", "auxiliary"),
                cost_tier="medium",
                mechanism="Optimize long-view ranking with one fixed-weight dislike auxiliary.",
                prerequisites=("baseline_parity", "legal_auxiliary_label"),
                allowed_data=(
                    "train_interactions",
                    "long_view",
                    "is_hate",
                    "auxiliary_engagement_labels",
                    "verified_predictions",
                ),
                expected_effect="Distinguish superficially long views carrying explicit dislike.",
                falsifier="Sparse feedback destabilizes training or hurts both metrics.",
                prohibition_conditions=("auxiliary_label_not_permitted",),
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
                method_id="temporal_drift_past_only",
                family="features",
                summary="Represent diagnosed chronological shift with past-only statistics.",
                tags=("temporal", "drift", "recency"),
                cost_tier="low",
                mechanism="Add one recency-decayed entity statistic or time interaction.",
                prerequisites=("strict_temporal_cutoff", "drift_diagnostics_material"),
                allowed_data=(
                    "train_interactions",
                    "date",
                    "user_id",
                    "video_id",
                    "author_id",
                    "duration_ms",
                    "long_view",
                ),
                expected_effect="Improve ranking on later windows without future leakage.",
                falsifier="The gain disappears under strict chronological recomputation.",
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
                method_id="features_frequency_crosses",
                family="features",
                summary="Add smoothed train-frequency and crossed-frequency residuals.",
                tags=("frequency", "feature_interaction", "cold_start", "residual"),
                cost_tier="low",
                mechanism=(
                    "Use fixed log-count transforms and a small fixed cross set for "
                    "user, item, author, and tab exposure regimes."
                ),
                prerequisites=("baseline_parity",),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Correct systematic errors across head and cold entities.",
                falsifier="The features reproduce popularity or leave user ranks unchanged.",
                prohibition_conditions=("validation_fitted_feature",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="features_duration_context_interactions",
                family="features",
                summary="Cross fixed duration buckets with tab and author-frequency context.",
                tags=("duration", "context", "feature_interaction", "residual"),
                cost_tier="low",
                mechanism="Fit smoothed train-only duration-context residual cells.",
                prerequisites=("baseline_parity", "duration_features_legal"),
                allowed_data=(
                    "train_interactions",
                    "duration_ms",
                    "tab",
                    "author_id",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Model context-dependent duration bias missed globally.",
                falsifier="Sparse interactions overfit or fail to alter within-user order.",
                prohibition_conditions=("watch_time_semantics_confused",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="model_field_aware_ranker",
                family="model",
                summary="Replace FM with a compact field-aware personalized ranker.",
                tags=("field_aware", "feature_interaction", "compact", "parent_replacement"),
                cost_tier="high",
                mechanism=(
                    "Use separate low-rank user-item, user-author, and user-tab "
                    "interaction parameters."
                ),
                prerequisites=("baseline_parity", "objective_data_frame_verified"),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "duration_ms",
                    "long_view",
                ),
                expected_effect="Represent personalized interactions shared FM factors blur.",
                falsifier="The model exceeds bounds or fails to beat simpler rankers.",
                prohibition_conditions=("unbounded_model_growth",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
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
                method_id="sampling_hard_negative_pairs",
                family="sampling",
                summary="Use deterministic confusable negatives from observed user impressions.",
                tags=("hard_negative", "pairwise", "sampling"),
                cost_tier="medium",
                mechanism=(
                    "Fit a bounded residual with a fixed hard/easy mixture of "
                    "within-user negative pairs."
                ),
                prerequisites=("baseline_parity", "within_user_positive_negative_pairs"),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "long_view",
                    "verified_predictions",
                ),
                expected_effect="Focus CPU updates on confusable rather than easy negatives.",
                falsifier="Hard sampling narrows coverage or hurts broad GAUC.",
                prohibition_conditions=("adaptive_validation_sampling",),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
            ),
            MethodCard(
                method_id="ensemble_causal_rolling_residual_blend",
                family="ensemble",
                summary=(
                    "Blend strict causal rolling feedback with diverse compact "
                    "rankers and fixed per-user residual corrections."
                ),
                tags=(
                    "ensemble",
                    "rolling_feedback",
                    "causal_history",
                    "residual",
                    "out_of_time",
                ),
                cost_tier="high",
                mechanism=(
                    "Build one strict earlier-row feature mode with deterministic "
                    "same-timestamp batching, train compact LambdaRank, "
                    "rank_xendcg, and CatBoost YetiRank members, then apply "
                    "frozen-history LightGBM, rank2, and DIN-style sequence/time "
                    "residual corrections. Sample-z-normalize each member per "
                    "user and use only the fixed sparse blend "
                    "Z(lab_base) - 0.40*Z(frozen_lgb) - 0.10*Z(rank2) "
                    "+ 0.15*Z(DIN50); do not tune weights on public validation."
                ),
                prerequisites=(
                    "baseline_parity",
                    "strict_temporal_cutoff",
                    "standard_public_evaluation_complete",
                    "rolling_feedback_mode_declared",
                ),
                allowed_data=(
                    "train_interactions",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "date",
                    "hourmin",
                    "duration_ms",
                    "long_view",
                    "is_click",
                    "is_like",
                    "is_follow",
                    "is_comment",
                    "is_forward",
                    "is_hate",
                    "play_time_ms",
                    "profile_stay_time",
                    "comment_stay_time",
                    "is_profile_enter",
                    "verified_predictions",
                ),
                expected_effect=(
                    "Capture causal preference and temporal drift while reducing "
                    "ranker variance through complementary residual ordering."
                ),
                falsifier=(
                    "Any self/future leakage, invalid residual fit, no trusted "
                    "full-fidelity gain beyond epsilon, or gain that disappears "
                    "on the later temporal arm or slice checks."
                ),
                prohibition_conditions=(
                    "rolling_feedback_mode_undeclared",
                    "future_or_self_outcome_leakage",
                    "adaptive_validation_weight_search",
                    "test_label_or_hidden_feedback",
                ),
                implementation_targets=(
                    "solution/candidate.py",
                    "solution/features.py",
                    "solution/model.py",
                    "solution/train.py",
                    "solution/inference.py",
                ),
                sources=(
                    "ROLLING_BLEND_062_PLAYBOOK.md",
                    "EXPERIMENT_SUMMARY.md",
                    "PLAYBOOK.md",
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
            MethodCard(
                method_id="evaluation_random_exposure_robustness",
                family="evaluation",
                summary="Audit a frozen candidate on the random-exposure population.",
                tags=("random_exposure", "unbiased_evaluation", "robustness"),
                cost_tier="low",
                mechanism="Compute separate diagnostics without fitting on audit labels.",
                prerequisites=(
                    "random_exposure_log",
                    "standard_public_evaluation_complete",
                ),
                allowed_data=("random_exposure_log", "verified_predictions"),
                expected_effect="Expose gains dependent on the standard logging policy.",
                falsifier="Populations are incomparable or uncertainty is too large.",
                prohibition_conditions=("adaptive_tuning_on_audit_labels",),
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
