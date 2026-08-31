"""Hash-bound, advisory literature bank for recommender-system planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schemas import LiteratureEvidence, ResourceDelta
from .graph_view import get_value
from .literature import LiteratureResearchError


EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_PAPER_COUNT = 70
EXPECTED_ORGANIZATION_COUNTS = {
    "bytedance": 22,
    "meta": 23,
    "kuaishou": 25,
}
ALLOWED_RELATIONSHIPS = {
    "company_authored",
    "company_coauthored",
    "company_deployed",
}
MAX_BANK_BYTES = 1024 * 1024
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,126}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")
RECORD_FIELDS = {
    "id",
    "paper_id",
    "title",
    "year",
    "venue",
    "authors",
    "authors_truncated",
    "organization",
    "relationship",
    "url",
    "summary",
    "topics",
    "priority",
    "prominence_basis",
}


METHOD_BANK_TOPICS = {
    "objective_direct_within_user_ranker": (
        "ranking",
        "objective",
        "collaborative_filtering",
        "feature_interaction",
        "evaluation",
    ),
    "objective_pairwise_bpr": (
        "ranking",
        "objective",
        "negative_feedback",
        "collaborative_filtering",
        "evaluation",
    ),
    "objective_pairwise_hinge_margin": (
        "objective",
        "ranking",
        "optimization",
        "automl",
        "negative_feedback",
    ),
    "objective_lambda_ndcg_surrogate": (
        "listwise",
        "ranking",
        "objective",
        "evaluation",
    ),
    "objective_listwise_user_softmax": (
        "listwise",
        "ranking",
        "reranking",
        "diversity",
        "objective",
    ),
    "objective_loss_aligned_features": (
        "objective",
        "ranking",
        "feature_interaction",
        "calibration",
        "automl",
    ),
    "temporal_history_compact": (
        "sequential",
        "user_modeling",
        "long_term",
        "realtime",
        "attention",
    ),
    "temporal_recency_weighted_ranker": (
        "sequential",
        "temporal_drift",
        "user_modeling",
        "attention",
    ),
    "temporal_hour_context": (
        "realtime",
        "context",
        "sequential",
        "cross_domain",
    ),
    "multitask_single_auxiliary": (
        "multitask",
        "multiobjective",
        "user_modeling",
        "long_term",
        "retention",
    ),
    "multitask_watch_time_auxiliary": (
        "watch_time",
        "multitask",
        "short_video",
        "regression",
    ),
    "multitask_negative_feedback_auxiliary": (
        "negative_feedback",
        "multitask",
        "multiobjective",
        "ranking",
    ),
    "duration_bias_censored_watch_time": (
        "watch_time",
        "duration_bias",
        "short_video",
        "debiasing",
        "regression",
    ),
    "features_author_affinity_past_only": (
        "creator_ecosystem",
        "user_modeling",
        "personalization",
        "graph",
        "context",
    ),
    "features_tab_context_residual": (
        "context",
        "cross_domain",
        "feature_interaction",
        "multi_domain",
        "personalization",
    ),
    "features_frequency_crosses": (
        "feature_interaction",
        "ctr",
        "cold_start",
        "ranking",
    ),
    "features_duration_context_interactions": (
        "watch_time",
        "duration_bias",
        "feature_interaction",
        "debiasing",
    ),
    "temporal_drift_past_only": (
        "temporal_drift",
        "streaming",
        "online_learning",
        "realtime",
        "sequential",
    ),
    "model_compact_ranker": (
        "ctr",
        "feature_interaction",
        "factorization_machine",
        "ranking",
        "efficiency",
    ),
    "model_field_aware_ranker": (
        "feature_interaction",
        "personalization",
        "embedding",
        "ranking",
    ),
    "sampling_deterministic_coverage": (
        "sampling",
        "exposure_bias",
        "negative_feedback",
        "debiasing",
        "exploration",
    ),
    "sampling_hard_negative_pairs": (
        "sampling",
        "negative_feedback",
        "debiasing",
        "training_data",
    ),
    "ensemble_causal_rolling_residual_blend": (
        "ensemble",
        "residual",
        "temporal_drift",
        "sequential",
        "user_modeling",
        "ranking",
        "evaluation",
    ),
    "ensemble_diverse_residual_candidate": (
        "ensemble",
        "diversity",
        "reranking",
        "multiobjective",
        "ranking",
    ),
    "ensemble_confirmed_members": (
        "ensemble",
        "ranking",
        "multiobjective",
        "robustness",
        "evaluation",
    ),
    "evaluation_random_exposure_robustness": (
        "evaluation",
        "exposure_bias",
        "dataset",
        "debiasing",
        "exploration",
    ),
}


def _required_text(record: Mapping[str, Any], field: str, *, limit: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LiteratureResearchError("paper bank %s must be non-empty text" % field)
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) > limit:
        raise LiteratureResearchError("paper bank %s exceeds its size limit" % field)
    return normalized


def _string_list(
    record: Mapping[str, Any],
    field: str,
    *,
    maximum: int,
    tags: bool = False,
) -> tuple[str, ...]:
    raw = record.get(field)
    if not isinstance(raw, list) or not raw or len(raw) > maximum:
        raise LiteratureResearchError(
            "paper bank %s must contain between one and %d values"
            % (field, maximum)
        )
    values = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise LiteratureResearchError(
                "paper bank %s values must be non-empty text" % field
            )
        normalized = value.strip()
        if len(normalized) > 120 or (tags and not TAG_PATTERN.fullmatch(normalized)):
            raise LiteratureResearchError(
                "paper bank %s contains an invalid value" % field
            )
        values.append(normalized)
    if len(values) != len(set(values)):
        raise LiteratureResearchError("paper bank %s values must be unique" % field)
    return tuple(values)


class PaperBankLiteratureSkill:
    """Select a small, method-relevant set from a frozen local bibliography."""

    def __init__(
        self,
        *,
        bank_path: Path,
        expected_sha256: str,
        max_papers: int = 6,
    ):
        if not 1 <= max_papers <= 8:
            raise ValueError("literature max_papers must be between one and eight")
        if not SHA256_PATTERN.fullmatch(expected_sha256):
            raise ValueError("paper bank hash must be lowercase sha256")
        self.bank_path = Path(bank_path)
        self.expected_sha256 = expected_sha256
        self.max_papers = max_papers
        self._wall_time_ms = 0
        self._papers = self._load()

    @property
    def requires_citation(self) -> bool:
        return False

    @property
    def resource_delta(self) -> ResourceDelta:
        return ResourceDelta(wall_time_ms=self._wall_time_ms)

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    @property
    def organization_counts(self) -> Mapping[str, int]:
        return dict(Counter(str(item["organization"]) for item in self._papers))

    def _load(self) -> tuple[Mapping[str, Any], ...]:
        path = self.bank_path
        if path.is_symlink() or not path.is_file():
            raise LiteratureResearchError(
                "paper bank must be a regular non-symlinked file"
            )
        size = path.stat().st_size
        if size <= 0 or size > MAX_BANK_BYTES:
            raise LiteratureResearchError("paper bank has an invalid size")
        raw_bytes = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha256 != self.expected_sha256:
            raise LiteratureResearchError("paper bank hash does not match run config")
        try:
            payload = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiteratureResearchError("paper bank is not valid UTF-8 JSON") from exc
        if not isinstance(payload, Mapping):
            raise LiteratureResearchError("paper bank envelope must be an object")
        if set(payload) != {
            "schema_version",
            "as_of",
            "description",
            "selection_policy",
            "papers",
        }:
            raise LiteratureResearchError("paper bank envelope fields are invalid")
        if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise LiteratureResearchError("paper bank schema version is unsupported")
        if not re.fullmatch(
            r"20[0-9]{2}-[01][0-9]-[0-3][0-9]",
            str(payload.get("as_of", "")),
        ):
            raise LiteratureResearchError("paper bank review date is invalid")
        _required_text(payload, "description", limit=1_000)
        selection = payload.get("selection_policy")
        records = payload.get("papers")
        if not isinstance(selection, Mapping) or not isinstance(records, list):
            raise LiteratureResearchError("paper bank envelope is incomplete")
        if set(selection) != {
            "paper_count",
            "organizations",
            "source_requirement",
            "use_policy",
        }:
            raise LiteratureResearchError("paper bank selection policy is invalid")
        _required_text(selection, "source_requirement", limit=1_000)
        _required_text(selection, "use_policy", limit=1_000)
        if selection.get("paper_count") != EXPECTED_PAPER_COUNT:
            raise LiteratureResearchError("paper bank declared count must be 70")
        if selection.get("organizations") != EXPECTED_ORGANIZATION_COUNTS:
            raise LiteratureResearchError(
                "paper bank declared organization balance is invalid"
            )
        if len(records) != EXPECTED_PAPER_COUNT:
            raise LiteratureResearchError("paper bank must contain exactly 70 papers")

        normalized = []
        seen_ids: set[str] = set()
        seen_paper_ids: set[str] = set()
        seen_titles: set[str] = set()
        for raw in records:
            if not isinstance(raw, Mapping):
                raise LiteratureResearchError("paper bank records must be objects")
            if set(raw) != RECORD_FIELDS:
                raise LiteratureResearchError("paper bank record fields are invalid")
            record_id = _required_text(raw, "id", limit=127)
            paper_id = _required_text(raw, "paper_id", limit=200)
            title = _required_text(raw, "title", limit=500)
            venue = _required_text(raw, "venue", limit=200)
            summary = _required_text(raw, "summary", limit=2_000)
            url = _required_text(raw, "url", limit=500)
            organization = _required_text(raw, "organization", limit=20)
            relationship = _required_text(raw, "relationship", limit=40)
            if not ID_PATTERN.fullmatch(record_id):
                raise LiteratureResearchError("paper bank record id is invalid")
            if not url.startswith("https://"):
                raise LiteratureResearchError("paper bank URLs must use HTTPS")
            if organization not in EXPECTED_ORGANIZATION_COUNTS:
                raise LiteratureResearchError("paper bank organization is invalid")
            if relationship not in ALLOWED_RELATIONSHIPS:
                raise LiteratureResearchError("paper bank relationship is invalid")
            if (
                record_id in seen_ids
                or paper_id.casefold() in seen_paper_ids
                or title.casefold() in seen_titles
            ):
                raise LiteratureResearchError("paper bank contains a duplicate paper")
            year = raw.get("year")
            priority = raw.get("priority")
            authors_truncated = raw.get("authors_truncated")
            if (
                not isinstance(year, int)
                or isinstance(year, bool)
                or not 1800 <= year <= 2100
            ):
                raise LiteratureResearchError("paper bank year is invalid")
            if (
                not isinstance(priority, int)
                or isinstance(priority, bool)
                or not 1 <= priority <= 3
            ):
                raise LiteratureResearchError("paper bank priority is invalid")
            if not isinstance(authors_truncated, bool):
                raise LiteratureResearchError("paper bank authors_truncated is invalid")
            authors = _string_list(raw, "authors", maximum=12)
            topics = _string_list(raw, "topics", maximum=12, tags=True)
            prominence = _string_list(
                raw, "prominence_basis", maximum=8, tags=True
            )
            normalized.append(
                {
                    "id": record_id,
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "authors": authors,
                    "authors_truncated": authors_truncated,
                    "organization": organization,
                    "relationship": relationship,
                    "url": url,
                    "summary": summary,
                    "topics": topics,
                    "priority": priority,
                    "prominence_basis": prominence,
                }
            )
            seen_ids.add(record_id)
            seen_paper_ids.add(paper_id.casefold())
            seen_titles.add(title.casefold())

        organization_counts = Counter(
            str(item["organization"]) for item in normalized
        )
        if dict(organization_counts) != EXPECTED_ORGANIZATION_COUNTS:
            raise LiteratureResearchError(
                "paper bank organization counts must be ByteDance 22, Meta 23, "
                "and Kuaishou 25"
            )
        return tuple(normalized)

    def preflight(self) -> None:
        # Re-read the file so a post-construction mutation cannot pass preflight.
        self._papers = self._load()

    @staticmethod
    def _method_id(policy_choice: Any) -> str:
        return str(get_value(policy_choice, "method_card_id", ""))

    def _rank(self, method_id: str) -> list[Mapping[str, Any]]:
        requested = METHOD_BANK_TOPICS.get(method_id, ("ranking", "evaluation"))
        topic_weight = {
            topic: len(requested) - index
            for index, topic in enumerate(requested)
        }

        def rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            topics = set(item["topics"])
            overlap = sum(topic_weight.get(topic, 0) for topic in topics)
            match_count = len(topics.intersection(requested))
            return (
                -match_count,
                -overlap,
                int(item["priority"]),
                -int(item["year"]),
                str(item["id"]),
            )

        matching = [
            item for item in self._papers if set(item["topics"]).intersection(requested)
        ]
        return sorted(matching or list(self._papers), key=rank_key)

    def _select(self, method_id: str) -> list[Mapping[str, Any]]:
        ranked = self._rank(method_id)
        per_organization_cap = max(1, math.ceil(self.max_papers / 2))
        selected = []
        counts: Counter[str] = Counter()
        for item in ranked:
            organization = str(item["organization"])
            if counts[organization] >= per_organization_cap:
                continue
            selected.append(item)
            counts[organization] += 1
            if len(selected) == self.max_papers:
                return selected
        for item in ranked:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) == self.max_papers:
                break
        return selected

    async def research(
        self, context: Any, policy_choice: Any
    ) -> Sequence[LiteratureEvidence]:
        del context
        started = time.monotonic()
        self._wall_time_ms = 0
        method_id = self._method_id(policy_choice)
        query = "paper-bank:%s" % (method_id or "ranking")
        try:
            return [
                LiteratureEvidence(
                    evidence_id="bank_" + str(item["id"]),
                    provider="paper_bank",
                    paper_id=str(item["paper_id"]),
                    title=str(item["title"]),
                    abstract=str(item["summary"]),
                    year=int(item["year"]),
                    authors=list(item["authors"]),
                    venue=str(item["venue"]),
                    citation_count=0,
                    influential_citation_count=0,
                    url=str(item["url"]),
                    query=query,
                    organization=str(item["organization"]),
                    relationship=str(item["relationship"]),
                    topics=list(item["topics"]),
                    priority=int(item["priority"]),
                    prominence_basis=list(item["prominence_basis"]),
                    authors_truncated=bool(item["authors_truncated"]),
                )
                for item in self._select(method_id)
            ]
        finally:
            self._wall_time_ms = max(
                0, int(round((time.monotonic() - started) * 1_000))
            )


__all__ = [
    "EXPECTED_ORGANIZATION_COUNTS",
    "EXPECTED_PAPER_COUNT",
    "METHOD_BANK_TOPICS",
    "PaperBankLiteratureSkill",
]
