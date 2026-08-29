import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from tacorank.evaluation.types import (
    EvaluationResult as DomainEvaluationResult,
    MetricDelta,
    MetricSet as DomainMetricSet,
    PredictionChange as DomainPredictionChange,
    TrustAssessment as DomainTrustAssessment,
)
from tacorank.schemas import (
    EVENT_PAYLOAD_MODELS,
    ArtifactKind,
    ArtifactRef,
    EvaluationCompletedPayload,
    EvaluationResult,
    Event,
    EventType,
    ExperimentFamily,
    ExperimentKind,
    Fidelity,
    Integrity,
    MetricSet,
    PlannerAction,
    PlannerOutput,
    Population,
    ResourceDelta,
    Stability,
    TrustVerdict,
    family_kind,
    parse_event_json,
    parse_method_card_markdown,
    validate_competition_contract_markdown,
)


RUN_ID = "run_20260829_a"
EXP_ID = "exp_0001"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
COMMIT = "d" * 40


def metric_set():
    return MetricSet(
        metrics={"gauc": 0.67, "ndcg_at_5": 0.55},
        primary_metric_name="primary",
        primary_score=0.61,
    )


def run_started_values():
    payload = {
        "type": "run.started",
        "config_sha256": HASH_C,
        "contract_sha256": HASH_A,
        "protected_paths_sha256": HASH_B,
        "max_experiments": 50,
        "wall_time_limit_seconds": 21600,
        "seed_schedule": [0, 1, 2],
    }
    input_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "seq": 1,
        "event_id": "evt_000001",
        "timestamp": "2026-08-29T03:14:15.123Z",
        "run_id": RUN_ID,
        "event_type": "run.started",
        "causation_event_id": None,
        "idempotency_key": "%s:run:start:0:%s" % (RUN_ID, input_hash),
        "payload": payload,
        "artifact_refs": [],
        "resource_delta": ResourceDelta().model_dump(mode="json"),
        "prev_event_hash": "0" * 64,
    }


class SharedSchemaTests(unittest.TestCase):
    def test_person5_extensions_are_canonical(self):
        self.assertEqual(set(EVENT_PAYLOAD_MODELS), set(EventType))
        self.assertEqual(Population.UNBIASED_AUDIT.value, "unbiased_audit")
        self.assertEqual(TrustVerdict.REDUNDANT.value, "redundant")
        self.assertEqual(ArtifactKind.DELTA_VECTOR.value, "delta_vector")
        self.assertEqual(family_kind(ExperimentFamily.OBJECTIVE), ExperimentKind.FRAME)
        self.assertEqual(family_kind(ExperimentFamily.MODEL), ExperimentKind.CAPACITY)

    def test_models_reject_extra_fields_and_nonfinite_metrics(self):
        with self.assertRaises(ValidationError):
            MetricSet(
                metrics={"gauc": float("nan")},
                primary_metric_name="primary",
                primary_score=0.5,
            )
        with self.assertRaises(ValidationError):
            ResourceDelta(unexpected=1)
        with self.assertRaises(ValidationError):
            ResourceDelta(llm_input_tokens=-1)

    def test_metric_contract_registry_is_enforced(self):
        metrics = metric_set()
        metrics.validate_contract(("gauc", "ndcg_at_5"), expected_primary=0.61)
        with self.assertRaisesRegex(ValueError, "undeclared"):
            MetricSet(
                metrics={"gauc": 0.6, "ndcg_at_5": 0.5, "secret": 1.0},
                primary_metric_name="primary",
                primary_score=0.55,
            ).validate_contract(("gauc", "ndcg_at_5"))

    def test_artifact_paths_and_bytes_are_verified(self):
        with self.assertRaises(ValidationError):
            ArtifactRef(
                artifact_id="art_0123abcd",
                kind="metrics",
                path="../outside.json",
                sha256=HASH_A,
                size_bytes=1,
                content_type="application/json",
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts" / "run" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"{}")
            ref = ArtifactRef(
                artifact_id="art_0123abcd",
                kind="metrics",
                path="artifacts/run/metrics.json",
                sha256=hashlib.sha256(b"{}").hexdigest(),
                size_bytes=2,
                content_type="application/json",
            )
            self.assertEqual(ref.verify_file(root), path.resolve())
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size"):
                ref.verify_file(root)

    def test_planner_output_discriminator(self):
        with self.assertRaises(ValidationError):
            PlannerOutput(
                action=PlannerAction.PROPOSE,
                spec=None,
                reason_code="READY",
                reason="A legal experiment exists.",
                supporting_event_ids=[],
            )
        blocked = PlannerOutput(
            action="blocked",
            spec=None,
            reason_code="NO_METHOD",
            reason="No method currently satisfies prerequisites.",
            supporting_event_ids=["evt_000003"],
        )
        self.assertIsNone(blocked.spec)

    def test_unbiased_audit_is_isolated(self):
        result = EvaluationResult(
            run_id=RUN_ID,
            experiment_id=EXP_ID,
            attempt=1,
            population="unbiased_audit",
            fidelity="full",
            seed=0,
            public_query_index=None,
            evaluator_sha256=HASH_A,
            contract_sha256=HASH_B,
            metric_set=metric_set(),
            baseline_delta=0.01,
            parent_delta=0.002,
            previous_best_delta=0.001,
            prediction_change={
                "spearman_vs_parent": 0.9,
                "changed_row_fraction": 0.5,
            },
            trust={
                "verdict": "accepted",
                "stability": "not_applicable",
                "integrity": "clean",
                "flags": [],
            },
        )
        payload = EvaluationCompletedPayload(result=result)
        self.assertEqual(payload.result.population, Population.UNBIASED_AUDIT)
        values = payload.model_dump()
        values["result"]["public_query_index"] = 1
        with self.assertRaises(ValidationError):
            EvaluationCompletedPayload.model_validate(values)

    def test_confirmed_seed_count_matches_evidence_events(self):
        result = EvaluationResult(
            run_id=RUN_ID,
            experiment_id=EXP_ID,
            attempt=3,
            population="public_validation",
            fidelity="full",
            seed=3,
            public_query_index=4,
            evaluator_sha256=HASH_A,
            contract_sha256=HASH_B,
            metric_set=metric_set(),
            baseline_delta=0.01,
            parent_delta=0.002,
            previous_best_delta=0.001,
            prediction_change={
                "spearman_vs_parent": 0.9,
                "changed_row_fraction": 0.5,
            },
            trust={
                "verdict": "accepted",
                "stability": "confirmed",
                "integrity": "clean",
                "flags": [],
                "eta_applied": 0.0016,
                "seed_mean": 0.61,
                "seed_stderr": 0.0002,
                "seed_count": 3,
            },
            seed_evidence_event_ids=["evt_000010", "evt_000011"],
        )
        payload = EvaluationCompletedPayload(result=result)
        self.assertEqual(payload.result.trust.seed_count, 3)
        values = payload.model_dump()
        values["result"]["seed_evidence_event_ids"] = ["evt_000010"]
        with self.assertRaisesRegex(ValidationError, "seed_count"):
            EvaluationCompletedPayload.model_validate(values)

    def test_event_discriminates_payload_and_round_trips_hash(self):
        event = Event.create(**run_started_values())
        self.assertEqual(event.event_type.value, "run.started")
        event.assert_hash_valid()
        parsed = parse_event_json(event.model_dump_json())
        self.assertEqual(parsed, event)
        self.assertEqual(parsed.payload.contract_sha256, HASH_A)

    def test_event_rejects_wrong_identity_and_payload(self):
        values = run_started_values()
        values["event_id"] = "evt_000002"
        with self.assertRaises(ValidationError):
            Event.create(**values)
        values = run_started_values()
        values["payload"] = {"actor": "x"}
        with self.assertRaises(ValidationError):
            Event.create(**values)

    def test_contract_and_method_card_markdown(self):
        headings = [
            "# Competition Contract",
            "## Identity and source precedence",
            "## Required benchmark",
            "## Data and temporal boundary",
            "## Target label and permitted inputs",
            "## Metrics and primary aggregation",
            "## Official baseline",
            "## Convergence and resource limits",
            "## Editable and protected paths",
            "## Allowed commands",
            "## Evaluation isolation",
            "## Submission schema",
            "## Resolved ambiguities",
            "## Human approvals",
        ]
        validate_competition_contract_markdown("\n\nDetails\n\n".join(headings))
        with self.assertRaises(ValueError):
            validate_competition_contract_markdown("\n".join(reversed(headings)))

        metadata = (
            '{"schema_version":"1.0","method_id":"method_pairwise_bpr",'
            '"family":"objective","status":"candidate","tags":["pairwise"],'
            '"cost_tier":"medium","sources":["https://example.org/paper"]}'
        )
        sections = [
            "## Mechanism",
            "## Preconditions",
            "## Allowed data",
            "## Expected effect",
            "## Falsification condition",
            "## Do not use when",
            "## Minimal implementation",
            "## Sources",
        ]
        card = "```json\n%s\n```\n\n%s" % (metadata, "\ntext\n".join(sections))
        self.assertEqual(parse_method_card_markdown(card).family, ExperimentFamily.OBJECTIVE)

    def test_person5_result_converts_to_canonical_transport(self):
        domain = DomainEvaluationResult(
            RUN_ID,
            EXP_ID,
            1,
            Population.PUBLIC_VALIDATION,
            Fidelity.FULL,
            0,
            1,
            HASH_A,
            HASH_B,
            HASH_C,
            DomainMetricSet({"gauc": 0.67, "ndcg_at_5": 0.55}, "primary", 0.61),
            MetricDelta(0.01, {"gauc": 0.01, "ndcg_at_5": 0.01}),
            MetricDelta(0.005, {"gauc": 0.006, "ndcg_at_5": 0.004}),
            MetricDelta(0.003, {"gauc": 0.004, "ndcg_at_5": 0.002}),
            DomainPredictionChange(0.8, 0.5, 0.2, 0.9),
            DomainTrustAssessment(
                TrustVerdict.ACCEPTED,
                Stability.CONFIRMED,
                Integrity.CLEAN,
                eta_applied=0.0016,
                seed_mean=0.61,
                seed_stderr=0.0002,
                seed_count=3,
            ),
            seed_evidence_event_ids=("evt_000010", "evt_000011"),
        )
        canonical = domain.to_canonical()
        self.assertIsInstance(canonical, EvaluationResult)
        self.assertEqual(canonical.parent_delta, 0.005)
        self.assertEqual(canonical.trust.verdict, TrustVerdict.ACCEPTED)
        self.assertEqual(
            canonical.seed_evidence_event_ids,
            ["evt_000010", "evt_000011"],
        )
        self.assertEqual(canonical.trust.seed_count, 3)
        self.assertEqual(canonical.trust.seed_mean, 0.61)


if __name__ == "__main__":
    unittest.main()
