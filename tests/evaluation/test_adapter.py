import hashlib
import math
from dataclasses import replace
from pathlib import Path
import unittest

from benchmarks.kuairand_pure.evaluator_adapter import (
    create_evaluator_adapter,
    published_reference_scores,
)
from tacorank.evaluation.adapter import (
    ContractSpec,
    EvaluationInputs,
    EvaluationIntegrityError,
    EvaluationService,
    OutputGateEvidence,
    PopulationManifest,
    PredictionBatch,
    ProtectedEvaluatorAdapter,
    ordered_prediction_sha256,
    ordered_row_identity_sha256,
    sha256_file,
)
from tacorank.evaluation.baseline import verify_metric_parity
from tacorank.evaluation.types import Population
from tacorank.evaluation.types import Fidelity, MetricDelta, MetricSet


ROOT = Path(__file__).resolve().parents[2]


def prediction_batch(users, scores, items=None, artifact_id="artifact_predictions"):
    item_ids = list(items or ("v%d" % index for index in range(len(scores))))
    row_ids = list(range(len(scores)))
    artifact_sha256 = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
    return PredictionBatch(
        artifact_id,
        artifact_sha256,
        row_ids,
        list(users),
        item_ids,
        list(scores),
    )


def gate_evidence(predictions, population=Population.PUBLIC_VALIDATION):
    return OutputGateEvidence(
        event_id="evt_000001",
        accepted=True,
        prediction_artifact_id=predictions.artifact_id,
        prediction_artifact_sha256=predictions.artifact_sha256,
        population=population,
        ordered_row_identity_sha256=ordered_row_identity_sha256(
            predictions.row_ids,
            predictions.user_ids,
            predictions.item_ids,
        ),
        ordered_prediction_sha256=ordered_prediction_sha256(
            predictions.row_ids,
            predictions.user_ids,
            predictions.item_ids,
            predictions.scores,
        ),
    )


def evaluation_service(adapter, output_gate, seed_result_resolver=None):
    gates = {output_gate.event_id: output_gate}
    return EvaluationService(
        adapter,
        output_gate_resolver=gates.__getitem__,
        seed_result_resolver=seed_result_resolver,
    )


class FixedScoreAdapter:
    def __init__(self, primary_score):
        self.primary_score = primary_score

    def verify_data_manifest(self, unused_data_manifest_sha256):
        return None

    def score(self, unused_predictions, unused_labels, *unused_identity):
        return MetricSet(
            {"GAUC": self.primary_score, "nDCG@5": self.primary_score},
            "primary",
            self.primary_score,
        )


def fixed_evaluation_request(attempt, seed, seed_event_ids=()):
    predictions = prediction_batch(
        ["u", "u", "u"],
        [0.3, 0.2, 0.1],
        artifact_id="artifact_%d" % attempt,
    )
    metric = lambda score: MetricSet(
        {"GAUC": score, "nDCG@5": score}, "primary", score
    )
    return EvaluationInputs(
        run_id="run_x",
        experiment_id="exp_0001",
        attempt=attempt,
        output_gate=gate_evidence(predictions),
        predictions=predictions,
        population=Population.PUBLIC_VALIDATION,
        fidelity=Fidelity.FULL,
        seed=seed,
        public_query_index=attempt,
        evaluator_sha256="a" * 64,
        contract_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        labels=[0, 1, 0],
        baseline=metric(0.50),
        parent=metric(0.59),
        previous_best=metric(0.57),
        parent_scores=[0.1, 0.2, 0.3],
        seed_evaluation_event_ids=tuple(seed_event_ids),
    )


class ProtectedAdapterTests(unittest.TestCase):
    def test_official_evaluator_matches_independent_metrics(self):
        adapter = create_evaluator_adapter(ROOT)
        users = ["u1", "u1", "u1", "u2", "u2", "u3"]
        labels = [1, 0, 1, 0, 0, 1]
        scores = [0.4, 0.4, 0.2, 0.9, 0.1, 0.7]
        predictions = prediction_batch(users, scores)
        metric_set = adapter.score(
            predictions,
            labels,
            adapter.expected_evaluator_sha256,
            adapter.expected_contract_sha256,
            Population.PUBLIC_VALIDATION,
        )
        parity = verify_metric_parity(metric_set, users, labels, scores)
        self.assertTrue(parity.passed)
        self.assertLess(parity.max_abs_deviation, 1e-12)

    def test_hash_mismatch_blocks_call(self):
        adapter = create_evaluator_adapter(ROOT)
        predictions = prediction_batch(["u", "u"], [0.1, 0.9])
        with self.assertRaisesRegex(EvaluationIntegrityError, "request evaluator"):
            adapter.score(
                predictions,
                [0, 1],
                "0" * 64,
                adapter.expected_contract_sha256,
                Population.PUBLIC_VALIDATION,
            )

    def test_fractional_labels_are_rejected_before_integer_coercion(self):
        adapter = create_evaluator_adapter(ROOT)
        predictions = prediction_batch(["u", "u"], [0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "exact numeric binary"):
            adapter.score(
                predictions,
                [0.9, 0.1],
                adapter.expected_evaluator_sha256,
                adapter.expected_contract_sha256,
                Population.PUBLIC_VALIDATION,
            )

    def test_official_evaluator_runs_outside_parent_module_state(self):
        adapter = create_evaluator_adapter(ROOT)
        predictions = prediction_batch(["u", "u"], [0.9, 0.1])
        args = (
            predictions,
            [0, 1],
            adapter.expected_evaluator_sha256,
            adapter.expected_contract_sha256,
            Population.PUBLIC_VALIDATION,
        )
        expected = adapter.score(*args)
        original_log2 = math.log2
        try:
            math.log2 = lambda unused_value: 1.0
            isolated = adapter.score(*args)
        finally:
            math.log2 = original_log2
        self.assertEqual(isolated, expected)

    def test_population_manifest_checks_ordered_user_alignment(self):
        evaluator = ROOT / "kuairand-starter-kit" / "evaluate.py"
        contract = ROOT / "contract" / "COMPETITION.md"
        expected = prediction_batch(["u", "u"], [0.1, 0.9], ["v1", "v2"])
        adapter = ProtectedEvaluatorAdapter(
            evaluator,
            sha256_file(evaluator),
            sha256_file(contract),
            ContractSpec(
                ("GAUC", "nDCG@5"),
                "primary",
                {"GAUC": 0.5, "nDCG@5": 0.5},
            ),
            contract,
            {
                Population.PUBLIC_VALIDATION: PopulationManifest(
                    2,
                    ordered_row_identity_sha256(
                        expected.row_ids, expected.user_ids, expected.item_ids
                    ),
                )
            },
        )
        misaligned = prediction_batch(
            ["u", "u"], [0.1, 0.9], ["v2", "v1"], "misaligned"
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "alignment"):
            adapter.score(
                misaligned,
                [0, 1],
                adapter.expected_evaluator_sha256,
                adapter.expected_contract_sha256,
                Population.PUBLIC_VALIDATION,
            )

    def test_published_registry_contains_all_six_reference_scores(self):
        scores = published_reference_scores(ROOT)
        self.assertEqual(len(scores), 6)
        self.assertEqual(scores[("fm_official", "public_validation")], 0.6016)

    def test_service_enforces_route_query_and_data_manifest(self):
        data_hash = "d" * 64
        adapter = create_evaluator_adapter(ROOT, expected_data_manifest_sha256=data_hash)
        metric = MetricSet({"GAUC": 0.5, "nDCG@5": 0.5}, "primary", 0.5)
        predictions = prediction_batch(["u", "u"], [0.1, 0.9])
        gate = gate_evidence(predictions)
        service = evaluation_service(adapter, gate)
        base = dict(
            run_id="run_x",
            experiment_id="exp_0001",
            attempt=1,
            output_gate=gate,
            predictions=predictions,
            population=Population.PUBLIC_VALIDATION,
            fidelity=Fidelity.FULL,
            seed=0,
            public_query_index=None,
            evaluator_sha256=adapter.expected_evaluator_sha256,
            contract_sha256=adapter.expected_contract_sha256,
            data_manifest_sha256=data_hash,
            labels=[0, 1],
            baseline=metric,
            parent=metric,
            previous_best=metric,
        )
        with self.assertRaisesRegex(ValueError, "query index"):
            service.evaluate(EvaluationInputs(**base))
        base["public_query_index"] = 1
        base["data_manifest_sha256"] = "e" * 64
        with self.assertRaisesRegex(EvaluationIntegrityError, "data manifest"):
            service.evaluate(EvaluationInputs(**base))

    def test_service_binds_scores_to_gate_b_artifact_rows(self):
        checked = prediction_batch(
            ["u", "u"], [0.9, 0.1], ["v", "v"], "checked"
        )
        tampered = PredictionBatch(
            checked.artifact_id,
            checked.artifact_sha256,
            checked.row_ids,
            checked.user_ids,
            checked.item_ids,
            [0.1, 0.9],
        )
        metric = MetricSet({"GAUC": 0.5, "nDCG@5": 0.5}, "primary", 0.5)
        request = EvaluationInputs(
            run_id="run_x",
            experiment_id="exp_0001",
            attempt=1,
            output_gate=gate_evidence(checked),
            predictions=tampered,
            population=Population.PUBLIC_VALIDATION,
            fidelity=Fidelity.FULL,
            seed=0,
            public_query_index=1,
            evaluator_sha256="a" * 64,
            contract_sha256="b" * 64,
            data_manifest_sha256="c" * 64,
            labels=[1, 0],
            baseline=metric,
            parent=metric,
            previous_best=metric,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "prediction values"):
            evaluation_service(
                FixedScoreAdapter(0.5), request.output_gate
            ).evaluate(request)

    def test_service_resolves_gate_b_evidence_instead_of_trusting_request(self):
        request = fixed_evaluation_request(1, 11)
        fabricated_gate = replace(
            request.output_gate,
            ordered_prediction_sha256="f" * 64,
        )
        fabricated_request = replace(request, output_gate=fabricated_gate)
        verified = {request.output_gate.event_id: request.output_gate}
        service = EvaluationService(
            FixedScoreAdapter(0.60),
            output_gate_resolver=verified.__getitem__,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "verified event"):
            service.evaluate(fabricated_request)

    def test_seed_confirmation_resolves_events_and_includes_current_score(self):
        first_request = fixed_evaluation_request(1, 11)
        first = evaluation_service(
            FixedScoreAdapter(0.60), first_request.output_gate
        ).evaluate(first_request)
        second_request = fixed_evaluation_request(2, 22)
        second = evaluation_service(
            FixedScoreAdapter(0.60), second_request.output_gate
        ).evaluate(second_request)
        prior = {"evt_seed_1": first, "evt_seed_2": second}
        current_request = fixed_evaluation_request(3, 33, tuple(prior))
        service = evaluation_service(
            FixedScoreAdapter(0.58),
            current_request.output_gate,
            seed_result_resolver=prior.__getitem__,
        )
        result = service.evaluate(current_request)
        self.assertAlmostEqual(result.trust.seed_mean, (0.60 + 0.60 + 0.58) / 3)
        self.assertEqual(result.trust.seed_count, 3)
        self.assertEqual(result.trust.stability.value, "unstable")
        self.assertNotEqual(result.trust.verdict.value, "accepted")

    def test_seed_confirmation_rejects_incompatible_or_duplicate_evidence(self):
        prior_request = fixed_evaluation_request(1, 11)
        prior_result = evaluation_service(
            FixedScoreAdapter(0.60), prior_request.output_gate
        ).evaluate(prior_request)
        resolver = {"evt_seed_1": prior_result}.__getitem__
        duplicate_events = fixed_evaluation_request(
            3, 33, ("evt_seed_1", "evt_seed_1")
        )
        service = evaluation_service(
            FixedScoreAdapter(0.58),
            duplicate_events.output_gate,
            seed_result_resolver=resolver,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "unique"):
            service.evaluate(duplicate_events)
        duplicate_seed = fixed_evaluation_request(2, 11, ("evt_seed_1",))
        service = evaluation_service(
            FixedScoreAdapter(0.58),
            duplicate_seed.output_gate,
            seed_result_resolver=resolver,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "duplicate seed"):
            service.evaluate(duplicate_seed)

        inconsistent_parent = replace(
            prior_result,
            parent_delta=MetricDelta(
                prior_result.parent_delta.primary,
                {
                    **prior_result.parent_delta.metrics,
                    "GAUC": prior_result.parent_delta.metrics["GAUC"] + 0.01,
                },
            ),
        )
        incompatible_reference = fixed_evaluation_request(
            2, 22, ("evt_seed_1",)
        )
        service = evaluation_service(
            FixedScoreAdapter(0.58),
            incompatible_reference.output_gate,
            seed_result_resolver={"evt_seed_1": inconsistent_parent}.__getitem__,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "parent reference"):
            service.evaluate(incompatible_reference)

    def test_hidden_final_requires_stop_evidence(self):
        adapter = create_evaluator_adapter(ROOT)
        metric = MetricSet({"GAUC": 0.5, "nDCG@5": 0.5}, "primary", 0.5)
        predictions = prediction_batch(["u", "u"], [0.1, 0.9])
        gate = gate_evidence(predictions, Population.HIDDEN_FINAL)
        service = evaluation_service(adapter, gate)
        request = EvaluationInputs(
            run_id="run_x",
            experiment_id="exp_0001",
            attempt=1,
            output_gate=gate,
            predictions=predictions,
            population=Population.HIDDEN_FINAL,
            fidelity=Fidelity.FINAL,
            seed=0,
            public_query_index=None,
            evaluator_sha256=adapter.expected_evaluator_sha256,
            contract_sha256=adapter.expected_contract_sha256,
            data_manifest_sha256="d" * 64,
            labels=[0, 1],
            baseline=metric,
            parent=metric,
            previous_best=metric,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "run.stopped"):
            service.evaluate(request)


if __name__ == "__main__":
    unittest.main()
