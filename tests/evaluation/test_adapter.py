import hashlib
from pathlib import Path
import tempfile
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
    PopulationManifest,
    ProtectedEvaluatorAdapter,
    ordered_values_sha256,
    sha256_file,
)
from tacorank.evaluation.baseline import verify_metric_parity
from tacorank.evaluation.types import Population
from tacorank.evaluation.types import Fidelity, MetricSet


ROOT = Path(__file__).resolve().parents[2]


class ProtectedAdapterTests(unittest.TestCase):
    def test_official_evaluator_matches_independent_metrics(self):
        adapter = create_evaluator_adapter(ROOT)
        users = ["u1", "u1", "u1", "u2", "u2", "u3"]
        labels = [1, 0, 1, 0, 0, 1]
        scores = [0.4, 0.4, 0.2, 0.9, 0.1, 0.7]
        metric_set = adapter.score(
            users,
            labels,
            scores,
            adapter.expected_evaluator_sha256,
            adapter.expected_contract_sha256,
            Population.PUBLIC_VALIDATION,
        )
        parity = verify_metric_parity(metric_set, users, labels, scores)
        self.assertTrue(parity.passed)
        self.assertLess(parity.max_abs_deviation, 1e-12)

    def test_hash_mismatch_blocks_call(self):
        adapter = create_evaluator_adapter(ROOT)
        with self.assertRaisesRegex(EvaluationIntegrityError, "request evaluator"):
            adapter.score(
                ["u", "u"],
                [0, 1],
                [0.1, 0.9],
                "0" * 64,
                adapter.expected_contract_sha256,
                Population.PUBLIC_VALIDATION,
            )

    def test_population_manifest_checks_ordered_user_alignment(self):
        evaluator = ROOT / "kuairand-starter-kit" / "evaluate.py"
        contract = ROOT / "contract" / "COMPETITION.md"
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
                    2, ordered_values_sha256(["u", "u"])
                )
            },
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "alignment"):
            adapter.score(
                ["x", "x"],
                [0, 1],
                [0.1, 0.9],
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
        service = EvaluationService(adapter)
        metric = MetricSet({"GAUC": 0.5, "nDCG@5": 0.5}, "primary", 0.5)
        base = dict(
            run_id="run_x",
            experiment_id="exp_0001",
            attempt=1,
            output_checked_event_id="evt_000001",
            output_gate_accepted=True,
            population=Population.PUBLIC_VALIDATION,
            fidelity=Fidelity.FULL,
            seed=0,
            public_query_index=None,
            evaluator_sha256=adapter.expected_evaluator_sha256,
            contract_sha256=adapter.expected_contract_sha256,
            data_manifest_sha256=data_hash,
            user_ids=["u", "u"],
            labels=[0, 1],
            scores=[0.1, 0.9],
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

    def test_hidden_final_requires_stop_evidence(self):
        adapter = create_evaluator_adapter(ROOT)
        service = EvaluationService(adapter)
        metric = MetricSet({"GAUC": 0.5, "nDCG@5": 0.5}, "primary", 0.5)
        request = EvaluationInputs(
            run_id="run_x",
            experiment_id="exp_0001",
            attempt=1,
            output_checked_event_id="evt_000001",
            output_gate_accepted=True,
            population=Population.HIDDEN_FINAL,
            fidelity=Fidelity.FINAL,
            seed=0,
            public_query_index=None,
            evaluator_sha256=adapter.expected_evaluator_sha256,
            contract_sha256=adapter.expected_contract_sha256,
            data_manifest_sha256="d" * 64,
            user_ids=["u", "u"],
            labels=[0, 1],
            scores=[0.1, 0.9],
            baseline=metric,
            parent=metric,
            previous_best=metric,
        )
        with self.assertRaisesRegex(EvaluationIntegrityError, "run.stopped"):
            service.evaluate(request)


if __name__ == "__main__":
    unittest.main()
