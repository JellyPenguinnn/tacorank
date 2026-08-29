"""KuaiRand-Pure binding for TacoRank's protected evaluator adapter."""

from pathlib import Path
import json
from typing import Mapping, Optional

from tacorank.evaluation.adapter import (
    ContractSpec,
    PopulationManifest,
    ProtectedEvaluatorAdapter,
    sha256_file,
)
from tacorank.evaluation.types import Population


REQUIRED_METRICS = ("GAUC", "nDCG@5")
PRIMARY_METRIC_NAME = "primary"


def kuairand_contract() -> ContractSpec:
    return ContractSpec(
        required_metrics=REQUIRED_METRICS,
        primary_metric_name=PRIMARY_METRIC_NAME,
        primary_weights={"GAUC": 0.5, "nDCG@5": 0.5},
        metric_ranges={"GAUC": (0.0, 1.0), "nDCG@5": (0.0, 1.0)},
        aggregation_tolerance=1e-12,
    )


def create_evaluator_adapter(
    repository_root: Path,
    expected_evaluator_sha256: Optional[str] = None,
    expected_contract_sha256: Optional[str] = None,
    expected_data_manifest_sha256: Optional[str] = None,
    population_manifests: Optional[Mapping[Population, PopulationManifest]] = None,
) -> ProtectedEvaluatorAdapter:
    """Build an adapter pinned to the repository's official starter-kit files.

    Production callers should pass hashes frozen by ``contract.verified``.  The
    optional defaults are convenient for the pre-run contract-freeze command;
    they pin current bytes at construction and still detect later mutation.
    """
    root = Path(repository_root)
    evaluator_path = root / "kuairand-starter-kit" / "evaluate.py"
    contract_path = root / "contract" / "COMPETITION.md"
    evaluator_hash = expected_evaluator_sha256 or sha256_file(evaluator_path)
    contract_hash = expected_contract_sha256 or sha256_file(contract_path)
    return ProtectedEvaluatorAdapter(
        evaluator_path=evaluator_path,
        expected_evaluator_sha256=evaluator_hash,
        expected_contract_sha256=contract_hash,
        contract=kuairand_contract(),
        contract_path=contract_path,
        population_manifests=population_manifests,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
    )


def published_reference_scores(repository_root: Path) -> Mapping[tuple, float]:
    """Return the six published baseline primary scores keyed by model/population."""
    score_path = Path(repository_root) / "kuairand-starter-kit" / "baseline_scores.json"
    with score_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    output = {}
    for model in ("random", "item_popularity", "fm_official"):
        for split, population in (
            ("valid", "public_validation"),
            ("test", "hidden_reference"),
        ):
            output[(model, population)] = float(payload["scores"][model][split]["primary"])
    return output
