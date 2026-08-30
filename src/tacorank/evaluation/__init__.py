"""Deterministic evaluation, trust, and experiment-decision APIs."""

from .adapter import (
    ContractSpec,
    EvaluationInputs,
    EvaluationService,
    OutputGateEvidence,
    PredictionBatch,
    ProtectedEvaluatorAdapter,
)
from .decisions import DecisionContext, NoOpRecoveryRequired, decide
from .diagnostics import DiagnosticFeatures, compute_evaluation_diagnostics
from .types import (
    Decision,
    EvaluationResult,
    ExperimentDecision,
    Fidelity,
    Integrity,
    MetricSet,
    Population,
    PredictionChange,
    Stability,
    TrustAssessment,
    Verdict,
)

__all__ = [
    "ContractSpec",
    "Decision",
    "DecisionContext",
    "DiagnosticFeatures",
    "EvaluationInputs",
    "EvaluationResult",
    "EvaluationService",
    "ExperimentDecision",
    "Fidelity",
    "Integrity",
    "MetricSet",
    "NoOpRecoveryRequired",
    "OutputGateEvidence",
    "Population",
    "PredictionBatch",
    "PredictionChange",
    "ProtectedEvaluatorAdapter",
    "Stability",
    "TrustAssessment",
    "Verdict",
    "compute_evaluation_diagnostics",
    "decide",
]
