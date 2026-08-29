"""KuaiRand-Pure benchmark adapters."""

from .evaluator_adapter import (
    create_evaluator_adapter,
    kuairand_contract,
    published_reference_scores,
)
from .submission_adapter import KuaiRandSubmissionAdapter, validate_submission

__all__ = [
    "KuaiRandSubmissionAdapter",
    "create_evaluator_adapter",
    "kuairand_contract",
    "published_reference_scores",
    "validate_submission",
]
