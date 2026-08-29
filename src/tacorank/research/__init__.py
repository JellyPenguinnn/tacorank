"""Person 1 research-planning primitives."""

from .duplicate_detection import DuplicateDetector, compute_duplicate_key
from .graph_view import ExperimentNodeView, GraphView
from .convergence_advisor import ConvergenceAdvice, ConvergenceAdvisor
from .plan_validation import PlanValidator, ValidationResult
from .portfolio import ExperimentPortfolio, MethodCard, default_portfolio
from .search_policy import PolicyChoice, SearchPolicy

__all__ = [
    "DuplicateDetector",
    "ConvergenceAdvice",
    "ConvergenceAdvisor",
    "ExperimentNodeView",
    "ExperimentPortfolio",
    "GraphView",
    "MethodCard",
    "PlanValidator",
    "PolicyChoice",
    "SearchPolicy",
    "ValidationResult",
    "compute_duplicate_key",
    "default_portfolio",
]
