"""Person 1 research-planning primitives."""

from .duplicate_detection import DuplicateDetector, compute_duplicate_key
from .agent_tools import ResearchToolRegistry
from .eda import PlannerEdaError, PlannerEdaToolbox
from .graph_view import ExperimentNodeView, GraphView
from .linucb import LinUCBLegalChoiceRanker
from .literature import (
    LiteratureResearchError,
    LiteratureResearchSkill,
    LiteratureSearchResult,
    OpenAlexLiteratureSkill,
)
from .convergence_advisor import ConvergenceAdvice, ConvergenceAdvisor
from .plan_validation import PlanValidator, ValidationResult
from .portfolio import ExperimentPortfolio, MethodCard, default_portfolio
from .search_policy import PolicyChoice, SearchPolicy
from .search_eligibility import (
    PruneDisposition,
    SearchEligibility,
    classify_search_eligibility,
)

__all__ = [
    "DuplicateDetector",
    "ResearchToolRegistry",
    "ConvergenceAdvice",
    "ConvergenceAdvisor",
    "ExperimentNodeView",
    "ExperimentPortfolio",
    "GraphView",
    "LinUCBLegalChoiceRanker",
    "LiteratureResearchError",
    "LiteratureResearchSkill",
    "LiteratureSearchResult",
    "MethodCard",
    "PlannerEdaError",
    "PlannerEdaToolbox",
    "PlanValidator",
    "PolicyChoice",
    "PruneDisposition",
    "SearchEligibility",
    "SearchPolicy",
    "OpenAlexLiteratureSkill",
    "ValidationResult",
    "compute_duplicate_key",
    "classify_search_eligibility",
    "default_portfolio",
]
