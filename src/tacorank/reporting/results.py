"""Judge-facing deterministic Markdown result projections."""

from collections import Counter
from typing import Mapping, Sequence

from tacorank.evaluation.comparisons import normalized_headroom
from tacorank.evaluation.types import EvaluationResult

from .resources import ResourceSummary


def render_metric_table(
    named_results: Mapping[str, EvaluationResult],
) -> str:
    if not named_results:
        return ""
    metric_names = sorted(next(iter(named_results.values())).metric_set.metrics)
    header = "| Candidate | " + " | ".join(metric_names) + " | primary |"
    rule = "| --- | " + " | ".join("---:" for _ in metric_names) + " | ---: |"
    rows = [header, rule]
    for name, result in named_results.items():
        values = ["%.6f" % result.metric_set.metrics[metric] for metric in metric_names]
        rows.append(
            "| %s | %s | %.6f |"
            % (name, " | ".join(values), result.metric_set.primary_score)
        )
    return "\n".join(rows)


def render_summary(
    run_id: str,
    final_result: EvaluationResult,
    baseline_primary: float,
    oracle_primary: float,
    resources: ResourceSummary,
    verdicts: Sequence[str],
    experiments_used: int,
    experiment_limit: int,
    public_queries: int,
    limitations: Sequence[str],
) -> str:
    score = final_result.metric_set.primary_score
    headroom = 100.0 * normalized_headroom(score, baseline_primary, oracle_primary)
    census = Counter(verdicts)
    census_text = " | ".join(
        "%s %d" % item for item in sorted(census.items())
    ) or "none"
    metric_text = " | ".join(
        "%s %.6f" % item for item in sorted(final_result.metric_set.metrics.items())
    )
    limitation_text = "\n".join("- %s" % value for value in limitations) or "- None recorded."
    return "\n".join(
        [
            "# Run Summary - %s" % run_id,
            "",
            "## Result",
            "",
            "primary %.6f (baseline %.6f, delta %+.6f, headroom captured %.2f%%)"
            % (score, baseline_primary, score - baseline_primary, headroom),
            metric_text,
            "",
            "## Resource",
            "",
            "provider tokens: %d in / %d out; estimated tokens: %d in / %d out"
            % (
                resources.llm_input_tokens_provider,
                resources.llm_output_tokens_provider,
                resources.llm_input_tokens_estimated,
                resources.llm_output_tokens_estimated,
            ),
            "action wall time: %.1fs; GPU-hours: %.4f; manual interventions: %d"
            % (
                resources.action_wall_time_ms / 1000.0,
                resources.gpu_hours,
                resources.manual_interventions,
            ),
            "experiments: %d/%d; public validation queries: %d"
            % (experiments_used, experiment_limit, public_queries),
            "",
            "## Verdict Census",
            "",
            census_text,
            "",
            "## Limitations",
            "",
            limitation_text,
        ]
    )
