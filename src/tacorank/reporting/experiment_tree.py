"""Static experiment-tree projection helpers."""

from typing import Mapping, Sequence


def render_mermaid(nodes: Sequence[Mapping[str, object]]) -> str:
    lines = ["flowchart TD"]
    for node in nodes:
        experiment_id = str(node["experiment_id"])
        verdict = str(node.get("verdict", "pending"))
        lines.append('  %s["%s: %s"]' % (experiment_id, experiment_id, verdict))
        parent = node.get("parent_experiment_id")
        if parent:
            lines.append("  %s --> %s" % (parent, experiment_id))
    return "\n".join(lines)
