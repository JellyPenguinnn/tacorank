from types import SimpleNamespace

from tacorank.research.graph_view import GraphView

from .conftest import make_summary


def test_graph_view_reconstructs_lineage_and_eligibility():
    root = make_summary("exp_0000")
    child = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        commit_sha="b" * 40,
        family="objective",
        score=0.601,
        parent_eligible=True,
    )
    context = SimpleNamespace(
        baseline=root,
        current_best=child,
        eligible_frontier=[root, child],
        family_history=[child],
    )

    graph = GraphView.from_context(context)

    assert graph.get("exp_0001").parent_experiment_id == "exp_0000"
    assert [node.experiment_id for node in graph.children_of("exp_0000")] == ["exp_0001"]
    assert [node.experiment_id for node in graph.ancestors_of("exp_0001")] == ["exp_0000"]
    assert {node.experiment_id for node in graph.eligible_parents()} == {"exp_0000", "exp_0001"}


def test_ineligible_proxy_and_retracted_nodes_are_excluded():
    root = make_summary("exp_0000")
    proxy = make_summary(
        "exp_0002",
        parent_experiment_id="exp_0000",
        fidelity="proxy",
        parent_eligible=None,
    )
    retracted = make_summary(
        "exp_0003",
        parent_experiment_id="exp_0000",
        fidelity="full",
        decision="reject",
        parent_eligible=True,
    )
    retracted.status = "retracted"
    context = SimpleNamespace(
        baseline=root,
        current_best=root,
        eligible_frontier=[root, proxy, retracted],
        family_history=[proxy, retracted],
    )

    graph = GraphView.from_context(context)

    assert {node.experiment_id for node in graph.eligible_parents()} == {"exp_0000"}


def test_graph_view_accepts_memory_schema_node_names():
    node = SimpleNamespace(
        experiment_id="exp_0004",
        parent_experiment_id="exp_0000",
        base_commit_sha="c" * 40,
        latest_patch_commit_sha="d" * 40,
        highest_fidelity_completed="full",
        metric_set=SimpleNamespace(primary_score=0.62),
        trust=SimpleNamespace(verdict="accepted", integrity="clean"),
        decision="accept",
        status="accepted",
        parent_eligible=True,
    )

    graph = GraphView.from_context(SimpleNamespace(eligible_frontier=[node]))
    projected = graph.get("exp_0004")

    assert projected.parent_commit_sha == "d" * 40
    assert projected.highest_completed_fidelity == "full"
    assert projected.primary_score == 0.62
    assert projected.is_parent_eligible


def test_confirmed_inconclusive_result_can_be_explicit_exploratory_parent():
    root = make_summary("exp_0000", score=0.601468756352959)
    exploratory = make_summary(
        "exp_0001",
        parent_experiment_id="exp_0000",
        score=0.6013,
        decision="accept",
        parent_eligible=True,
        trust_verdict="inconclusive",
        stability="confirmed",
        integrity="clean",
    )
    context = SimpleNamespace(
        baseline=root,
        current_best=root,
        eligible_frontier=[root, exploratory],
        family_history=[exploratory],
    )

    graph = GraphView.from_context(context)

    assert graph.get("exp_0001").is_exploratory_parent
    assert {node.experiment_id for node in graph.eligible_parents()} == {
        "exp_0000",
        "exp_0001",
    }

    exploratory.parent_eligible = None
    unapproved = GraphView.from_context(
        SimpleNamespace(eligible_frontier=[root, exploratory])
    )
    assert {node.experiment_id for node in unapproved.eligible_parents()} == {
        "exp_0000"
    }


def test_explicit_empty_frontier_is_authoritative():
    root = make_summary("exp_0000")
    context = SimpleNamespace(
        baseline=root,
        current_best=root,
        eligible_frontier=[],
        family_history=[root],
    )

    graph = GraphView.from_context(context)

    assert graph.nodes() == ()
    assert graph.eligible_parents() == ()
