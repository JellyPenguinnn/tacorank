from __future__ import annotations

import hashlib
from pathlib import Path

from tacorank.coding.standalone import (
    TraeStandaloneConfig,
    build_example_context,
    load_example_spec,
)
from tacorank.research.duplicate_detection import compute_duplicate_key


def test_checked_example_resolves_to_canonical_person2_handoff(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "contract").mkdir(parents=True)
    (repository / "contract/COMPETITION.md").write_text(
        "Contract status: FROZEN\n", encoding="utf-8"
    )
    (repository / "PROTECTED_PATHS.md").write_text(
        "# Protected\n\n- `contract/`\n- `runs/`\n", encoding="utf-8"
    )
    (repository / "research/methods").mkdir(parents=True)
    (repository / "research/methods/temporal_drift_past_only.md").write_text(
        "# Past-only method\n", encoding="utf-8"
    )
    (repository / "solution").mkdir()
    (repository / "artifacts").mkdir()
    (repository / "runs").mkdir()
    config = TraeStandaloneConfig(
        repository_root=repository,
        worktree_root=tmp_path / "worktrees",
        contract_path="contract/COMPETITION.md",
        protected_paths_path="PROTECTED_PATHS.md",
        artifact_roots=["artifacts", "runs"],
        editable_roots=["solution"],
        allowed_command_ids=["candidate_smoke"],
        target_interface_excerpts={"candidate": "def run(invocation) -> None"},
        coding_step_limit=20,
        coding_token_limit=None,
        coding_wall_time_limit_seconds=900,
        data_boundary_sha256="d" * 64,
        trae={},
    )
    example = Path(__file__).parents[2] / "examples/trae/experiment-spec.json"
    spec = load_example_spec(
        example,
        base_commit_sha="a" * 40,
        run_id="trae_test_001",
        experiment_id="exp_0001",
    )

    context = build_example_context(
        config,
        spec,
        contract_sha256=hashlib.sha256(
            (repository / "contract/COMPETITION.md").read_bytes()
        ).hexdigest(),
    )

    assert spec.duplicate_key == compute_duplicate_key(spec)
    assert context.experiment_spec == spec
    assert context.parent_commit_sha == "a" * 40
    assert context.editable_roots == ["solution"]
    assert context.selected_method_cards[0]["method_id"] == (
        "temporal_drift_past_only"
    )
    assert context.artifact.path.startswith("runs/trae_test_001/contexts/")
    context.artifact.verify_file(repository, ("artifacts", "runs"))
