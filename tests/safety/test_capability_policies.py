from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tacorank.safety import (
    CommandCapability,
    CommandPolicy,
    DataAccessPolicy,
    DataViewPolicy,
    ViolationCode,
    inspect_dependency_changes,
    inspect_secrets,
    inspect_source_capabilities,
)


@dataclass
class Resolved:
    command_id: str
    argv: Sequence[str]
    cwd: str
    environment: Mapping[str, str]
    network_enabled: bool
    shell: bool = False


def test_symbolic_command_policy_accepts_only_sealed_shape() -> None:
    policy = CommandPolicy(
        (CommandCapability("candidate_smoke", ("python", "-m", "solution.smoke")),)
    )
    accepted = Resolved(
        "candidate_smoke",
        ("python", "-m", "solution.smoke", "--seed", "7"),
        "solution",
        {},
        False,
    )
    assert policy.validate_resolved(accepted) == ()

    raw_shell = Resolved(
        "candidate_smoke", "python -m solution.smoke", "solution", {}, False, True
    )
    findings = policy.validate_resolved(raw_shell)
    assert {finding.code for finding in findings} == {ViolationCode.UNAPPROVED_COMMAND}


def test_source_scanner_detects_command_network_dependency_and_secret() -> None:
    source = (
        "import requests\n"
        "import subprocess\n"
        "subprocess.run(['whoami'])\n"
        "run_command('not_registered')\n"
    )
    findings = inspect_source_capabilities(
        source, "solution/model.py", allowed_command_ids=("candidate_smoke",)
    )
    codes = {finding.code for finding in findings}
    assert ViolationCode.UNAPPROVED_COMMAND in codes
    assert ViolationCode.UNAPPROVED_NETWORK in codes

    dependency_findings = inspect_dependency_changes(
        ("solution/model.py", "requirements.txt")
    )
    assert [finding.code for finding in dependency_findings] == [
        ViolationCode.DEPENDENCY_CHANGE
    ]

    secret_findings = inspect_secrets(
        'api_key = "abcdefghijklmnopqrstuvwxyz123456"', "solution/model.py"
    )
    assert [finding.code for finding in secret_findings] == [
        ViolationCode.SECRET_DETECTED
    ]
    assert "abcdefghijklmnopqrstuvwxyz" not in secret_findings[0].message


def test_data_policy_allows_training_target_but_blocks_protected_and_future() -> None:
    policy = DataAccessPolicy(
        views=(
            DataViewPolicy(
                "training",
                ("user_id", "training_target"),
                ("data/training",),
            ),
        ),
        protected_columns=("validation_target",),
        hidden_path_tokens=("hidden_labels",),
        future_column_patterns=(r"^future_",),
    )
    assert (
        policy.validate_access(
            "training",
            paths=("data/training/events.csv",),
            columns=("user_id", "training_target"),
        )
        == ()
    )

    findings = policy.validate_access(
        "training",
        paths=("protected/hidden_labels.csv",),
        columns=("validation_target", "future_watch_time"),
    )
    codes = {finding.code for finding in findings}
    assert ViolationCode.HIDDEN_LABEL_ACCESS in codes
    assert ViolationCode.FUTURE_INFORMATION_LEAKAGE in codes

    source_findings = policy.inspect_source(
        'columns = ["validation_target", "future_watch_time"]', "solution/model.py"
    )
    assert {finding.code for finding in source_findings} == codes

    assert policy.inspect_source(
        'label = int(row["training_target"])', "solution/model.py"
    ) == ()
    literal_findings = policy.inspect_source(
        'label = int(row["validation_target"])', "solution/model.py"
    )
    assert {finding.code for finding in literal_findings} == {
        ViolationCode.HIDDEN_LABEL_ACCESS
    }
