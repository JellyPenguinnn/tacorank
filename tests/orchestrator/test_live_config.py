from pathlib import Path
from types import SimpleNamespace

import pytest

from tacorank.config import ContractError
from tacorank.evaluation.adapter import ordered_row_identity_sha256
from tacorank.orchestrator.live import (
    _PopulationData,
    _load_population,
    _load_training_profiles,
    _mount_policies,
    _protected_population_manifests,
    _trae_config_from_mapping,
)
from tacorank.schemas import Population


def test_mount_policies_expose_pipeline_roots_at_canonical_paths(
    tmp_path: Path,
) -> None:
    contract_root = tmp_path / "contract"
    contract_root.mkdir()
    input_roots = {}
    command_ids = (
        "baseline_full",
        "candidate_smoke",
        "candidate_proxy",
        "candidate_full",
        "candidate_final_infer",
        "clean_reproduce",
    )
    for command_id in command_ids:
        input_root = tmp_path / command_id
        input_root.mkdir()
        input_roots[command_id] = input_root

    policies = _mount_policies(
        SimpleNamespace(data_manifest_sha256="a" * 64),
        SimpleNamespace(contract_root=contract_root, input_roots=input_roots),
    )

    assert {policy.command_id for policy in policies} == set(command_ids)
    for policy in policies:
        assert tuple(mount.target for mount in policy.mounts) == (
            "/contracts",
            "/inputs",
        )
        assert policy.mounts[0].source == contract_root.resolve()
        assert policy.mounts[1].source == input_roots[policy.command_id].resolve()


def test_trae_config_normalizes_json_path_and_tuple_values(tmp_path: Path) -> None:
    executable = tmp_path / "trae-cli"
    config_file = tmp_path / "trae.yaml"
    docker = tmp_path / "docker"
    install = tmp_path / "install"
    runtime = install / "site-packages"
    direct_url = runtime / "trae_agent.dist-info" / "direct_url.json"
    dotenv = runtime / "python_dotenv.dist-info" / "METADATA"

    config = _trae_config_from_mapping(
        {
            "command_prefix": [str(executable)],
            "trae_version": "0.1.0",
            "provider": "openai",
            "model_id": "deepseek-v4-flash",
            "config_file": str(config_file),
            "config_sha256": "a" * 64,
            "max_steps_cap": 20,
            "max_token_cap": 4096,
            "max_wall_time_seconds_cap": 900,
            "repair_step_limit": 12,
            "repair_token_limit": 2048,
            "repair_wall_time_limit_seconds": 600,
            "repair_allowed_command_ids": ["candidate_smoke"],
            "approved_environment_names": ["DEEPSEEK_API_KEY"],
            "credential_environment_names": ["DEEPSEEK_API_KEY"],
            "credential_environment_aliases": [
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
            ],
            "trae_install_root": str(install),
            "trae_install_identity_file": str(direct_url),
            "trae_runtime_root": str(runtime),
            "python_dotenv_metadata_file": str(dotenv),
            "docker_executable": str(docker),
        }
    )

    assert config.command_prefix == (str(executable),)
    assert config.config_file == config_file
    assert config.docker_executable == docker
    assert config.trae_install_root == install
    assert config.trae_install_identity_file == direct_url
    assert config.trae_runtime_root == runtime
    assert config.python_dotenv_metadata_file == dotenv
    assert config.repair_allowed_command_ids == ("candidate_smoke",)
    assert config.credential_environment_aliases == (
        ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    )


def test_protected_population_manifests_bind_proxy_and_full_row_identity() -> None:
    proxy = _PopulationData(
        rows=(
            {"row_id": 0, "user_id": "user-1", "video_id": "video-1"},
            {"row_id": 1, "user_id": "user-2", "video_id": "video-2"},
        ),
        labels=(1, 0),
    )
    full = _PopulationData(
        rows=(
            {"row_id": 0, "user_id": "user-3", "video_id": "video-3"},
        ),
        labels=(1,),
    )

    manifests = _protected_population_manifests(
        {"smoke": proxy, "proxy": proxy, "full": full}
    )

    assert set(manifests) == {
        Population.INTERNAL_PROXY,
        Population.PUBLIC_VALIDATION,
    }
    assert manifests[Population.INTERNAL_PROXY].rows == 2
    assert manifests[
        Population.INTERNAL_PROXY
    ].ordered_row_identity_sha256 == ordered_row_identity_sha256(
        (0, 1), ("user-1", "user-2"), ("video-1", "video-2")
    )
    assert manifests[Population.PUBLIC_VALIDATION].rows == 1
    assert manifests[
        Population.PUBLIC_VALIDATION
    ].ordered_row_identity_sha256 == ordered_row_identity_sha256(
        (0,), ("user-3",), ("video-3",)
    )


def test_population_loader_binds_manifest_attested_diagnostic_features(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.csv"
    train.write_text(
        "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
        "20220420,u1,v1,a1,1,5000,1\n"
        "20220421,u1,v2,a2,1,9000,0\n"
        "20220421,u2,v2,a2,1,9000,1\n",
        encoding="utf-8",
    )
    population = tmp_path / "population.csv"
    population.write_text(
        "row_id,user_id,video_id,label\n"
        "0,u1,v1,1\n"
        "1,u2,v3,0\n",
        encoding="utf-8",
    )
    score = tmp_path / "score.csv"
    score.write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220422,u1,v1,a1,1,5000\n"
        "1,20220423,u2,v3,a3,1,20000\n",
        encoding="utf-8",
    )

    user_history, item_popularity = _load_training_profiles(train)
    loaded = _load_population(
        population,
        feature_path=score,
        user_history=user_history,
        item_popularity=item_popularity,
    )

    assert loaded.diagnostic_features is not None
    assert loaded.diagnostic_features.dates == (20220422, 20220423)
    assert loaded.diagnostic_features.user_history_count == (2, 1)
    assert loaded.diagnostic_features.item_popularity == (1, 0)
    assert len(loaded.diagnostic_features.validation_arms) == 2


def test_population_loader_rejects_misaligned_diagnostic_rows(tmp_path: Path) -> None:
    population = tmp_path / "population.csv"
    population.write_text(
        "row_id,user_id,video_id,label\n0,u1,v1,1\n", encoding="utf-8"
    )
    score = tmp_path / "score.csv"
    score.write_text(
        "row_id,date,user_id,video_id,author_id,tab,duration_ms\n"
        "0,20220422,other,v1,a1,1,5000\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="population order"):
        _load_population(
            population,
            feature_path=score,
            user_history={"u1": 1},
            item_popularity={"v1": 1},
        )
