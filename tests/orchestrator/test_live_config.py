from pathlib import Path

from tacorank.orchestrator.live import _trae_config_from_mapping


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
            "model_id": "deepseek-v4-pro",
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
