"""Controller-owned adapters for reviewed baseline, candidate, and submission code.

The symbolic registry invokes this installed module, never a wrapper from the
Trae-editable ``solution/`` tree.  Only the explicitly registered implementation
entrypoint is imported from that tree.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)


class ControllerCLIError(RuntimeError):
    """A reviewed command integration or output violated its contract."""


@dataclass(frozen=True)
class PipelineInvocation:
    mode: str
    fidelity: str
    seed: Optional[int]
    output_path: Path
    contract_root: Path
    input_root: Path
    clean_reproduce: bool


@dataclass(frozen=True)
class SubmissionCheckInvocation:
    prediction_path: Path
    contract_root: Path
    artifact_root: Path


EntrypointLoader = Callable[[str], Callable[[Any], None]]


def execute_pipeline(
    argv: Sequence[str],
    *,
    environment: Optional[Mapping[str, str]] = None,
    entrypoint_loader: Optional[EntrypointLoader] = None,
) -> PipelineInvocation:
    arguments = _pipeline_parser().parse_args(tuple(argv))
    source_environment = dict(os.environ if environment is None else environment)
    mode = "baseline" if arguments.baseline else "candidate"
    if mode == "baseline":
        if arguments.fidelity is not None or arguments.seed is not None:
            raise ControllerCLIError(
                "baseline mode cannot accept candidate fidelity or seed settings"
            )
        fidelity = "full"
        entrypoint_name = "TACORANK_BASELINE_ENTRYPOINT"
    else:
        if arguments.fidelity is None or arguments.seed is None:
            raise ControllerCLIError("candidate mode requires fidelity and seed")
        fidelity = str(arguments.fidelity)
        entrypoint_name = "TACORANK_CANDIDATE_ENTRYPOINT"
    if arguments.clean_reproduce and (mode != "candidate" or fidelity != "full"):
        raise ControllerCLIError(
            "clean reproduction is defined only for a full candidate invocation"
        )

    contract_root = _required_directory(
        source_environment, "TACORANK_CONTRACT_ROOT"
    )
    input_root = _required_directory(source_environment, "TACORANK_INPUT_ROOT")
    artifact_root = _required_directory(
        source_environment, "TACORANK_ARTIFACT_ROOT"
    )
    output = _new_output_path(str(arguments.output), artifact_root)
    entrypoint = _required_entrypoint(source_environment, entrypoint_name)
    invocation = PipelineInvocation(
        mode=mode,
        fidelity=fidelity,
        seed=arguments.seed,
        output_path=output,
        contract_root=contract_root,
        input_root=input_root,
        clean_reproduce=bool(arguments.clean_reproduce),
    )
    implementation = (entrypoint_loader or _load_entrypoint)(entrypoint)
    if implementation(invocation) is not None:
        raise ControllerCLIError("pipeline entrypoint must return None")
    if output.is_symlink() or not output.is_file():
        raise ControllerCLIError("pipeline did not create the exact prediction output")
    if output.resolve(strict=True).parent != output.parent.resolve(strict=True):
        raise ControllerCLIError("prediction output resolved through an unsafe path")
    return invocation


def execute_submission_check(
    argv: Sequence[str],
    *,
    environment: Optional[Mapping[str, str]] = None,
    entrypoint_loader: Optional[EntrypointLoader] = None,
) -> SubmissionCheckInvocation:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("prediction_path")
    arguments = parser.parse_args(tuple(argv))
    source_environment = dict(os.environ if environment is None else environment)
    contract_root = _required_directory(
        source_environment, "TACORANK_CONTRACT_ROOT"
    )
    artifact_root = _required_directory(
        source_environment, "TACORANK_ARTIFACT_ROOT"
    )
    prediction = Path(arguments.prediction_path)
    if not prediction.is_absolute() or prediction.is_symlink() or not prediction.is_file():
        raise ControllerCLIError("prediction must be an absolute regular file")
    prediction = prediction.resolve(strict=True)
    expected_prediction = _required_file(
        source_environment, "TACORANK_VERIFIED_PREDICTION_PATH"
    )
    if prediction != expected_prediction:
        raise ControllerCLIError(
            "prediction differs from the controller-verified artifact"
        )
    entrypoint = _required_entrypoint(
        source_environment, "TACORANK_SUBMISSION_CHECK_ENTRYPOINT"
    )
    invocation = SubmissionCheckInvocation(prediction, contract_root, artifact_root)
    checker = (entrypoint_loader or _load_entrypoint)(entrypoint)
    if checker(invocation) is not None:
        raise ControllerCLIError("submission checker must return None")
    return invocation


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            raise ControllerCLIError("a reviewed controller command is required")
        if arguments[0] == "pipeline":
            execute_pipeline(arguments[1:])
        elif arguments[0] == "submission-check":
            execute_submission_check(arguments[1:])
        else:
            raise ControllerCLIError("unknown controller command")
    except (ControllerCLIError, SystemExit) as error:
        if isinstance(error, SystemExit) and error.code == 0:
            return 0
        print("execution contract error: {0}".format(error), file=sys.stderr)
        return 2
    return 0


def _pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--fidelity", choices=("smoke", "proxy", "full", "final"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean-reproduce", action="store_true")
    return parser


def _required_directory(environment: Mapping[str, str], name: str) -> Path:
    raw = environment.get(name)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ControllerCLIError("missing controller-owned {0}".format(name))
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ControllerCLIError("{0} must be an absolute real directory".format(name))
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ControllerCLIError("{0} must not traverse symlinks".format(name))
    return resolved


def _required_file(environment: Mapping[str, str], name: str) -> Path:
    raw = environment.get(name)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ControllerCLIError("missing controller-owned {0}".format(name))
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ControllerCLIError("{0} must be an absolute real file".format(name))
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ControllerCLIError("{0} must not traverse symlinks".format(name))
    return resolved


def _new_output_path(raw: str, artifact_root: Path) -> Path:
    if not raw or "\x00" in raw:
        raise ControllerCLIError("invalid prediction output path")
    path = Path(raw)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ControllerCLIError("prediction output must be a new absolute path")
    parent = path.parent.resolve(strict=True)
    try:
        parent.relative_to(artifact_root)
    except ValueError as error:
        raise ControllerCLIError("prediction output escaped the artifact root") from error
    return parent / path.name


def _required_entrypoint(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or _ENTRYPOINT.fullmatch(value) is None:
        raise ControllerCLIError(
            "missing or invalid controller-owned {0}".format(name)
        )
    return value


def _load_entrypoint(value: str) -> Callable[[Any], None]:
    module_name, function_name = value.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, function_name)
    except Exception as error:
        raise ControllerCLIError(
            "configured implementation entrypoint could not be loaded"
        ) from error
    if not callable(implementation):
        raise ControllerCLIError("configured implementation entrypoint is not callable")
    return implementation


if __name__ == "__main__":
    raise SystemExit(main())
