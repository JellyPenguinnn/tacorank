from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tacorank.coding import FakeCodingWorker
from tacorank.execution import FakeExecutionRunner
from tacorank.safety import FakeOutputGate, FakePatchGate


def test_fake_adapters_return_canonical_objects_and_record_calls() -> None:
    patch = SimpleNamespace(patch_commit_sha="a" * 40)
    repaired = SimpleNamespace(patch_commit_sha="b" * 40)
    patch_check = SimpleNamespace(accepted=True)
    run_result = SimpleNamespace(outcome="success")
    output_check = SimpleNamespace(accepted=True)
    context = SimpleNamespace(context_id="context-1")
    spec = SimpleNamespace(experiment_id="experiment-1")
    decision = SimpleNamespace(action="trae_repair")
    request = SimpleNamespace(attempt=1)
    observer = SimpleNamespace()

    coding = FakeCodingWorker(create_result=patch, repair_result=repaired)
    patch_gate = FakePatchGate(patch_check)
    execution = FakeExecutionRunner(run_result)
    output_gate = FakeOutputGate(output_check)

    assert asyncio.run(coding.create_patch(context, spec)) is patch
    assert asyncio.run(coding.repair_patch(context, decision)) is repaired
    assert asyncio.run(patch_gate.check(patch)) is patch_check
    assert asyncio.run(execution.run(request, observer)) is run_result
    assert asyncio.run(output_gate.check(run_result)) is output_check
    assert [call[0] for call in coding.calls] == ["create_patch", "repair_patch"]
    assert patch_gate.calls == [patch]
    assert execution.calls == [(request, observer)]
    assert output_gate.calls == [run_result]


def test_fake_adapter_can_script_a_failure() -> None:
    failure = RuntimeError("scripted failure")
    gate = FakePatchGate(failure)

    with pytest.raises(RuntimeError, match="scripted failure"):
        asyncio.run(gate.check(SimpleNamespace()))
