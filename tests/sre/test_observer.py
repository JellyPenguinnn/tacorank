from datetime import datetime, timezone
from math import inf, nan

import pytest

from tacorank.schemas import TelemetrySample
from tacorank.sre import (
    HealthPolicyConfig,
    HeartbeatPolicy,
    SREObserver,
    TelemetryWindow,
)


def sample(**overrides):
    values = dict(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_id="run-1",
        experiment_id="exp-1",
        attempt=1,
        elapsed_ms=2_000,
        process_alive=True,
        last_output_age_ms=500,
        cpu_percent=40.0,
        rss_mb=256.0,
        gpu_utilization_percent=None,
        gpu_memory_mb=None,
        loss=1.0,
        gradient_norm=None,
        disk_free_mb=10_000,
        recent_output_tail=None,
    )
    values.update(overrides)
    return TelemetrySample(**values)


def reasons(observer, samples):
    return [observer.observe(item).reason_code for item in samples]


def test_healthy_cpu_baseline_does_not_require_gpu():
    observer = SREObserver()
    directive = observer.observe(sample())
    assert directive.action == "continue"
    assert directive.reason_code is None


def test_healthy_gpu_run_tolerates_intermittent_low_utilization():
    observer = SREObserver(
        config=HealthPolicyConfig(
            heartbeat=HeartbeatPolicy(deadline_ms=10_000, gpu_expected=True)
        )
    )
    directives = [
        observer.observe(sample(gpu_utilization_percent=utilization))
        for utilization in (70.0, 0.0, 65.0)
    ]
    assert all(item.action == "continue" for item in directives)


def test_process_death_terminates_immediately():
    directive = SREObserver().observe(sample(process_alive=False))
    assert (directive.action, directive.reason_code) == ("terminate", "PROCESS_DIED")


@pytest.mark.parametrize("metric,value", [("loss", nan), ("loss", inf), ("gradient_norm", -inf)])
def test_nonfinite_metric_terminates_immediately(metric, value):
    values = sample().model_dump() if hasattr(TelemetrySample, "model_construct") else vars(sample())
    values[metric] = value
    # The observer remains defensive even when a collector bypasses ordinary
    # Pydantic validation while parsing a raw non-finite training metric.
    anomalous = (
        TelemetrySample.model_construct(**values)
        if hasattr(TelemetrySample, "model_construct")
        else TelemetrySample(**values)
    )
    directive = SREObserver().observe(anomalous)
    assert directive.reason_code == "NUMERICAL_NONFINITE"


def test_stale_output_with_idle_cpu_and_gpu_becomes_hang():
    observer = SREObserver(
        config=HealthPolicyConfig(
            heartbeat=HeartbeatPolicy(
                deadline_ms=10_000, minimum_stale_samples=2, gpu_expected=True
            )
        )
    )
    first = observer.observe(
        sample(last_output_age_ms=11_000, cpu_percent=1.0, gpu_utilization_percent=0.0)
    )
    second = observer.observe(
        sample(last_output_age_ms=13_000, cpu_percent=1.2, gpu_utilization_percent=1.0)
    )
    assert first.action == "continue"
    assert second.reason_code == "HEARTBEAT_STALE"


def test_long_compute_with_active_resources_is_not_a_hang():
    observer = SREObserver(
        config=HealthPolicyConfig(
            heartbeat=HeartbeatPolicy(deadline_ms=10_000, gpu_expected=True)
        )
    )
    directives = [
        observer.observe(
            sample(
                last_output_age_ms=age,
                cpu_percent=80.0,
                gpu_utilization_percent=90.0,
            )
        )
        for age in (11_000, 13_000, 15_000)
    ]
    assert all(item.action == "continue" for item in directives)


def test_missing_expected_gpu_signal_does_not_guess_hang():
    observer = SREObserver(
        config=HealthPolicyConfig(
            heartbeat=HeartbeatPolicy(deadline_ms=10_000, gpu_expected=True)
        )
    )
    directives = [
        observer.observe(sample(last_output_age_ms=age, cpu_percent=0.0))
        for age in (11_000, 13_000)
    ]
    assert all(item.action == "continue" for item in directives)


def test_single_loss_spike_is_tolerated_but_persistent_spike_terminates():
    observer = SREObserver(
        config=HealthPolicyConfig(anomaly_baseline_samples=3, anomaly_persistence=2)
    )
    observed = reasons(
        observer,
        [sample(loss=value) for value in (1.0, 1.1, 0.9, 7.0, 7.5)],
    )
    assert observed[-2] is None
    assert observed[-1] == "LOSS_EXPLOSION"


def test_gradient_detector_is_disabled_until_metric_is_reliably_emitted():
    observer = SREObserver(
        config=HealthPolicyConfig(anomaly_baseline_samples=3, anomaly_persistence=2)
    )
    observed = reasons(
        observer,
        [
            sample(loss=None, gradient_norm=value)
            for value in (None, 1.0, 1.1, 0.9, 8.0, 8.5)
        ],
    )
    assert observed[-1] == "GRADIENT_EXPLOSION"


@pytest.mark.parametrize(
    "tail",
    ["RuntimeError: CUDA out of memory", "MemoryError", "cannot allocate memory"],
)
def test_explicit_oom_signatures_are_detected_without_leaking_tail(tail):
    directive = SREObserver().observe(
        sample(recent_output_tail=f"token=secret /credential/path {tail}")
    )
    assert directive.reason_code == "EXPLICIT_OOM"
    assert "secret" not in directive.summary
    assert "/credential" not in directive.summary


@pytest.mark.parametrize(
    "overrides,config,reason",
    [
        ({"rss_mb": 501.0}, HealthPolicyConfig(cpu_memory_limit_mb=500), "CPU_MEMORY_LIMIT"),
        (
            {"gpu_memory_mb": 1_025},
            HealthPolicyConfig(gpu_memory_limit_mb=1_024),
            "GPU_MEMORY_LIMIT",
        ),
        ({"disk_free_mb": 99}, HealthPolicyConfig(disk_free_floor_mb=100), "DISK_LOW"),
    ],
)
def test_resource_limits(overrides, config, reason):
    assert SREObserver(config=config).observe(sample(**overrides)).reason_code == reason


def test_execution_policy_violation_is_redacted():
    directive = SREObserver().observe(
        sample(recent_output_tail="protected path violation at C:/keys/token-secret")
    )
    assert directive.reason_code == "EXECUTION_POLICY_VIOLATION"
    assert "keys" not in directive.summary


def test_window_is_bounded_and_keeps_order():
    window = TelemetryWindow(capacity=3)
    for elapsed in range(5):
        window.add(sample(elapsed_ms=elapsed))
    assert len(window) == 3
    assert [item.elapsed_ms for item in window] == [2, 3, 4]
    assert window.latest.elapsed_ms == 4


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_window_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        TelemetryWindow(capacity)


def test_policy_rejects_invalid_anomaly_configuration_eagerly():
    with pytest.raises(ValueError):
        HealthPolicyConfig(anomaly_persistence=1)
