"""Checks for launcher-side device timing gating and diagnostics.

Device timing has two independent runtime opt-ins, each read once per launcher
module: ``TILELANG_TX_EVENT_TIMING`` selects TX event timing and
``TSM_PROFILER_EN`` selects the ``libtsm_profiler.so`` query path. Neither is
baked into the native launcher, so the same ``.so`` is valid regardless of the
environment it was first built under.

The runtime adapter reports diagnostics from the launcher's ``timing_kind``
result rather than re-reading the environment.

These tests need no device; they exercise the diagnostics mapping and the
generated C++ launcher source.
"""

from __future__ import annotations

import pytest
from triton.backends.tsingmicro import runtime as tsm_runtime


@pytest.fixture(autouse=True)
def _clean_timing_env(monkeypatch):
    for name in ("TILELANG_TX_EVENT_TIMING", "TSM_PROFILER_EN"):
        monkeypatch.delenv(name, raising=False)


def _driver_module():
    try:
        from triton.backends.tsingmicro import driver
    except Exception as exc:  # noqa: BLE001 - tx8_deps / torch_txda may be absent
        pytest.skip(f"tsingmicro driver unavailable: {exc}")
    return driver


class _FakeLaunchResult:

    def __init__(self, kernel_time_ns: int, timing_kind: int):
        self.kernel_time_ns = kernel_time_ns
        self.timing_kind = timing_kind


@pytest.mark.parametrize("kernel_time_ns,timing_kind,expected", [
    (0, 0, (False, False, "disabled")),
    (1234, 1, (True, True, "tx-event")),
    (0, 1, (True, False, "tx-event")),
    (1234, 2, (True, True, "tsm-profiler")),
    (0, 2, (True, False, "tsm-profiler")),
])
def test_diagnostics_from_launcher_result(kernel_time_ns, timing_kind, expected):
    result = _FakeLaunchResult(kernel_time_ns, timing_kind)
    assert tsm_runtime._describe_timing(result) == expected


def test_native_launcher_is_environment_independent(monkeypatch):
    driver = _driver_module()
    src = "// launcher"

    baseline = driver._native_build_fingerprint(src)
    monkeypatch.setenv("TSM_PROFILER_EN", "1")
    monkeypatch.setenv("TILELANG_TX_EVENT_TIMING", "1")

    # Device timing and the host profiler are runtime decisions: neither may
    # change the compiled artifact.
    assert driver._native_build_fingerprint(src) == baseline
    assert "tx_event_timing" not in baseline


def test_launcher_checks_timing_at_runtime_and_skips_capture():
    driver = _driver_module()
    src = driver.make_launcher(
        {},
        {"x_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        "capture_probe",
        "/tmp/capture_probe.so",
    )

    # Two independent runtime switches, not a compile-time macro.
    assert "static const bool _tx_event_enabled = env_truthy(\"TILELANG_TX_EVENT_TIMING\")" in src
    assert "static const bool _profiler_enabled = env_truthy(\"TSM_PROFILER_EN\")" in src
    assert "#ifdef TSM_PROFILER_EN" not in src
    # Capture is detected on the launch stream itself.
    assert "txStreamIsCapturing(l_args.stream, &_capture_status)" in src
    # Every host-synchronizing path is guarded by the capture check.
    assert "const bool _measure = !_stream_capturing && (_tx_event_enabled || _profiler_enabled);" in src
    assert "if (_measure && _profiler_enabled && _prof_available)" in src
    assert "if (!_stream_capturing && env != nullptr" in src
