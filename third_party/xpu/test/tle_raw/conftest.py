"""Pytest configuration for the `tle.raw` suite.

These tests only exercise the compiler (payload frontend, deferred source store
and the TritonSDNN passes), so they run on CPU-only hosts. The marker is
registered here because the suite no longer sits under `third_party/xpu/test/`,
whose conftest used to provide it -- and whose device check skips everything
unmarked when no XPU is reachable.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_xpu_required: mark a test as runnable without an XPU device.",
    )
