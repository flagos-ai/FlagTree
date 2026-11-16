# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static validation of the explicit ring source-to-module configuration API."""
from types import SimpleNamespace
import pytest
import triton.language as tl
import triton.experimental.tle.language as tle


def test_trace_configure_export():
    assert callable(tle.gpu.trace_configure)


class Recorder:
    def __init__(self):
        self.calls = []

    def configure_tle_ws_trace(self, capacity, ctas):
        self.calls.append((capacity, ctas))


@pytest.mark.parametrize('capacity,ctas,expected', [
    (1, None, None), (1048576, [0, 127], [0, 127]),
    (tl.constexpr(64), tl.constexpr((0, 15, 127)), [0, 15, 127]),
    (64, tl.tuple((tl.constexpr(0), tl.constexpr(127))), [0, 127]),
], ids=['runtime_only', 'limits', 'constexpr', 'triton_tuple'])
def test_static_configuration_normalizes_exact_values(capacity, ctas, expected):
    builder = Recorder()
    tle.gpu.trace_configure(capacity, ctas, _semantic=SimpleNamespace(builder=builder))
    assert builder.calls == [(tl.core._unwrap_if_constexpr(capacity), expected)]


@pytest.mark.parametrize('capacity,ctas', [
    (True, [0]), (1.0, [0]), (0, [0]), (-1, [0]), (1048577, [0]),
    (64, []), (64, [0, 0]), (64, [2, 1]), (64, [-1]),
    (64, [2147483648]), (64, [False]), (64, [1.0]), (64, 1), (64, '0'),
])
def test_invalid_values_reject_before_builder_mutation(capacity, ctas):
    builder = Recorder()
    with pytest.raises(ValueError):
        tle.gpu.trace_configure(capacity, ctas, _semantic=SimpleNamespace(builder=builder))
    assert builder.calls == []


def test_missing_builder_rejects_explicitly():
    with pytest.raises(ValueError, match='rebuilt.*TLE'):
        tle.gpu.trace_configure(64, [0], _semantic=SimpleNamespace(builder=object()))
