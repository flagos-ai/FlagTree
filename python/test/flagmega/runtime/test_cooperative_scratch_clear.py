# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A cooperative-grid launch must start from a zero-initialized scratch.

The AMD launcher clears the global scratch via ``zero_``/``fill_`` when the
buffer exposes them. A buffer that exposes neither must explicitly declare it is
already zero-initialized (``pre_zeroed``); otherwise the launch is refused rather
than running the sense-reversing grid barrier against a possibly-garbage counter.
"""

import pytest

driver = pytest.importorskip("triton.backends.amd.driver")
_clear = driver._clear_cooperative_grid_scratch


class _Zeroable:
    def __init__(self):
        self.zeroed = False

    def zero_(self):
        self.zeroed = True


class _Fillable:
    def __init__(self):
        self.filled = None

    def fill_(self, value):
        self.filled = value


class _PreZeroed:
    pre_zeroed = True


class _UnknownAllocatorBuffer:
    """A non-tensor scratch buffer: no zero_/fill_ and no pre_zeroed marker."""


def test_clear_uses_zero_():
    buffer = _Zeroable()
    _clear(buffer)
    assert buffer.zeroed is True


def test_clear_uses_fill_when_no_zero_():
    buffer = _Fillable()
    _clear(buffer)
    assert buffer.filled == 0


def test_pre_zeroed_buffer_skips_clear():
    _clear(_PreZeroed())  # no exception, no clear needed


def test_non_tensor_allocator_without_marker_is_refused():
    with pytest.raises(RuntimeError):
        _clear(_UnknownAllocatorBuffer())


def test_prepared_scratch_view_declares_pre_zeroed():
    from triton.flagmega.runtime.prepared import _ScratchBufferView

    assert getattr(_ScratchBufferView, "pre_zeroed", False) is True
    # The prepared view hides zero_/fill_ on purpose; pre_zeroed lets the
    # launcher skip the redundant clear instead of refusing the launch.
    _clear(_ScratchBufferView.__new__(_ScratchBufferView))
