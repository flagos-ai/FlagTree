# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def test_tle_does_not_expose_the_retired_tensor_argument_opt_in():
    assert not hasattr(tle, "jit")


@triton.jit(noinline=True)
def _add_one(value):
    return value + 1


@triton.jit
def _scalar_caller(output):
    value = _add_one(tl.program_id(0))
    tl.store(output + tl.program_id(0), value)


@triton.jit
def _tensor_caller(output):
    value = _add_one(tl.arange(0, 32))
    tl.store(output + tl.arange(0, 32), value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_standard_noinline_scalar_call_still_executes():
    output = torch.empty((4,), dtype=torch.int32, device="cuda")
    kernel = _scalar_caller[(4,)](output)
    torch.testing.assert_close(output, torch.arange(1, 5, device="cuda", dtype=torch.int32))
    assert "tt.call" in kernel.asm["ttir"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_standard_noinline_tensor_arguments_remain_rejected():
    output = torch.empty((32,), dtype=torch.int32, device="cuda")
    with pytest.raises(triton.CompilationError, match="marked noinline.*non-scalar argument"):
        _tensor_caller[(1,)](output)
