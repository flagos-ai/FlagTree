# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Register layout contiguity is not a proof of pointer contiguity."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_layout = tl.constexpr(tle.gpu.BlockEncoding([4], [32], [4], [0]))


@triton.jit
def _strided_shared_load(source, output, STEP: tl.constexpr):
    shared = tle.gpu.alloc(
        [512], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    indices = tl.arange(0, 512)
    pointer = tle.gpu.local_ptr(shared, (indices,))
    tl.store(pointer, tl.load(source + indices))
    gather = tle.gpu.set_layout(tl.arange(0, 128), _layout)
    gathered = tle.gpu.local_ptr(shared, (gather * STEP,))
    values = tle.gpu.set_layout(tl.load(gathered), _layout)
    tl.store(output + gather, values)


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.require_tle("gpu.alloc", "gpu.local_ptr")
def test_shared_vector_load_requires_pointer_contiguity(step):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = torch.arange(512, dtype=torch.float32, device="cuda")
    result = torch.empty(128, dtype=torch.float32, device="cuda")
    _strided_shared_load[(1,)](source, result, step)
    torch.testing.assert_close(result, source[:128 * step:step], rtol=0, atol=0)
